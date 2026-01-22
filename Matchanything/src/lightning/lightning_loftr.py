from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from Matchanything.src.loftr import LoFTR
from Matchanything.src.loftr.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
    compute_roma_supervision,
)
from Matchanything.src.optimizers import build_optimizer, build_scheduler
from Matchanything.src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    compute_homo_corner_warp_errors,
    compute_homo_match_warp_errors,
    compute_warp_control_pts_errors,
    aggregate_metrics,
)
from Matchanything.src.utils.plotting import make_matching_figures, make_scores_figures
from Matchanything.src.utils.comm import gather, all_gather
from Matchanything.src.utils.misc import lower_config, flattenList
from Matchanything.src.utils.profiler import PassThroughProfiler
from Matchanything.third_party.ROMA.roma.matchanything_roma_model import (
    MatchAnything_Model,
)

# import pynvml


def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, "switch_to_deploy"):
        module.switch_to_deploy()
        print("m0 switch to deploy ok")
    for modules in [
        matcher.backbone.layer1,
        matcher.backbone.layer2,
        matcher.backbone.layer3,
    ]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
                print("backbone switch to deploy ok")
    for modules in [
        matcher.fine_preprocess.layer2_outconv2,
        matcher.fine_preprocess.layer1_outconv2,
    ]:
        for module in modules:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
                print("fpn switch to deploy ok")
    return matcher


class PL_LoFTR(pl.LightningModule):
    def __init__(
        self,
        config,
        pretrained_ckpt=None,
        profiler=None,
        dump_dir=None,
        test_mode=False,
        baseline_config=None,
    ):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        if config.METHOD == "matchanything_eloftr":
            self.matcher = LoFTR(config=_config["loftr"], profiler=self.profiler)
        elif config.METHOD == "matchanything_roma":
            self.matcher = MatchAnything_Model(
                config=_config["roma"], test_mode=test_mode
            )
        elif config.METHOD == "minima_roma":
            self.matcher = MatchAnything_Model(
                config=_config["roma"], test_mode=test_mode
            )
        else:
            raise NotImplementedError

        if config.FINETUNE.ENABLE and test_mode:
            # Inference time change model architecture before load pretrained model:
            raise NotImplementedError

        # Pretrained weights
        if pretrained_ckpt:
            if config.METHOD in ["matchanything_eloftr", "matchanything_roma"]:
                state_dict = torch.load(pretrained_ckpt, map_location="cpu")[
                    "state_dict"
                ]
                logger.info(
                    f"Load model from:{self.matcher.load_state_dict(state_dict, strict=False)}"
                )
            elif config.METHOD in ["minima_roma"]:
                state_dict = torch.load(pretrained_ckpt, map_location="cpu")
                logger.info(
                    f"Load model from:{self.matcher.load_state_dict(state_dict, strict=False)}"
                )
                # state_dict = ckpt['model']
            else:
                raise NotImplementedError

        if (
            self.config.LOFTR.BACKBONE_TYPE == "RepVGG"
            and test_mode
            and (config.METHOD == "loftr")
        ):
            module = self.matcher.backbone.layer0
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
                print("m0 switch to deploy ok")
            for modules in [
                self.matcher.backbone.layer1,
                self.matcher.backbone.layer2,
                self.matcher.backbone.layer3,
            ]:
                for module in modules:
                    if hasattr(module, "switch_to_deploy"):
                        module.switch_to_deploy()
                        print("m switch to deploy ok")

        # Testing
        self.dump_dir = dump_dir
        self.max_gpu_memory = 0
        self.GPUID = 0
        self.warmup = False

    #     def gpumem(self, des, gpuid=None):
    #         NUM_EXPAND = 1024 * 1024 * 1024
    #         gpu_id = self.GPUID if self.GPUID is not None else gpuid
    #         handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    #         info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #         gpu_Used = info.used
    #         logger.info(f"GPU {gpu_id} memory used: {gpu_Used / NUM_EXPAND} GB while {des}")
    #         # print(des, gpu_Used / NUM_EXPAND)
    #         if gpu_Used / NUM_EXPAND > self.max_gpu_memory:
    #             self.max_gpu_memory = gpu_Used / NUM_EXPAND
    #             logger.info(
    #                 f"[MAX]GPU {gpu_id} memory used: {gpu_Used / NUM_EXPAND} GB while {des}"
    #             )
    #             print("max_gpu_memory", self.max_gpu_memory)

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )

        # update params
        if self.config.LOFTR.FP16:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):

            with torch.autocast(enabled=False, device_type="cuda"):
                if ("roma" in self.config.METHOD) or ("dkm" in self.config.METHOD):
                    pass
                else:
                    compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("LoFTR"):
            with torch.autocast(enabled=self.config.LOFTR.FP16, device_type="cuda"):
                self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=False, device_type="cuda"):
                if ("roma" in self.config.METHOD) or ("dkm" in self.config.METHOD):
                    compute_roma_supervision(batch, self.config)
                else:
                    compute_supervision_fine(batch, self.config, self.logger)

        with self.profiler.profile("Compute losses"):
            pass

    def _compute_metrics(self, batch):
        if "gt_2D_matches" in batch:
            compute_warp_control_pts_errors(batch, self.config)
        elif batch["homography"].sum() != 0 and batch["T_0to1"].sum() == 0:
            compute_homo_match_warp_errors(
                batch, self.config
            )  # compute warp_errors for each match
            compute_homo_corner_warp_errors(
                batch, self.config
            )  # compute mean corner warp error each pair
        else:
            compute_symmetrical_epipolar_errors(
                batch, self.config
            )  # compute epi_errs for each match
            compute_pose_errors(
                batch, self.config
            )  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].size(0)
        if self.config.LOFTR.FINE.MTD_SPVS:
            topk = self.config.LOFTR.MATCH_FINE.TOPK
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    (batch["epi_errs"].reshape(-1, topk))[batch["m_bids"] == b]
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
                "num_matches": [batch["mconf"].shape[0]],  # batch size = 1 only
                "percent_inliers": [
                    (
                        batch["inliers"][0].shape[0] / batch["mconf"].shape[0]
                        if batch["mconf"].shape[0] != 0
                        else 1
                    )
                ],  # batch size = 1 only
            }
        else:
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    batch["epi_errs"][batch["m_bids"] == b].cpu().numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
                "num_matches": [batch["mconf"].shape[0]],  # batch size = 1 only
                "percent_inliers": [
                    (
                        batch["inliers"][0].shape[0] / batch["mconf"].shape[0]
                        if batch["mconf"].shape[0] != 0
                        else 1
                    )
                ],  # batch size = 1 only
            }
        ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

            # net-params
            method = "LOFTR"
            if self.config[method]["MATCH_COARSE"]["MATCH_TYPE"] == "sinkhorn":
                self.logger.experiment.add_scalar(
                    f"skh_bin_score",
                    self.matcher.coarse_matching.bin_score.clone().detach().cpu().data,
                    self.global_step,
                )

            figures = {}
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(
                    batch, self.config
                )  # compute epi_errs for each match
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.add_figure(
                        f"train_match/{k}", v, self.global_step
                    )

        return {"loss": batch["loss"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                "train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )
            if self.config.LOFTR.PLOT_SCORES:
                figs = make_scores_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                figures[self.config.TRAINER.PLOT_MODE] += figs[
                    self.config.TRAINER.PLOT_MODE
                ]
                del figs

        return {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            "figures": figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = (
            [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if (
                not self.trainer.resume_from_checkpoint
                and self.trainer.running_sanity_check
            ):
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {
                k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, self.config.LOFTR.EVAL_TIMES
            )
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(
                        f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch
                    )

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f"val_match_{valset_idx}/{k}/pair-{plot_idx}",
                                fig,
                                cur_epoch,
                                close=True,
                            )
            plt.close("all")

        for thr in [5, 10, 20]:
            self.log(
                f"auc@{thr}", torch.tensor(np.mean(multi_val_metrics[f"auc@{thr}"]))
            )  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        if self.warmup:
            for i in range(50):
                self.matcher(batch)
            self.warmup = False

        with torch.autocast(enabled=self.config.LOFTR.FP16, device_type="cuda"):
            with self.profiler.profile("LoFTR"):
                self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        print(ret_dict["metrics"]["num_matches"])
        self.dump_dir = None

        return ret_dict

    def test_epoch_end(self, outputs):
        print(self.config)
        print("max GPU memory: ", self.max_gpu_memory)
        print(self.profiler.summary())
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        if self.trainer.global_rank == 0:
            NUM_EXPAND = 1024 * 1024 * 1024
            gpu_id = self.GPUID
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_Used = info.used
            print("pynvml", gpu_Used / NUM_EXPAND)
            if gpu_Used / NUM_EXPAND > self.max_gpu_memory:
                self.max_gpu_memory = gpu_Used / NUM_EXPAND

            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(
                metrics,
                self.config.TRAINER.EPI_ERR_THR,
                self.config.LOFTR.EVAL_TIMES,
                self.config.TRAINER.THRESHOLDS,
                method=self.config.TRAINER.AUC_METHOD,
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "LoFTR_pred_eval", dumps)
