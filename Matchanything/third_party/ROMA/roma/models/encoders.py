from typing import Optional, Union
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc


class ResNet50(nn.Module):
    def __init__(
        self,
        pretrained=False,
        high_res=False,
        weights=None,
        dilation=None,
        freeze_bn=True,
        anti_aliased=False,
        early_exit=False,
        amp=False,
        amp_dtype=torch.float16,
    ) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(
                    weights=weights, replace_stride_with_dilation=dilation
                )
            else:
                net = tvm.resnet50(
                    pretrained=pretrained, replace_stride_with_dilation=dilation
                )
                self.net = nn.Sequential(
                    net.conv1,
                    net.bn1,
                    net.relu,
                    net.maxpool,
                    net.layer1,
                    net.layer2,
                    net.layer3,
                )

        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
            # net = self.net
            # feats = {1:x}
            # x = net.conv1(x)
            # x = net.bn1(x)
            # x = net.relu(x)
            # feats[2] = x
            # x = net.maxpool(x)
            # x = net.layer1(x)
            # feats[4] = x
            # x = net.layer2(x)
            # feats[8] = x
            # if self.early_exit:
            #     return feats
            # x = net.layer3(x)
            # feats[16] = x
            # x = net.layer4(x)
            # feats[32] = x
            return self.net(x)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass


class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp=False, amp_dtype=torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale * 2
                x = layer(x)
            return feats


class CNNandDinov2(nn.Module):
    def __init__(
        self,
        cnn_kwargs=None,
        amp=False,
        use_vgg=False,
        coarse_backbone="DINOv2_large",
        coarse_patch_size=14,
        coarse_feat_dim=1024,
        dinov2_weights=None,
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.coarse_backbone = coarse_backbone
        self.coarse_patch_size = coarse_patch_size
        self.coarse_feat_dim = coarse_feat_dim
        if "DINOv2" in coarse_backbone:
            if "large" in coarse_backbone:
                if dinov2_weights is None:
                    dinov2_weights = torch.hub.load_state_dict_from_url(
                        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                        map_location="cpu",
                    )
                from .transformer import vit_large as vit_model

                vit_kwargs = dict(
                    img_size=518,
                    patch_size=coarse_patch_size,
                    init_values=1.0,
                    ffn_layer="mlp",
                    block_chunks=0,
                )
            else:
                raise NotImplementedError

            dinov2_vitl14 = vit_model(**vit_kwargs).eval()
            dinov2_vitl14.load_state_dict(dinov2_weights)

            if self.amp:
                dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
            self.dinov2_vitl14 = [
                dinov2_vitl14
            ]  # ugly hack to not show parameters to DDP
        elif coarse_backbone == "ResNet50":
            self.backbone_model = ResNet50(pretrained=True, amp=self.amp)
        else:
            raise NotImplementedError

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)

    def train(self, mode: bool = True):
        return self.cnn.train(mode)

    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        feature_pyramid = self.cnn(x)

        if not upsample:
            with torch.no_grad():
                if "DINOv2" in self.coarse_backbone:
                    if self.dinov2_vitl14[0].device != x.device:
                        self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
                    dinov2_features_16 = self.dinov2_vitl14[0].forward_features(
                        x.to(self.amp_dtype) if self.amp else x
                    )
                    features_16 = (
                        dinov2_features_16["x_norm_patchtokens"]
                        .permute(0, 2, 1)
                        .reshape(
                            B,
                            self.coarse_feat_dim,
                            H // self.coarse_patch_size,
                            W // self.coarse_patch_size,
                        )
                    )
                    del dinov2_features_16
                else:
                    raise NotImplementedError
            if self.coarse_backbone == "ResNet50":
                features_16 = self.backbone_model(
                    x.to(self.amp_dtype) if self.amp else x
                )
            feature_pyramid[16] = features_16
        return feature_pyramid
