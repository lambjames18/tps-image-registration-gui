from yacs.config import CfgNode as CN

_CN = CN()
############## ROMA Pipeline #########
_CN.ROMA = CN()
_CN.ROMA.MATCH_THRESH = 0.0
_CN.ROMA.RESIZE_BY_STRETCH = False  # Used for test mode
_CN.ROMA.NORMALIZE_IMG = False  # Used for test mode

_CN.ROMA.MODE = "train_framework"  # Used in Lightning Train & Val
_CN.ROMA.MODEL = CN()
_CN.ROMA.MODEL.COARSE_BACKBONE = "DINOv2_large"
_CN.ROMA.MODEL.COARSE_FEAT_DIM = 1024
_CN.ROMA.MODEL.MEDIUM_FEAT_DIM = 512
_CN.ROMA.MODEL.COARSE_PATCH_SIZE = 14
_CN.ROMA.MODEL.AMP = True  # FP16 mode

_CN.ROMA.SAMPLE = CN()
_CN.ROMA.SAMPLE.METHOD = "threshold_balanced"
_CN.ROMA.SAMPLE.N_SAMPLE = 5000
_CN.ROMA.SAMPLE.THRESH = 0.05

_CN.ROMA.TEST_TIME = CN()
_CN.ROMA.TEST_TIME.COARSE_RES = (560, 560)  # need to divisable by 14 & 8
_CN.ROMA.TEST_TIME.UPSAMPLE = True
_CN.ROMA.TEST_TIME.UPSAMPLE_RES = (864, 864)  # need to divisable by 8
_CN.ROMA.TEST_TIME.SYMMETRIC = True
_CN.ROMA.TEST_TIME.ATTENUTATE_CERT = True

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.LOFTR = CN()
_CN.LOFTR.BACKBONE_TYPE = "ResNetFPN"
_CN.LOFTR.ALIGN_CORNER = True
_CN.LOFTR.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.LOFTR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.LOFTR.FINE_WINDOW_MATCHING_SIZE = (
    5  # window_size for loftr fine-matching, odd for select and even for average
)
_CN.LOFTR.FINE_CONCAT_COARSE_FEAT = True
_CN.LOFTR.FINE_SAMPLE_COARSE_FEAT = False
_CN.LOFTR.COARSE_FEAT_ONLY = False  # TO BE DONE
_CN.LOFTR.INTER_FEAT = False  # FPN backbone inter feat with coarse_attn.
_CN.LOFTR.FP16 = False
_CN.LOFTR.FIX_BIAS = False
_CN.LOFTR.MATCHABILITY = False
_CN.LOFTR.FORCE_LOOP_BACK = False
_CN.LOFTR.NORM_FPNFEAT = False
_CN.LOFTR.NORM_FPNFEAT2 = False
_CN.LOFTR.REPLACE_NAN = False
_CN.LOFTR.PLOT_SCORES = False
_CN.LOFTR.REP_FPN = False
_CN.LOFTR.REP_DEPLOY = False
_CN.LOFTR.EVAL_TIMES = 1

# 1. LoFTR-backbone (local feature CNN) config
_CN.LOFTR.RESNETFPN = CN()
_CN.LOFTR.RESNETFPN.INITIAL_DIM = 128
_CN.LOFTR.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3
_CN.LOFTR.RESNETFPN.SAMPLE_FINE = False
_CN.LOFTR.RESNETFPN.COARSE_FEAT_ONLY = False  # TO BE DONE
_CN.LOFTR.RESNETFPN.INTER_FEAT = False  # FPN backbone inter feat with coarse_attn.
_CN.LOFTR.RESNETFPN.LEAKY = False
_CN.LOFTR.RESNETFPN.REPVGGMODEL = None

# 2. LoFTR-coarse module config
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.COARSE.D_MODEL = 256
_CN.LOFTR.COARSE.D_FFN = 256
_CN.LOFTR.COARSE.NHEAD = 8
_CN.LOFTR.COARSE.LAYER_NAMES = ["self", "cross"] * 4
_CN.LOFTR.COARSE.ATTENTION = "linear"  # options: ['linear', 'full']
_CN.LOFTR.COARSE.TEMP_BUG_FIX = True
_CN.LOFTR.COARSE.NPE = [832, 832, 832, 832]
_CN.LOFTR.COARSE.PAN = False
_CN.LOFTR.COARSE.POOl_SIZE = 4
_CN.LOFTR.COARSE.POOl_SIZE2 = 4
_CN.LOFTR.COARSE.BN = True
_CN.LOFTR.COARSE.XFORMER = False
_CN.LOFTR.COARSE.BIDIRECTION = False
_CN.LOFTR.COARSE.DEPTH_CONFIDENCE = -1.0
_CN.LOFTR.COARSE.WIDTH_CONFIDENCE = -1.0
_CN.LOFTR.COARSE.LEAKY = -1.0
_CN.LOFTR.COARSE.ASYMMETRIC = False
_CN.LOFTR.COARSE.ASYMMETRIC_SELF = False
_CN.LOFTR.COARSE.ROPE = False
_CN.LOFTR.COARSE.TOKEN_MIXER = None
_CN.LOFTR.COARSE.SKIP = False
_CN.LOFTR.COARSE.DWCONV = False
_CN.LOFTR.COARSE.DWCONV2 = False
_CN.LOFTR.COARSE.SCATTER = False
_CN.LOFTR.COARSE.ROPE = False
_CN.LOFTR.COARSE.NPE = None
_CN.LOFTR.COARSE.NORM_BEFORE = True
_CN.LOFTR.COARSE.VIT_NORM = False
_CN.LOFTR.COARSE.ROPE_DWPROJ = False
_CN.LOFTR.COARSE.ABSPE = False


# 3. Coarse-Matching config
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_COARSE.THR = 0.1
_CN.LOFTR.MATCH_COARSE.BORDER_RM = 2
_CN.LOFTR.MATCH_COARSE.MATCH_TYPE = (
    "dual_softmax"  # options: ['dual_softmax, 'sinkhorn']
)
_CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFTR.MATCH_COARSE.SKH_ITERS = 3
_CN.LOFTR.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.LOFTR.MATCH_COARSE.SKH_PREFILTER = False
_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_COARSE.MTD_SPVS = False
_CN.LOFTR.MATCH_COARSE.FIX_BIAS = False
_CN.LOFTR.MATCH_COARSE.BINARY = False
_CN.LOFTR.MATCH_COARSE.BINARY_SPV = "l2"
_CN.LOFTR.MATCH_COARSE.NORMFEAT = False
_CN.LOFTR.MATCH_COARSE.NORMFEATMUL = False
_CN.LOFTR.MATCH_COARSE.DIFFSIGN2 = False
_CN.LOFTR.MATCH_COARSE.DIFFSIGN3 = False
_CN.LOFTR.MATCH_COARSE.CLASSIFY = False
_CN.LOFTR.MATCH_COARSE.D_CLASSIFY = 256
_CN.LOFTR.MATCH_COARSE.SKIP_SOFTMAX = False
_CN.LOFTR.MATCH_COARSE.FORCE_NEAREST = False  # in case binary is True, force nearest neighbor, preventing finding a reasonable threshold
_CN.LOFTR.MATCH_COARSE.FP16MATMUL = False
_CN.LOFTR.MATCH_COARSE.SEQSOFTMAX = False
_CN.LOFTR.MATCH_COARSE.SEQSOFTMAX2 = False
_CN.LOFTR.MATCH_COARSE.RATIO_TEST = False
_CN.LOFTR.MATCH_COARSE.RATIO_TEST_VAL = -1.0
_CN.LOFTR.MATCH_COARSE.USE_GT_COARSE = False
_CN.LOFTR.MATCH_COARSE.CROSS_SOFTMAX = False
_CN.LOFTR.MATCH_COARSE.PLOT_ORIGIN_SCORES = False
_CN.LOFTR.MATCH_COARSE.USE_PERCENT_THR = False
_CN.LOFTR.MATCH_COARSE.PERCENT_THR = 0.1
_CN.LOFTR.MATCH_COARSE.ADD_SIGMOID = False
_CN.LOFTR.MATCH_COARSE.SIGMOID_BIAS = 20.0
_CN.LOFTR.MATCH_COARSE.SIGMOID_SIGMA = 2.5
_CN.LOFTR.MATCH_COARSE.CAL_PER_OF_GT = False

# 4. LoFTR-fine module config
_CN.LOFTR.FINE = CN()
_CN.LOFTR.FINE.SKIP = False
_CN.LOFTR.FINE.D_MODEL = 128
_CN.LOFTR.FINE.D_FFN = 128
_CN.LOFTR.FINE.NHEAD = 8
_CN.LOFTR.FINE.LAYER_NAMES = ["self", "cross"] * 1
_CN.LOFTR.FINE.ATTENTION = "linear"
_CN.LOFTR.FINE.MTD_SPVS = False
_CN.LOFTR.FINE.PAN = False
_CN.LOFTR.FINE.POOl_SIZE = 4
_CN.LOFTR.FINE.BN = True
_CN.LOFTR.FINE.XFORMER = False
_CN.LOFTR.FINE.BIDIRECTION = False


# Fine-Matching config
_CN.LOFTR.MATCH_FINE = CN()
_CN.LOFTR.MATCH_FINE.THR = 0
_CN.LOFTR.MATCH_FINE.TOPK = 3
_CN.LOFTR.MATCH_FINE.NORMFINEM = False
_CN.LOFTR.MATCH_FINE.USE_GT_FINE = False
_CN.LOFTR.MATCH_COARSE.FINE_TOPK = _CN.LOFTR.MATCH_FINE.TOPK
_CN.LOFTR.MATCH_FINE.FIX_FINE_MATCHING = False
_CN.LOFTR.MATCH_FINE.SKIP_FINE_SOFTMAX = False
_CN.LOFTR.MATCH_FINE.USE_SIGMOID = False
_CN.LOFTR.MATCH_FINE.SIGMOID_BIAS = 0.0
_CN.LOFTR.MATCH_FINE.NORMFEAT = False
_CN.LOFTR.MATCH_FINE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_FINE.FORCE_NEAREST = False
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS = False
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_RMBORDER = False
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_NOMASK = False
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 1.0
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_PADONE = False
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICE = False
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_INNER = False
_CN.LOFTR.MATCH_FINE.MULTI_REGRESS = False


# 5. LoFTR Losses
# -- # coarse-level
_CN.LOFTR.LOSS = CN()
_CN.LOFTR.LOSS.COARSE_TYPE = "focal"  # ['focal', 'cross_entropy']
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
_CN.LOFTR.LOSS.COARSE_SIGMOID_WEIGHT = 1.0
_CN.LOFTR.LOSS.LOCAL_WEIGHT = 0.5
_CN.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = False
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = False
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT2 = False
# _CN.LOFTR.LOSS.SPARSE_SPVS = False
# -- - -- # focal loss (coarse)
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0
_CN.LOFTR.LOSS.CORRECT_NEG_WEIGHT = False
# _CN.LOFTR.LOSS.DUAL_SOFTMAX = False  # whether coarse-level use dual-softmax or not.
# use `_CN.LOFTR.MATCH_COARSE.MATCH_TYPE`

# -- # fine-level
_CN.LOFTR.LOSS.FINE_TYPE = "l2_with_std"  # ['l2_with_std', 'l2']
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)

# -- # ROMA:
_CN.LOFTR.ROMA_LOSS = CN()
_CN.LOFTR.ROMA_LOSS.IGNORE_EMPTY_IN_SPARSE_MATCH_SPV = False  # ['l2_with_std', 'l2']

# -- # DKM:
_CN.LOFTR.DKM_LOSS = CN()
_CN.LOFTR.DKM_LOSS.IGNORE_EMPTY_IN_SPARSE_MATCH_SPV = False  # ['l2_with_std', 'l2']

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TB_LOG_DIR = "logs/tb_logs"  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_SAMPLE_RATIO = [1.0]  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = (
    None  # None if val data from all scenes are bundled into a single npz file
)
_CN.DATASET.VAL_INTRINSIC_PATH = None
_CN.DATASET.FP16 = False
_CN.DATASET.TRAIN_GT_MATCHES_PADDING_N = 8000
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = (
    None  # None if test data from all scenes are bundled into a single npz file
)
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = (
    0.4  # discard data with overlap_score < min_overlap_score
)
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# debug options
_CN.DATASET.TEST_N_PAIRS = None  # Debug first N pairs
# DEBUG
_CN.LOFTR.FP16LOG = False
_CN.LOFTR.MATCH_COARSE.FP16LOG = False

# scanNet options
_CN.DATASET.SCAN_IMG_RESIZEX = (
    640  # resize the longer side, zero-pad bottom-right to square.
)
_CN.DATASET.SCAN_IMG_RESIZEY = (
    480  # resize the shorter side, zero-pad bottom-right to square.
)

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = (
    640,
    640,
)  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8
_CN.DATASET.LOAD_ORIGIN_RGB = False  # Only open in test mode, useful for RGB required baselines such as DKM, ROMA.
_CN.DATASET.READ_GRAY = True
_CN.DATASET.RESIZE_BY_STRETCH = False
_CN.DATASET.NORMALIZE_IMG = (
    False  # For backbone using pretrained DINO feats, use True may be better.
)
_CN.DATASET.HOMO_WARP_USE_MASK = False

_CN.DATASET.NPE_NAME = "megadepth"

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.OPTIMIZER_EPS = (
    1e-8  # Default for optimizers, but set smaller, e.g., 1e-7 for fp16 mix training
)
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.0  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = "linear"  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.0
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = "MultiStepLR"  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = "epoch"  # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 8  # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = "evaluation"  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = "dynamic"

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = (
    5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
)
_CN.TRAINER.POSE_GEO_MODEL = "E"  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = "RANSAC"  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.WARP_ESTIMATOR_MODEL = "affine"  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False
_CN.TRAINER.THRESHOLDS = [5, 10, 20]

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = (
    "scene_balance"  # options: ['scene_balance', 'random', 'normal']
)
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = (
    True  # whether sample each scene with replacement or not
)
_CN.TRAINER.SB_SUBSET_SHUFFLE = (
    True  # after sampling from scenes, whether shuffle within the epoch or not
)
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
_CN.TRAINER.AUC_METHOD = "exact_auc"
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# Finetune Mode:
_CN.FINETUNE = CN()
_CN.FINETUNE.ENABLE = False
_CN.FINETUNE.METHOD = "lora"  # ['lora', 'whole_network']

_CN.FINETUNE.LORA = CN()
_CN.FINETUNE.LORA.RANK = 2
_CN.FINETUNE.LORA.MODE = "linear&conv"  # ["linear&conv", "linear_only"]
_CN.FINETUNE.LORA.SCALE = 1.0

_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
