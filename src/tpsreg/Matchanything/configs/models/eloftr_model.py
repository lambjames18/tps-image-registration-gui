from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1

cfg.TRAINER.MSLR_MILESTONES = [4, 6, 8, 10, 12, 14, 16]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1

cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.1

cfg.LOFTR.MATCH_COARSE.MTD_SPVS = True
cfg.LOFTR.FINE.MTD_SPVS = True

cfg.LOFTR.RESOLUTION = (8, 1)  # options: [(8, 2), (16, 4)]
cfg.LOFTR.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be odd
cfg.LOFTR.MATCH_FINE.THR = 0
cfg.LOFTR.LOSS.FINE_TYPE = 'l2'  # ['l2_with_std', 'l2']

cfg.TRAINER.EPI_ERR_THR = 5e-4 # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = True

# PAN
cfg.LOFTR.COARSE.PAN = True
cfg.LOFTR.COARSE.POOl_SIZE = 4
cfg.LOFTR.COARSE.BN = False
cfg.LOFTR.COARSE.XFORMER = True
cfg.LOFTR.COARSE.ATTENTION = 'full'  # options: ['linear', 'full']

cfg.LOFTR.FINE.PAN = False
cfg.LOFTR.FINE.POOl_SIZE = 4
cfg.LOFTR.FINE.BN = False
cfg.LOFTR.FINE.XFORMER = False

# noalign
cfg.LOFTR.ALIGN_CORNER = False

# fp16
cfg.DATASET.FP16 = False
cfg.LOFTR.FP16 = False

# DEBUG
cfg.LOFTR.FP16LOG = False
cfg.LOFTR.MATCH_COARSE.FP16LOG = False

# fine skip
cfg.LOFTR.FINE.SKIP = True

# clip
cfg.TRAINER.GRADIENT_CLIPPING = 0.5

# backbone
cfg.LOFTR.BACKBONE_TYPE = 'RepVGG'

# A1
cfg.LOFTR.RESNETFPN.INITIAL_DIM = 64
cfg.LOFTR.RESNETFPN.BLOCK_DIMS = [64, 128, 256]  # s1, s2, s3
cfg.LOFTR.COARSE.D_MODEL = 256
cfg.LOFTR.FINE.D_MODEL = 64

# FPN backbone_inter_feat with coarse_attn.
cfg.LOFTR.COARSE_FEAT_ONLY = True
cfg.LOFTR.INTER_FEAT = True
cfg.LOFTR.RESNETFPN.COARSE_FEAT_ONLY = True
cfg.LOFTR.RESNETFPN.INTER_FEAT = True

# loop back spv coarse match
cfg.LOFTR.FORCE_LOOP_BACK = False

# fix norm fine match
cfg.LOFTR.MATCH_FINE.NORMFINEM = True

# loss cf weight
cfg.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = True
cfg.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = True

# leaky relu
cfg.LOFTR.RESNETFPN.LEAKY = False
cfg.LOFTR.COARSE.LEAKY = 0.01

# prevent FP16 OVERFLOW in dirty data
cfg.LOFTR.NORM_FPNFEAT = True
cfg.LOFTR.REPLACE_NAN = True

# force mutual nearest
cfg.LOFTR.MATCH_COARSE.FORCE_NEAREST = True
cfg.LOFTR.MATCH_COARSE.THR = 0.1

# fix fine matching
cfg.LOFTR.MATCH_FINE.FIX_FINE_MATCHING = True

# dwconv
cfg.LOFTR.COARSE.DWCONV = True

# localreg
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS = True
cfg.LOFTR.LOSS.LOCAL_WEIGHT = 0.25

# it5
cfg.LOFTR.EVAL_TIMES = 1

# rope
cfg.LOFTR.COARSE.ROPE = True

# local regress temperature
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0

# SLICE
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICE = True
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

# inner with no mask [64,100]
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_INNER = True
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_NOMASK = True

cfg.LOFTR.MATCH_FINE.TOPK = 1
cfg.LOFTR.MATCH_COARSE.FINE_TOPK = 1

cfg.LOFTR.MATCH_COARSE.FP16MATMUL = False