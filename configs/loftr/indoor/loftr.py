from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.DATASET.HOMO.TRAIN = True
cfg.DATASET.HOMO.VAL = False
# cfg.LOFTR.SUPERPOINT = True

# cfg.LOFTR.MATCH_COARSE.THR = 0.2

# cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.1  # training tricks: save GPU memory
# cfg.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 20  # training tricks: avoid DDP deadlock
# cfg.LOFTR.LOSS.FINE_WEIGHT = 1.0
# cfg.LOFTR.LOSS.COARSE_WEIGHT = 5.0
cfg.LOFTR.BACKBONE_TYPE = 'ResNetFPN' # options: ['ResNetFPN', 'CrossModal']
cfg.TRAINER.ENABLE_PLOTTING =True
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False
# cfg.TRAINER.N_SAMPLES_PER_SUBSET = 10
# cfg.LOFTR.COARSE.ATTENTION = 'full'
# cfg.LOFTR.FINE.ATTENTION = 'full'

cfg.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
# cfg.TRAINER.CANONICAL_LR = 1e-4

# cfg.TRAINER.MSLR_MILESTONES = [15, 25, 35, 40, 45, 50, 60, 65]
cfg.TRAINER.MSLR_MILESTONES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# cfg.TRAINER.MSLR_MILESTONES = [5, 10, 15, 20, 25, 30, 35, 40]