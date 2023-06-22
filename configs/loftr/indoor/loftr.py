from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

# cfg.DATASET.HOMO.TRAIN = True
# cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.1  # training tricks: save GPU memory
# cfg.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
# cfg.LOFTR.LOSS.FINE_WEIGHT = 1.0

cfg.LOFTR.BACKBONE_TYPE = 'ResNetFPN' # options: ['ResNetFPN', 'CrossModal']
cfg.TRAINER.ENABLE_PLOTTING =False
cfg.TRAINER.PLOTTING_SUPERVISION = True
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False
# cfg.TRAINER.WARMUP_TYPE = 'constant'  # [linear, constant]

cfg.TRAINER.MSLR_MILESTONES = [15, 25, 35, 40, 45, 50, 60, 65]
# cfg.TRAINER.MSLR_MILESTONES = [10, 20, 30, 40, 50, 60, 65]