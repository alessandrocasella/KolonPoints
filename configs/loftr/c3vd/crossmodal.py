from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.LOFTR.BACKBONE_TYPE = 'CrossModal' # options: ['ResNetFPN', 'CrossModal']
cfg.TRAINER.ENABLE_PLOTTING =False
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False
# cfg.TRAINER.WARMUP_TYPE = 'constant'  # [linear, constant]

cfg.TRAINER.MSLR_MILESTONES = [15, 25, 35, 40, 45, 50, 60, 65]
