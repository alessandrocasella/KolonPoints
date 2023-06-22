from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.DATASET.HOMO.TRAIN = True
# cfg.TRAINER.N_SAMPLES_PER_SUBSET=100

cfg.LOFTR.BACKBONE_TYPE = 'ResNetFPN' # options: ['ResNetFPN', 'CrossModal']
cfg.TRAINER.ENABLE_PLOTTING =False
cfg.TRAINER.PLOTTING_SUPERVISION = True
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False
# cfg.TRAINER.WARMUP_TYPE = 'constant'  # [linear, constant]

cfg.TRAINER.MSLR_MILESTONES = [5, 10, 15, 20, 25, 30, 35, 40]
