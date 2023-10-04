from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.DATASET.HOMO.TRAIN = True
cfg.DATASET.HOMO.VAL = False
cfg.TRAINER.N_SAMPLES_PER_SUBSET=100

cfg.LOFTR.BACKBONE_TYPE = 'ResNetFPN' # options: ['ResNetFPN', 'CrossModal']
cfg.TRAINER.ENABLE_PLOTTING =True
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

# cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.05  # training tricks: save GPU memory
# cfg.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 0  # training tricks: avoid DDP deadlock

cfg.LOFTR.MATCH_COARSE.THR = 0.2
# cfg.LOFTR.LOSS.COARSE_WEIGHT = 0.0

# cfg.TRAINER.WARMUP_TYPE = 'constant'  # [linear, constant]
# cfg.TRAINER.MSLR_MILESTONES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
cfg.TRAINER.MSLR_MILESTONES = [10, 15, 20, 25, 30, 35, 40]
