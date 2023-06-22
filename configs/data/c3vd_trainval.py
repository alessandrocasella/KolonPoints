from configs.data.base import cfg

TRAIN_BASE_PATH = "data/c3vd_indices"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "C3VD"
cfg.DATASET.TRAIN_DATA_ROOT = "data/c3vd"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/train"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list/c3vd_all.txt"

TEST_BASE_PATH = "data/c3vd_indices"
cfg.DATASET.TEST_DATA_SOURCE = "C3VD"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/c3vd"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/val"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/train_list/c3vd_all.txt"
# cfg.DATASET.VAL_INTRINSIC_PATH = cfg.DATASET.TEST_INTRINSIC_PATH = f"{TEST_BASE_PATH}/intrinsics.npz"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val