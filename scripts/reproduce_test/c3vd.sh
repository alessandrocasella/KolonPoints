#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/c3vd_trainval.py"
main_cfg_path="configs/loftr/c3vd/eval.py"
ckpt_path="weights/indoor_ds_new.ckpt" #"logs/tb_logs/c3vd-ds-bs=2_loftr/version_1/checkpoints/epoch=35-auc@5=0.201-auc@10=0.403-auc@20=0.583.ckpt"
# dump_dir="dump/loftr_ds_indoor"
dump_img_dir="result_c3vd/"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=8
batch_size=1  # per gpu

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark \
    --dump_img_dir=${dump_img_dir} \
    #--dump_dir=${dump_dir} 
    