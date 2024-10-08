#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/scannet_trainval.py"
main_cfg_path="configs/loftr/indoor/scannet/loftr_ds_eval.py"
ckpt_path="logs/tb_logs/indoor-ds-bs=2_loftr/version_1/checkpoints/epoch=32-homo@3=0.881-homo@5=0.930-homo@10=0.956.ckpt"
# dump_dir="dump/loftr_ds_indoor"
dump_img_dir="result_homo/"
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
    