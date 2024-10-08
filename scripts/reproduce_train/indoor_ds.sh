#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

cd $PROJECT_DIR

data_cfg_path="configs/data/scannet_trainval.py"
main_cfg_path="configs/loftr/indoor/loftr.py"   #options: [loftr.py crossmodal.py]

n_nodes=1
n_gpus_per_node=1
torch_num_workers=8
batch_size=2
pin_memory=true

base_name=$(basename "$main_cfg_path" .py)
exp_name="indoor-ds-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))_${base_name}"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=70 \
    # --ckpt_path "logs/tb_logs/indoor-ds-bs=2_loftr/version_9/checkpoints/epoch=4-auc@5=0.021-auc@10=0.052-auc@20=0.101.ckpt"
