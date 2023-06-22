#!/usr/bin/env bash
SCRIPTPATH=$(dirname $(readlink -f "$0"))
DATA_DIR="${SCRIPTPATH}"

cd $DATA_DIR

if [[ $# != 2 ]]; then
    echo 'Usage: bash prepare_imgs_pairs.sh /path/to/C3VD /output/path'
    exit
fi

export dataset_path=$1
export output_path=$2

mkdir -p "${output_path}/train"
mkdir -p "${output_path}/val"
mkdir -p "${output_path}/train_list"
echo 0
ls $dataset_path | xargs -P 16 -I % sh -c 'python prepare_imgs_pairs.py --base_path $dataset_path --scene_id % --output_path $output_path'
ls $dataset_path > "${output_path}/train_list/c3vd_all.txt"