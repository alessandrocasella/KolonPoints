#!/bin/bash
scan_name="cecum_t2_a_under_review"
rm -rf tracker_outputs
python ./demo_track.py ../KolonPoints/data/c3vd_undistort/${scan_name} \
        --img_glob *color.png \
        --skip 1 \
        --H 480 \
        --W 640 \
        --no_display \
        --write \
        --max_length 10 \
        --min_length 2 \
        --weights_path logs/tb_logs/c3vd-ds-bs=2_loftr/version_6/checkpoints/epoch=26-auc@5=0.123-auc@10=0.310-auc@20=0.507.ckpt \
        --video_name ${scan_name} \
        --cuda