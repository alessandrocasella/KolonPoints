#!/bin/bash
scan_name="cecum_t1_b_under_review" #[cecum_t1_b, cecum_t2_a, cecum_t2_b, cecum_t2_c, cecum_t4_a, sigmoid_t1_a, trans_t4_b]
model="LOFTR" #LOFTR SuperPoint
weights_path="logs/tb_logs/c3vd-ds-bs=2_loftr/version_19/checkpoints/epoch=9-auc@5=0.280-auc@10=0.480-auc@20=0.662.ckpt"

# weights_path="logs/tb_logs/c3vd-ds-bs=2_loftr/version_6/checkpoints/epoch=26-auc@5=0.123-auc@10=0.310-auc@20=0.507.ckpt"
# weights_path="logs/tb_logs/c3vd-ds-bs=2_loftr/version_19/checkpoints/epoch=9-auc@5=0.280-auc@10=0.480-auc@20=0.662.ckpt"

# model="SuperPoint"
# weights_path="third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"
# weights_path="third_party/weights/superglue_indoor.pth"
rm -rf tracker_outputs/${model} 
python ./demo_track.py ../KolonPoints/data/c3vd/${scan_name} \
        --model ${model} \
        --weights_path  ${weights_path} \
        --img_glob *color.png \
        --skip 5 \
        --H 480 \
        --W 640 \
        --no_display \
        --write \
        --max_length 10 \
        --min_length  3\
        --dis_thresh 4 \
        --video_name ${model}_${scan_name} \
        --write_dir tracker_outputs/${model} \
        --cuda
# #!/bin/bash
# scan_name="scene0001_00" #[cecum_t1_b, cecum_t2_a, cecum_t2_b, cecum_t2_c, cecum_t4_a, sigmoid_t1_a, trans_t4_b]
# # model="LOFTR" #LOFTR SuperPoint
# # #weights_path="logs/tb_logs/c3vd-ds-bs=2_loftr/version_19/checkpoints/epoch=9-auc@5=0.280-auc@10=0.480-auc@20=0.662.ckpt"
# # weights_path="weights/indoor_ds_new.ckpt"
# # weights_path="logs/tb_logs/c3vd-ds-bs=2_loftr/version_6/checkpoints/epoch=26-auc@5=0.123-auc@10=0.310-auc@20=0.507.ckpt"
# # weights_path="logs/tb_logs/c3vd-ds-bs=2_loftr/version_19/checkpoints/epoch=9-auc@5=0.280-auc@10=0.480-auc@20=0.662.ckpt"

# model="SuperPoint"
# weights_path="third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"
# # weights_path="third_party/weights/superglue_indoor.pth"
# rm -rf tracker_outputs/${model} 
# python ./demo_track.py /home/jhuang/dataset/scannet/train/${scan_name}/color \
#         --model ${model} \
#         --weights_path  ${weights_path} \
#         --img_glob *.jpg \
#         --skip 1 \
#         --H 480 \
#         --W 640 \
#         --no_display \
#         --write \
#         --max_length 10 \
#         --min_length  3\
#         --dis_thresh 4 \
#         --video_name ${model}_${scan_name} \
#         --write_dir tracker_outputs/${model} \
#         --cuda