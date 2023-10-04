# KolonPoints

## Tracking visualization

The tracking visualization script is ready to be used by demo_track.sh, it will run the python script demo_track.py. It includes 3 models, superpoint, superglue and LOFTR. The input is always a consecutive two frames even for superpoint.



This is a project base on the implementation of [LOFTR]([zju3dv/LoFTR: Code for "LoFTR: Detector-Free Local Feature Matching with Transformers", CVPR 2021 (github.com)](https://github.com/zju3dv/LoFTR))


To Do

- [ ] Include SuperPoint + SuperGlue
- [ ] Upgrade the code with up to date pytorch version

Finished features

- [X] Testing for LOFTR, SuperPoint, SuperGlue
- [X] Visualization for point tracking
- [X] Training with pinhole or omnidirectional camera model

## Environment Setting
docker image: [junchongh97/loftr:2.0.0]([Image Layer Details - junchongh97/loftr:2.0.0 | Docker Hub](https://hub.docker.com/layers/junchongh97/loftr/2.0.0/images/sha256-813f4eb2dddd00656494b9c9f3f19008630aa0d4f07b3cb9b4d5dc7e2f01b833?context=repo))

## Dataset Preparation

#### Scannet

Please follow the instruction from LOFTR

#### C3VD

Download the dataset using `bash data/download_dataset.sh` . This script will extract the data and keep only color images, depth images and pose ground truth. Error could happen if download request is too fast, so the download script is not absolutely guaranteed.

To create image pair indices, run `bash data/prepare_imgs_pairs.sh`. More clever method is to be updated.

## Training

```
#training for scannet
bash scripts/reproduce_train/indoor_ds.sh

#training for c3vd
bash scripts/reproduce_train/c3vd.sh
```

Training on C3VD is still under debugging.

## To be added...







