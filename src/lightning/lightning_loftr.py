
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import cv2
import os

import torch
import numpy as np
import pytorch_lightning as pl

from src.loftr import LoFTR, SuperPointFrontend, Matching
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    compute_homo_error,
    aggregate_metrics,
    aggregate_homo_estimation_metric
)
from src.utils.plotting import make_matching_figures, make_supervision_figures, make_attention_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, dump_img_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr'])
        
        # self.superpoint = SuperPointFrontend(weights_path="../SuperPointPretrainedNetwork/superpoint_v1.pth",
        #                   nms_dist=4,
        #                   conf_thresh=0.005)

        # config = {
        # 'superpoint': {
        #     'nms_radius': 4, #opt.nms_radius,
        #     'keypoint_threshold': 0.005, #opt.keypoint_threshold,
        #     'max_keypoints': 1024, #opt.max_keypoints
        # },
        # 'superglue': {
        #     'weights': 'indoor',#opt.superglue,
        #     'sinkhorn_iterations': 20, #opt.sinkhorn_iterations,
        #     'match_threshold': 0.2 #opt.match_threshold,
        # }
        # }
        # self.superpoint_superglue = Matching(config).eval()

        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
            # frozen_layers = ["backbone", "loftr_coarse"]
            # for name, param in self.matcher.named_parameters():
            #     for frozen_layer in frozen_layers:
            #         if frozen_layer in name:
            #             param.requires_grad = False
        
        # Testing
        self.dump_dir = dump_dir
        self.dump_img_dir = dump_img_dir
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        if self.trainer.global_step==0:
            self.config.TRAINER.WARMUP_STEP = 3*self.trainer.num_training_batches
        if self.trainer.global_step < self.config.TRAINER.WARMUP_STEP:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        
    def _train_inference(self, batch):
        # if self.config['LOFTR']['SUPERPOINT'] and train:
        #     with torch.no_grad():
        #         self.superpoint.run(batch)
        
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        '''
            data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        '''

        with self.profiler.profile("LoFTR"):
            self.matcher(batch)
        '''
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
                'm_bids'
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        '''
         
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
        
        '''
            "expec_f_gt"
        '''
            
        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _val_inference(self, batch):

        with self.profiler.profile("LoFTR"):
            self.matcher(batch)
        '''
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
                'm_bids'
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        '''
                
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            if not self.config['DATASET']['HOMO']['VAL']:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

                rel_pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].size(0)
                metrics = {
                    # to filter duplicate pairs caused by DistributedSampler
                    'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                    'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                    'R_errs': batch['R_errs'],
                    't_errs': batch['t_errs'],
                    'inlier_pose': batch['inlier_pose']}
                ret_dict = {'metrics': metrics}
            else:
                rel_pair_names = list(zip(*batch['pair_names']))
                compute_homo_error(batch)

                metrics = {
                    'point_errs': batch['point_errs'], #list [(N,), (M)]
                    'mean_dist': batch['mean_dist']
                }
                ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def training_step(self, batch, batch_idx):
        
        self._train_inference(batch)

        # logging
        if self.trainer.global_rank == 0 and self.global_step % int(0.25*self.trainer.num_training_batches) == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                if not self.config.DATASET.HOMO.TRAIN:
                    compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                    all_match_figures, _, pair_figures = make_matching_figures(batch, self.config, error_type = 'epi', mode = self.config.TRAINER.PLOT_MODE)

                else:
                    compute_homo_error(batch)
                    all_match_figures, _, pair_figures = make_matching_figures(batch, self.config, error_type = 'homo', mode = self.config.TRAINER.PLOT_MODE)
                for k, v in all_match_figures.items():
                    pairs = pair_figures[k]
                    out_pair = [np.concatenate([pairs[idx],v[idx]], axis=0) for idx in range(len(v))]
                    self.logger.experiment.add_images(f'train_all_match/{k}', np.concatenate(out_pair, axis=1), self.global_step, dataformats='HWC')


                figures = make_supervision_figures(batch)
                for k, v in figures.items():
                    v = np.concatenate(v,axis=0)
                    self.logger.experiment.add_images(f'{k}', v, self.global_step,dataformats='HWC')
                
                # figures = make_attention_figures(batch)
                # for k, v in figures.items():
                #     v = np.concatenate(v,axis=0)
                #     self.logger.experiment.add_images(f'{k}', v, self.global_step,dataformats='HWC')

        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        self._val_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        all_match_figures = {self.config.TRAINER.PLOT_MODE: []}
        inlier_match_figures = {self.config.TRAINER.PLOT_MODE: []}
        pair_figures = {self.config.TRAINER.PLOT_MODE: []}

        if batch_idx % val_plot_interval == 0:
            all_match_figures,  inlier_match_figures, pair_figures= make_matching_figures(batch, self.config, error_type = 'homo' if self.config.DATASET.HOMO.VAL else 'epi', 
                                                        mode=self.config.TRAINER.PLOT_MODE)
        return {
            **ret_dict,
            # 'loss_scalars': batch['loss_scalars'],
            'all_match_figures': all_match_figures,
            'inlier_match_figures': inlier_match_figures,
            'pair_figures': pair_figures
        }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # # 1. loss_scalars: dict of list, on cpu
            # _loss_scalars = [o['loss_scalars'] for o in outputs]
            # loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            if not self.config['DATASET']['HOMO']['VAL']:
                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
                for thr in [5, 10, 20]:
                    multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            else:
                homo_auc = aggregate_homo_estimation_metric(metrics['mean_dist'])
                for thr in [3, 5, 10]:
                    multi_val_metrics[f'homo@{thr}'].append(homo_auc[f'homo@{thr}'])
            # # 3. figures
            _all_match_figures = [o['all_match_figures'] for o in outputs]
            all_match_figures = {k: flattenList(gather(flattenList([_me[k] for _me in _all_match_figures]))) for k in _all_match_figures[0]}
            
            _inlier_match_figures = [o['inlier_match_figures'] for o in outputs]
            inlier_match_figures = {k: flattenList(gather(flattenList([_me[k] for _me in _inlier_match_figures]))) for k in _inlier_match_figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                # for k, v in loss_scalars.items():
                #     mean_v = torch.stack(v).mean()
                #     self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                if not self.config['DATASET']['HOMO']['VAL']:
                    for k, v in val_metrics_4tb.items():
                        self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                else:
                    for k, v in homo_auc.items():
                        self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                for k, v in all_match_figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_images(
                                f'val_all_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, dataformats='HWC')
                for k, v in inlier_match_figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_images(
                                f'val_inlier_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, dataformats='HWC')

        if not self.config['DATASET']['HOMO']['VAL']:
            for thr in [5, 10, 20]:
                # log on all ranks for ModelCheckpoint callback to work properly
                self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this
        else:
            for thr in [3, 5, 10]:
                self.log(f'homo@{thr}', torch.tensor(np.mean(multi_val_metrics[f'homo@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        #LOFTR
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        # # SuperPoint
        # pts0, des0, _ = self.superpoint.run(batch['image0'])
        # pts1, des1, _ = self.superpoint.run(batch['image1'])
        # matches = self.superpoint.nn_match_two_way(des0,des1, nn_thresh=0.7)
        # batch['m_bids'] = torch.zeros(matches.shape[1], device=batch['image0'].device)
        # batch['mkpts0_f'] = torch.tensor(pts0[:2,matches[0].astype(int)].T, device=batch['image0'].device)
        # batch['mkpts1_f'] = torch.tensor(pts1[:2,matches[1].astype(int)].T, device=batch['image0'].device)

        # # SuperPoint + SuperGlue
        # pred = self.superpoint_superglue({'image0': batch['image0'], 'image1': batch['image1']})
        # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        # kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        # matches, conf = pred['matches0'], pred['matching_scores0']
        # batch['m_bids'] = torch.zeros(matches[matches>-1].shape[0], device=batch['image0'].device)
        # batch['mkpts0_f'] = torch.tensor(kpts0[matches>-1], device=batch['image0'].device)
        # batch['mkpts1_f'] = torch.tensor(kpts1[matches[matches>-1]], device=batch['image0'].device)
        
        # valid0 = [batch['mask'][0][int(pt[1]),int(pt[0])] for pt in batch['mkpts0_f']]
        # valid1 = [batch['mask'][0][int(pt[1]),int(pt[0])] for pt in batch['mkpts1_f']]
        # valid = torch.tensor(valid0) * torch.tensor(valid1)
        # batch['m_bids'] = batch['m_bids'][valid]
        # batch['mkpts0_f'] = batch['mkpts0_f'][valid]
        # batch['mkpts1_f'] = batch['mkpts1_f'][valid]

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inlier_pose']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps
            if self.config.TRAINER.PLOTTING_TEST_RESULT:
                all_match_figures,  inlier_match_figures, pair_figures = make_matching_figures(batch, self.config, error_type = 'homo' if self.config.DATASET.HOMO.VAL else 'epi', 
                                                        mode=self.config.TRAINER.PLOT_MODE)
                ret_dict.update({'all_match_figures': all_match_figures,
                                 'inlier_match_figures': inlier_match_figures,
                                 'pair_figures': pair_figures})

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')
        
        if self.config.TRAINER.PLOTTING_TEST_RESULT:
            
            logger.info(f'Matching results will be saved to: {self.dump_img_dir} ...')
            Path(self.dump_img_dir).mkdir(parents=True, exist_ok=True)
            
            _all_match_figures = [o['all_match_figures'] for o in outputs]
            all_match_figures = {k: flattenList(gather(flattenList([_me[k] for _me in _all_match_figures]))) for k in _all_match_figures[0]}
            
            _inlier_match_figures = [o['inlier_match_figures'] for o in outputs]
            inlier_match_figures = {k: flattenList(gather(flattenList([_me[k] for _me in _inlier_match_figures]))) for k in _inlier_match_figures[0]}
            
            _pair_figures = [o['pair_figures'] for o in outputs]
            pair_figures = {k: flattenList(gather(flattenList([_me[k] for _me in _pair_figures]))) for k in _pair_figures[0]}
            
            for k, v in all_match_figures.items():
                for plot_idx, fig in enumerate(v):
                    all_match = fig[..., ::-1]
                    inlier_match = inlier_match_figures[k][plot_idx][..., ::-1]
                    pair = pair_figures[k][plot_idx][..., ::-1]
                    cv2.imwrite(os.path.join(self.dump_img_dir, f'pair_{plot_idx}.jpg'), np.concatenate([pair,all_match,inlier_match],axis=0))



        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            if not self.config.DATASET.HOMO.VAL:
                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
                logger.info('\n' + pprint.pformat(val_metrics_4tb))
            else:
                homo_auc = aggregate_homo_estimation_metric(metrics['mean_dist'])
                logger.info('\n' + pprint.pformat(homo_auc))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
