import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch
import matplotlib.colors as mcolors


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    elif dataset_name == 'c3vd':
        thr = 5e-4
    elif dataset_name == 'c3vd_undistort':
        thr = 5e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

# def make_matching_figure(
#         img0, img1, mkpts0, mkpts1, color,
#         kpts0=None, kpts1=None, text=[], dpi=75, path=None):
#     # draw image pair
#     assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
#     fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
#     axes[0].imshow(img0, cmap='gray')
#     axes[1].imshow(img1, cmap='gray')
#     for i in range(2):   # clear all frames
#         axes[i].get_yaxis().set_ticks([])
#         axes[i].get_xaxis().set_ticks([])
#         for spine in axes[i].spines.values():
#             spine.set_visible(False)
#     plt.tight_layout(pad=1)
    
#     if kpts0 is not None:
#         assert kpts1 is not None
#         axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
#         axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

#     # draw matches
#     if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
#         fig.canvas.draw()
#         transFigure = fig.transFigure.inverted()
#         fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
#         fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
#         fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
#                                             (fkpts0[i, 1], fkpts1[i, 1]),
#                                             transform=fig.transFigure, c=color[i], linewidth=1)
#                                         for i in range(len(mkpts0))]
        
#         axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
#         axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

#     # put txts
#     txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
#     fig.text(
#         0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
#         fontsize=15, va='top', ha='left', color=txt_color)

#     # save or return figure
#     if path:
#         plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
#         plt.close()
#     else:
#         return fig

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], path=None):
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    
    # Create a blank canvas to draw the images side by side
    height = max(img0.shape[0], img1.shape[0])
    width = img0.shape[1] + img1.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place the images on the canvas
    canvas[:img0.shape[0], :img0.shape[1]] = img0[...,None]
    canvas[:img1.shape[0], img0.shape[1]:] = img1[...,None]

    if kpts0 is not None and kpts1 is not None:
        # Draw keypoints on the images
        kpts0 = kpts0.astype(np.int32)
        kpts1 = kpts1.astype(np.int32)
        for pt in kpts0:
            cv2.circle(canvas, tuple(pt), 2, (255, 255, 255), -1)
        for pt in kpts1:
            cv2.circle(canvas, (pt[0] + img0.shape[1], pt[1]), 2, (255, 255, 255), -1)

    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        # Convert colormap colors to BGR format
        rgb_colors = [mcolors.to_rgb(c) for c in color]  # Convert RGBA to RGB
        rgb_colors = (np.array(rgb_colors) * 255).astype(np.uint8)

        # Draw matches
        for i in range(len(mkpts0)):
            pt1 = tuple(mkpts0[i].astype(np.int32))
            pt2 = tuple(mkpts1[i].astype(np.int32) + np.array([img0.shape[1], 0]))
            
            cv2.line(canvas, pt1, pt2, tuple([int(x) for x in rgb_colors[i]]), 1)
            cv2.circle(canvas, pt1, 4, tuple([int(x) for x in rgb_colors[i]]), -1)
            cv2.circle(canvas, pt2, 4, tuple([int(x) for x in rgb_colors[i]]), -1)

    # Add text
    txt_color = (0, 0, 0) if np.mean(img0[:100, :200]) > 200 else (255, 255, 255)
    y=30
    for line in text:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)
        y += 20

    # Display or save the figure
    if path is None:
        return canvas
    else:
        cv2.imwrite(path, canvas)


def _make_evaluation_figure(data, b_id, error_type = 'epi', alpha='dynamic'):
    b_mask = data['m_bids'] == b_id

    disance_thr = 3
    conf_thr = _compute_conf_thresh(data) if error_type == 'epi' else disance_thr
    
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    if error_type == 'epi':
        errs = data['epi_errs'][b_mask].cpu().numpy()
        correct_mask = errs < conf_thr
    elif error_type == 'homo':
        errs = data['point_errs'][b_id]
        correct_mask = errs < disance_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    # n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    # recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        # f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    all_match_figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    
    if error_type == 'epi':
        mask = data['inlier_pose'][b_id]

    elif error_type == 'homo':
        mask = data['inlier_homo'][b_id]
    
    text = ['#ID {} ({}_{})'.format(data['pair_id'][b_id], data['pair_names'][0][b_id], data['pair_names'][1][b_id]),
            'distance_error ({})'.format(np.mean(errs[mask]))]
    inlier_match_figure = make_matching_figure(img0, img1, kpts0[mask], kpts1[mask],
                                  color[mask], text=text)

    return all_match_figure, inlier_match_figure, cv2.cvtColor(np.concatenate([img0,img1], axis=1), cv2.COLOR_GRAY2RGB)

def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, error_type = 'epi', mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    all_match_figures = {mode: []}
    inlier_match_figures = {mode: []}
    pair_figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            all_match_fig, inlier_match_fig, pair_fig = _make_evaluation_figure(
                data, b_id, error_type= error_type,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        all_match_figures[mode].append(all_match_fig)
        inlier_match_figures[mode].append(inlier_match_fig)
        pair_figures[mode].append(pair_fig)
    return all_match_figures, inlier_match_figures, pair_figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

def make_supervision_figures(data):
    """ Make supervision figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    figures = {'spv_c': [], 'spv_f':[]}
    for b_id in range(data['image0'].size(0)):
        fig_c, fig_f = _make_spv_figure(data,b_id)
        figures['spv_c'].append(fig_c)
        figures['spv_f'].append(fig_f)

    return figures

def _make_spv_figure(data, b_id):

    b_mask = data['spv_b_ids'] == b_id
    m_b_mask = data['m_bids'] == b_id
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)

    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)    
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)  

    kpts0 = data['spv_pt0_i'][b_id].cpu().numpy()
    kpts1 = data['spv_pt1_i'][b_id].cpu().numpy()
    w_kpts0 = data['spv_w_pt0_i'][b_id].cpu().numpy()
    
    m_kpts0_c = data['mkpts0_c'][m_b_mask].cpu().numpy()
    m_kpts1_c = data['mkpts1_c'][m_b_mask].cpu().numpy()
    m_conf = data['mconf'][m_b_mask].cpu().numpy()
    m_kpts0_f = data['mkpts0_f'][m_b_mask].cpu().numpy()
    m_kpts1_f = data['mkpts1_f'][m_b_mask].cpu().numpy()

    # nearest_index1 = data['spv_nearest_index1'][b_id].cpu().numpy()
    # nearest_index0 = data['spv_nearest_index0'][b_id].cpu().numpy()
    

    # make the figure
    i_ids = data['spv_i_ids'][b_mask].cpu().numpy()
    j_ids = data['spv_j_ids'][b_mask].cpu().numpy()

    b_mask = data['b_ids'] == b_id
    i_ids_f = data['i_ids'][b_mask].cpu().numpy()
    j_ids_f = data['j_ids'][b_mask].cpu().numpy()
    



    # spv_c
    img0_c = img0.copy()
    for pts in kpts0[i_ids]:
        cv2.circle(img0_c, tuple([int(x) for x in pts]), 4, (0,255,0), -1)
    # Create a blank mask and fill it with transparency
    # mask = np.zeros((img0.shape[0], img0.shape[1], 4), dtype=np.uint8)
    for idx in range(len(m_kpts0_c)):
        cv2.circle(img0_c, tuple([int(x) for x in m_kpts0_c[idx]]), 2, conf_colormap(m_conf[idx]), -1)
    # Perform alpha blending to overlay the masked point onto the image
    # img0_c = cv2.addWeighted(img0_c, 1, cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR), 1, 0)    
    
    
    img1_c = img1.copy()
    for pts in kpts1[j_ids]:
        cv2.circle(img1_c, tuple([int(x) for x in pts]), 4, (0,255,0), -1)
    # Create a blank mask and fill it with transparency
    # mask = np.zeros((img1.shape[0], img1.shape[1], 4), dtype=np.uint8)
    for idx in range(len(m_kpts1_c)):
        cv2.circle(img1_c, tuple([int(x) for x in m_kpts1_c[idx]]), 2, conf_colormap(m_conf[idx]), -1)
    # Perform alpha blending to overlay the masked point onto the image
    # img1_c = cv2.addWeighted(img1_c, 1, cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR), 1, 0)    

    # spv_f
    img0_f = img0.copy()
    for pts in kpts0[i_ids_f]:
        cv2.circle(img0_f, tuple([int(x) for x in pts]), 2, (0,255,0), -1)
    
    for pts in m_kpts0_f:
        cv2.circle(img0_f, tuple([int(x) for x in pts]), 4, (0,0,255))

    img1_f = img1.copy()
    for pts in kpts1[j_ids_f]:
        cv2.circle(img1_f, tuple([int(x) for x in pts]), 2, (0,255,0), -1)
    
    for pts in w_kpts0[i_ids_f]:
        cv2.circle(img1_f, tuple([int(x) for x in pts]), 1, (255,0,0), -1)
    
    for pts in m_kpts1_f:
        cv2.circle(img1_f, tuple([int(x) for x in pts]), 4, (0,0,255))


    out_c = np.concatenate([img0_c,img1_c],axis=1)
    out_f = np.concatenate([img0_f,img1_f],axis=1)
    return out_c, out_f

def conf_colormap(conf):
    if conf>=0.5:
        return (int((1-conf)/0.5*255),0,0)
    else:
        return (255, int(-conf/0.5*255+255), int(-conf/0.5*255+255))

def make_attention_figures(data):
    """ Make attention figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    figures = {'self_att': [], 'cross_att':[]}
    for b_id in range(data['image0'].size(0)):
        fig_self, fig_cross = _make_att_figure(data,b_id)
        figures['self_att'].append(fig_self)
        figures['cross_att'].append(fig_cross)

    return figures

def _make_att_figure(data, b_id, top_k=20):

    self_att = [i[b_id].cpu().numpy() for i in data['self_att']] #([4800,4800],[4800,4800])
    cross_att = [i[b_id].cpu().numpy() for i in data['cross_att']]

    m_b_mask = data['m_bids'] == b_id
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)

    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)    
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)  

    m_kpts0_c = data['mkpts0_c'][m_b_mask].cpu().numpy()
    m_kpts1_c = data['mkpts1_c'][m_b_mask].cpu().numpy()
    m_conf = data['mconf'][m_b_mask].cpu().numpy()
    
    #choose one point
    idx = np.random.randint(0, len(m_kpts0_c))

    #self_att
    img0_self = img0.copy()
    cv2.circle(img0_self, tuple([int(x) for x in m_kpts0_c[idx]]), 4, (0,255,0), -1)
    scale = data['hw0_i'][1]/data['hw0_c'][1]
    points = np.round(m_kpts0_c[idx]/scale).astype(np.int32)
    c_idx = points[0] + points[1] * data['hw0_c'][1] #(N,L)
    self_att0 = self_att[0][c_idx]
    # Find the indices that would sort the array in descending order
    sorted_indices = np.argsort(self_att0)[::-1]
    # Extract the top 100 numbers and their locations
    top_k_values = self_att0[sorted_indices[:top_k]]
    top_k_values = (top_k_values - np.min(top_k_values)) / (np.max(top_k_values) - np.min(top_k_values))
    top_k_indices = sorted_indices[:top_k]

    for i in range(len(top_k_values)):
        value = top_k_values[i]
        indice = top_k_indices[i]
        loc = np.array([indice % data['hw0_c'][1]*scale, indice // data['hw0_c'][1]*scale], dtype=np.int32)
        pt1 = tuple(m_kpts0_c[idx].astype(np.int32))
        pt2 = tuple(loc)
        cv2.circle(img0_self, tuple([int(x) for x in loc]), 2, (255,0,0), -1)
        cv2.line(img0_self, pt1, pt2, conf_colormap(value), 1)
        

    img1_self = img1.copy()
    cv2.circle(img1_self, tuple([int(x) for x in m_kpts1_c[idx]]), 4, (0,255,0), -1)
    scale = data['hw1_i'][1]/data['hw1_c'][1]
    points = np.round(m_kpts1_c[idx]/scale).astype(np.int32)
    c_idx = points[0] + points[1] * data['hw1_c'][1] #(N,L)
    self_att1 = self_att[1][c_idx]
    # Find the indices that would sort the array in descending order
    sorted_indices = np.argsort(self_att1)[::-1]
    # Extract the top 100 numbers and their locations
    top_k_values = self_att1[sorted_indices[:top_k]]
    top_k_values = (top_k_values - np.min(top_k_values)) / (np.max(top_k_values) - np.min(top_k_values))
    top_k_indices = sorted_indices[:top_k]

    for i in range(len(top_k_values)):
        value = top_k_values[i]
        indice = top_k_indices[i]
        loc = np.array([indice % data['hw1_c'][1]*scale, indice // data['hw1_c'][1]*scale], dtype=np.int32)
        pt1 = tuple(m_kpts1_c[idx].astype(np.int32))
        pt2 = tuple(loc)
        cv2.circle(img1_self, tuple([int(x) for x in loc]), 2, (255,0,0), -1)
        cv2.line(img1_self, pt1, pt2, conf_colormap(value), 1)

    #cross_att
    img0_cross = img0.copy()
    img1_cross = img1.copy()
    cv2.circle(img0_cross, tuple([int(x) for x in m_kpts0_c[idx]]), 4, (0,255,0), -1)
    cv2.circle(img1_cross, tuple([int(x) for x in m_kpts1_c[idx]]), 4, (0,255,0), -1)
    cross_att_fig = np.concatenate([img0_cross,img1_cross],axis=1)
    scale = data['hw0_i'][1]/data['hw0_c'][1]
    points = np.round(m_kpts0_c[idx]/scale).astype(np.int32)
    c_idx = points[0] + points[1] * data['hw0_c'][1] #(N,L)
    cross_att0 = cross_att[0][c_idx]
    # Find the indices that would sort the array in descending order
    sorted_indices = np.argsort(cross_att0)[::-1]
    # Extract the top 100 numbers and their locations
    top_k_values = cross_att0[sorted_indices[:top_k]]
    top_k_values = (top_k_values - np.min(top_k_values)) / (np.max(top_k_values) - np.min(top_k_values))
    top_k_indices = sorted_indices[:top_k]

    for i in range(len(top_k_values)):
        value = top_k_values[i]
        indice = top_k_indices[i]
        loc = np.array([indice % data['hw1_c'][1]*scale+data['hw0_i'][1], indice // data['hw1_c'][1]*scale], dtype=np.int32)
        pt1 = tuple(m_kpts0_c[idx].astype(np.int32))
        pt2 = tuple(loc)
        cv2.circle(cross_att_fig, tuple([int(x) for x in loc]), 2, (255,0,0), -1)
        cv2.line(cross_att_fig, pt1, pt2, conf_colormap(value), 1)

    return np.concatenate([img0_self,img1_self],axis=1), cross_att_fig