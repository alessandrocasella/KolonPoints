import torch
import torch.nn as nn
import numpy as np


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0

@torch.no_grad()
def omni_warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt under omnidirectional camera modal
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 4, 4],
        K0 (dict:torch.Tensor): {'xc': [N,1],'yc': [N,1],'ss': [N,5],'c': [N,1],'d': [N,1],'e': [N,1]},
        K1 (dict:torch.Tensor): {'xc': [N,1],'yc': [N,1],'ss': [N,5],'c': [N,1],'d': [N,1],'e': [N,1]}
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0
    
    # Unproject
    kpts0_unit = cam2world(kpts0.transpose(1,2), K0).transpose(1,2)  #(N,L,3)
    kpts0_cam = kpts0_unit/kpts0_unit[:,:,[2]]*kpts0_depth[...,None] #(N,L,3)
    
    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam.transpose(2,1) + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0 = world2cam(w_kpts0_cam, K1).transpose(1,2) #(N,L,2)
    
    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0    

# @torch.no_grad()
# def cam2world(m, ocam_model):
#     """Warp point in pixel coordinate to unit sphere coordinate
#     Args:
#         m (torch.Tensor): [N,2,L]
#         ocam_model (dict):
#     Returns:
#         M (torch.Tensor): [N,3,L]
#     """

#     ss = ocam_model['ss'][0]
#     xc = ocam_model['xc'][0]
#     yc = ocam_model['yc'][0]
#     c = ocam_model['c'][0]
#     d = ocam_model['d'][0]
#     e = ocam_model['e'][0]

#     A = torch.tensor([[c, d], [e, 1]], dtype=torch.float32, device=m.device)
#     T = torch.tensor([[xc], [yc]], dtype=torch.float32, device=m.device)

#     m = torch.matmul(torch.linalg.inv(A), (m - T)) # (N,2,L)
#     M = getpoint(ss, m)
#     M = torch.div(M, torch.norm(M, dim=1, keepdim=True))  # normalizes coordinates to unit length

#     return M

# @torch.no_grad()
# def getpoint(ss, m):
#     """
#     Args:
#         m (torch.Tensor): [N,2,L]
#     Returns:
#         w (torch.Tensor): [N,3,L]
#     """
#     # Given an image point, it returns the 3D coordinates of its corresponding optical ray
#     rho = polyval(ss.flip(0), torch.sqrt(m[:, 0]**2 + m[:, 1]**2))
#     w = torch.stack((m[:, 0], m[:, 1], rho), dim=1)

#     return w

# @torch.no_grad()
# def polyval(coefficients, x):
#     """
#     Evaluate a polynomial at specific values.

#     Arguments:
#     - coefficients [dim]: A tensor of polynomial coefficients in descending order.
#     - x [N, L]: A tensor of values at which to evaluate the polynomial.

#     Returns:
#     - [N,L]A tensor of the evaluated polynomial values.
#     """
#     powers = torch.arange(coefficients.size(0) - 1, -1, -1, dtype=x.dtype, device=x.device)
#     powers = powers.unsqueeze(0)  # Add a batch dimension (1,dim)

#     x = x.unsqueeze(-1)  # Add a dimension for broadcasting (N,L,1)
#     coefficients = coefficients[None,None,...]  # Add a dimension for broadcasting

#     result = torch.sum(coefficients * (x ** powers), dim=-1) #(N,L)

#     return result

def cam2world(m, ocam_model):
    ss = ocam_model['ss'][0]
    xc = ocam_model['xc'][0]
    yc = ocam_model['yc'][0]
    c = ocam_model['c'][0]
    d = ocam_model['d'][0]
    e = ocam_model['e'][0]

    A = np.array([[c, d], [e, 1]])
    T = np.array([[xc], [yc]])

    m = np.linalg.inv(A) @ (m - T)
    M = getpoint(ss, m)
    M = np.divide(M, np.linalg.norm(M, axis=0))  # normalizes coordinates to unit length

    return M

def getpoint(ss, m):
    # Given an image point, it returns the 3D coordinates of its corresponding optical ray
    w = np.vstack((m[0], m[1], np.polyval(ss[::-1], np.sqrt(m[0]**2 + m[1]**2))))
    return w

@torch.no_grad()
def world2cam(M, ocam_model):
    """
    Args:
        M (torch.Tensor): [N,3,L]
    Returns:
        m (torch.Tensor): [N,2,L]

    """
    device = M.device
    ss = ocam_model['ss'][0]
    xc = ocam_model['xc'][0]
    yc = ocam_model['yc'][0]
    c = ocam_model['c'][0]
    d = ocam_model['d'][0]
    e = ocam_model['e'][0]

    x, y = omni3d2pixel(ss, M) #[N,L]
    m = torch.zeros((M.shape[0], 2, M.shape[2]), dtype=torch.float32, device=device)
    m[:, 0, :] = x * c + y * d + xc
    m[:, 1, :] = x * e + y + yc

    return m

@torch.no_grad()
def omni3d2pixel(ss, xx):
    """
    Args:
        xx (torch.Tensor): [N, 3, L]
    Returns:
        x (torch.Tensor): [N, L]
        y (torch.Tensor): [N, L]
    """
    N = xx.shape[0]
    device = xx.device
    
    ind0 = torch.where((xx[:, 0, :] == 0) & (xx[:, 1, :] == 0))
    xx[:, 0, :][ind0] = torch.finfo(torch.float32).eps
    xx[:, 1, :][ind0] = torch.finfo(torch.float32).eps

    m = xx[:, 2, :] / torch.sqrt(xx[:, 0, :] ** 2 + xx[:, 1, :] ** 2)  # [N, L]
    rho = torch.zeros(m.shape, device=device) #[N, L]
    poly_coef = ss.flip(0).unsqueeze(0).repeat(N,1) #[N,poly]
    poly_coef_tmp = poly_coef.clone() #[N,poly]
    
    for j in range(m.shape[1]):
        poly_coef_tmp[:,-2] = poly_coef[:,-2] - m[:, j] #[N]
        for i in range(m.shape[0]):
            rho_tmp = np.roots(poly_coef_tmp[i].cpu().numpy())
            res = rho_tmp[np.logical_and(np.imag(rho_tmp) == 0, rho_tmp > 0)]
            if len(res) == 0:
                rho[i, j] = float('nan')
            elif len(res) > 1:
                rho[i, j] = np.min(res)

    x = xx[:, 0, :] / torch.sqrt(xx[:, 0, :] ** 2 + xx[:, 1, :] ** 2) * rho
    y = xx[:, 1, :] / torch.sqrt(xx[:, 0, :] ** 2 + xx[:, 1, :] ** 2) * rho

    return x, y