import io
from loguru import logger

import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv
import math
from numpy.random import normal
from numpy.random import uniform
from scipy.stats import truncnorm
from math import pi

from .rgbd_util import *


try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def load_array_from_s3(
    path, client, cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale


def read_megadepth_depth(path, pad_to=None):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None, homo=False):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    if homo:
        return image
    else:
        return torch.from_numpy(image).float()[None] / 255

def read_c3vd_gray(path, resize=None, augment_fn=None, with_mask=True, return_np=False):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    if resize:
        image = cv2.resize(image, resize)
    if with_mask:
        # Set a darkness threshold
        darkness_threshold = 5

        # Create a mask by thresholding the grayscale image
        mask = np.where(image < darkness_threshold, 0, 255).astype(np.uint8)

        # Perform morphological closing to fill small holes in the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        closed_mask = cv2.GaussianBlur(closed_mask, (13, 13), 0)

        closed_mask = closed_mask==255
    else:
        closed_mask = np.array([np.nan])

    # (h, w) -> (1, h, w) and normalized
    if return_np:
        return image, closed_mask
    else:
        return torch.from_numpy(image).float()[None] / 255, torch.from_numpy(closed_mask)


def read_scannet_depth(path,K,cross_modal=False):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(str(path), SCANNET_CLIENT, cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()
    
    if cross_modal:
        hha = cv2.imread(str(path).replace("depth","hha"), cv2.IMREAD_UNCHANGED)
        if hha is None:
            K = np.reshape(K, (3,3))
            hha = getHHA(K, depth, depth)
            cv2.imwrite(str(path).replace("depth","hha"), hha)
            # (h, w)
        hha = torch.from_numpy(hha).float().permute(2, 0, 1)/255
    else: 
        hha = None
    return depth, hha

def read_c3vd_depth_mask(path, resize=(640, 480)):
    depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if resize:
        depth = cv2.resize(depth, resize)
    sobelx = cv2.Sobel(depth, cv2.CV_16S, 1, 0, ksize=-1)  
    sobely = cv2.Sobel(depth, cv2.CV_16S, 0, 1, ksize=-1)  
    abx = cv2.convertScaleAbs(sobelx) 
    aby = cv2.convertScaleAbs(sobely)  
    dst = cv2.addWeighted(abx, 0.5, aby, 0.5, 0)  

    blurred_edges = dst
    blurred_edges = cv2.GaussianBlur(blurred_edges, (5, 5), 0)
    blurred_edges = blurred_edges>np.quantile(blurred_edges, 0.9)

    return blurred_edges

def read_c3vd_depth(path, K ,cross_modal=False, resize=(640, 480)):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(str(path), SCANNET_CLIENT, cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    depth = depth / 1000
    if resize:
        depth = cv2.resize(depth, resize)
    depth = torch.from_numpy(depth).float()
    
    if cross_modal:
        # hha = cv2.imread(str(path).replace("depth","hha"), cv2.IMREAD_UNCHANGED)
        # if hha is None:
        #     K = np.reshape(K, (3,3))
        #     hha = getHHA(K, depth, depth)
        #     cv2.imwrite(str(path).replace("depth","hha"), hha)
        #     # (h, w)
        # hha = torch.from_numpy(hha).float().permute(2, 0, 1)/255
        
        hha = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if resize:
            hha = cv2.resize(hha, resize)
        hha = torch.from_numpy(hha).float()[None] / 255
    else: 
        hha = None
    return depth, hha

def getHHA(C, D, RD):
    missingMask = (RD == 0)
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C)

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180        


    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I>255] = 255
    HHA = I.astype(np.uint8)
    return HHA


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam

def read_c3vd_pose(path, name):
    """ Read C3VD's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    with open(path, 'r') as f:
        raw = f.readlines()
    world2cam = inv(np.fromstring(raw[int(name)], dtype=float, sep=",").reshape((4,4)))
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]

def sample_homography_np(
        shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # print("debugging")


    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    
    patch_ratio = np.random.uniform(patch_ratio, 0.8)
    
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        # perspective_displacement = tf.truncated_normal([1], 0., perspective_amplitude_y/2)
        # perspective_displacement = normal(0., perspective_amplitude_y/2, 1)
        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        # h_displacement_left = normal(0., perspective_amplitude_x/2, 1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        # h_displacement_right = normal(0., perspective_amplitude_x/2, 1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        # scales = np.concatenate( (np.ones((n_scales,1)), scales[:,np.newaxis]), axis=1)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            # valid = np.where((scaled >= 0.) * (scaled < 1.))
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]
        # idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        # pts2 = rotated[idx]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]反转
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    # a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    # p_mat = tf.transpose(tf.stack(
    #     [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    # homography = tf.transpose(tf.matrix_solve_ls(a_mat, p_mat, fast=True))
   #  pts1*= pts1*shape
    homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
    return homography