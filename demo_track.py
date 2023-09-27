#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch

from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.loftr import LoFTR, default_cfg

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, dis_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.dis_thresh = dis_thresh
    self.all_pairs = []
    for n in range(self.maxl):
      self.all_pairs.append(np.zeros((0,2)))
    self.tracks = np.zeros((0, self.maxl+1))
    self.track_count = 0

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pairs)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pairs[i].shape[0])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pairs):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pairs is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    
    if self.track_count == 0: #initialize the track when have the first image pair
      self.all_pairs.pop(0)
      self.all_pairs.append(pairs['pts0'])
      self.all_pairs.pop(0)
      self.all_pairs.append(pairs['pts1'])
      offsets = self.get_offsets()
      new_ids0 = np.arange(pairs['pts0'].shape[0]) + offsets[-2]
      new_ids1 = np.arange(pairs['pts1'].shape[0]) + offsets[-1]
      first_tracks = -1*np.ones((new_ids0.shape[0], self.maxl + 1))
      first_tracks[:, -1] = new_ids1
      first_tracks[:, -2] = new_ids0
      new_num = new_ids0.shape[0]
      new_trackids = self.track_count + np.arange(new_num)
      first_tracks[:, 0] = new_trackids
      self.tracks = np.vstack((self.tracks, first_tracks))
      self.track_count += new_num # Update the track count.
      return
    
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pairs[0].shape[0]
    self.all_pairs.pop(0)
    self.all_pairs.append(pairs['pts1'])
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 1, axis=1)
    # Update track offsets.
    for i in range(1, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 1:][self.tracks[:, 1:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pairs['pts0'].shape[0])).astype(bool)

    # Add track if the current point pairs follow the previous ones.
    current_pts = pairs['pts0'][:, np.newaxis, :] # [N,1,2]
    previous_pts = self.all_pairs[-2][np.newaxis, :, :] # [1,N,2]
    distances = np.linalg.norm(current_pts - previous_pts, axis=2)
    #mutual finds and thresholding
    idx1_idx2 = np.argmin(distances, axis=1)
    idx2_idx1 = np.argmin(distances, axis=0)
    keep = np.min(distances,axis=1)<self.dis_thresh
    keep_bi = np.arange(len(idx1_idx2)) == idx2_idx1[idx1_idx2]
    keep = np.logical_and(keep, keep_bi)
    ids_current = np.arange(pairs['pts0'].shape[0])[keep] + offsets[-1]
    ids_previous = idx1_idx2[keep] + offsets[-2]
    matched[keep]=True
    for id_previous, id_current in zip(ids_previous, ids_current):
      found = np.argwhere(self.tracks[:, -2] == id_previous)
      if found.shape[0] > 0:
        row = int(found)
        self.tracks[row, -1] = id_current

    # Add unmatched tracks.
    new_ids = np.arange(pairs['pts0'].shape[0]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 1))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 1:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]

    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 1:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pairs
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      
      for i in range(N-1):
        if track[i+1] == -1 or track[i+2] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+1]-offset1)
        idx2 = int(track[i+2]-offset2)
        pt1 = pts_mem[i][idx1,:2]
        pt2 = pts_mem[i+1][idx2,:2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        clr = myjet[int(np.clip(i+2, 0, 9)), :]*255
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen-1:
      return (None, None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image0 = self.read_image(image_file, self.sizer)
      image_file = self.listing[self.i+1]
      input_image1 = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image0 = input_image0.astype('float32')
    input_image1 = input_image1.astype('float32')
    return (input_image0, input_image1, True)

if __name__ == '__main__':

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('input', type=str, default='',
      help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--img_glob', type=str, default='*.png',
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--H', type=int, default=120,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=160,
      help='Input image width (default:160).')
  parser.add_argument('--display_scale', type=int, default=1,
      help='Factor to scale output visualization (default: 1).')
  parser.add_argument('--min_length', type=int, default=2,
      help='Minimum length of point tracks (default: 2).')
  parser.add_argument('--max_length', type=int, default=5,
      help='Maximum length of point tracks (default: 5).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--dis_thresh', type=float, default=4,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--waitkey', type=int, default=1,
      help='OpenCV waitkey time in ms (default: 1).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely (default: False).')
  parser.add_argument('--write', action='store_true',
      help='Save output frames to a directory (default: False)')
  parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
      help='Directory where to write output frames (default: tracker_outputs/).')
  parser.add_argument('--video_name', type=str,
      help='Name for the output video.')
  opt = parser.parse_args()
  print(opt)

  _default_cfg = get_cfg_defaults()
  _default_cfg['LOFTR']['COARSE']['TEMP_BUG_FIX'] = True  # set to False when using the old ckpt
  _default_cfg['LOFTR']['MATCH_COARSE']['THR'] = 0.9

  # This class helps load input images from different sources.
  vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

  print('==> Loading pre-trained network.')
  matcher = LoFTR(config=lower_config(_default_cfg['LOFTR']))
  matcher.load_state_dict(torch.load(opt.weights_path)['state_dict'])
  matcher = matcher.eval().cuda()
  print('==> Successfully loaded pre-trained network.')

  # This class helps merge consecutive point matches into tracks.
  tracker = PointTracker(opt.max_length, dis_thresh=opt.dis_thresh)

  # Font parameters for visualizaton.
  font = cv2.FONT_HERSHEY_DUPLEX
  font_clr = (255, 255, 255)
  font_pt = (4, 12)
  font_sc = 0.4

  # Create output directory if desired.
  if opt.write:
    print('==> Will write outputs to %s' % opt.write_dir)
    if not os.path.exists(opt.write_dir):
      os.makedirs(opt.write_dir)

  print('==> Running Demo.')
  while True:

    # Get a new image.
    img0, img1, status = vs.next_frame()
    if status is False:
      break

    # Get points and descriptors.
    batch = {'image0': torch.from_numpy(img0)[None][None].cuda(), 
         'image1': torch.from_numpy(img1)[None][None].cuda(),
         'dataset_name': ['C3VD_undistort']}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        # mkpts0 = batch['mkpts0_f'].cpu().numpy()
        # mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        i_ids = batch['i_ids'].cpu().numpy()
        j_ids = batch['j_ids'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    if mkpts0.shape[0] == 0:
      pairs = None
    else:
      pairs = {'pts0': mkpts0, 'pts1': mkpts1}

    tracker.update(pairs)

    # Get tracks for points which were match successfully across all frames.
    tracks = tracker.get_tracks(opt.min_length)
    # print(matches)
    # Primary output - Show point tracks overlayed on top of input image.
    out1 = (np.dstack((img1, img1, img1)) * 255.).astype('uint8')
    tracker.draw_tracks(out1, tracks)

    out = cv2.resize(out1, (opt.display_scale*opt.W, opt.display_scale*opt.H))

    # Optionally write images to disk.
    if opt.write:
      out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
      print('Writing image to %s' % out_file)
      cv2.imwrite(out_file, out)
      
  #writing video
  
  # Define the output video file name and parameters
  output_file = opt.video_name +'.avi'
  frame_width = 640
  frame_height = 480
  frame_rate = 20.0

  # Create a VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
  out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

  # Assuming you have a list of frames (frames_list) or a loop capturing frames
  search = os.path.join(opt.write_dir, '*.png')
  frames_list = glob.glob(search)
  frames_list.sort()
  for frame in frames_list:
      # Resize the frame if necessary
      frame = cv2.imread(frame, -1)
      frame = cv2.resize(frame, (frame_width, frame_height))
      
      # Write the frame to the output video
      out.write(frame)

  # Release the VideoWriter and close the output file
  out.release()

  print('==> Finshed Demo.')

