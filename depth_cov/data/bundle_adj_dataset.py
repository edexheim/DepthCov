from random import random
import torch
from torch.utils.data import Dataset

import numpy as np
import cv2
import gtsam

class BundleAdjDataset(Dataset):
  def __init__(self, base_dir, seqs, window_size):
    
    self.seq_dirs = []
    self.seq_lines = []
    for i in range(len(seqs)):
      seq_dir = base_dir + seqs[i]

      filename = seq_dir + "rgb_depth_groundtruth.txt"
      file = open(filename)
      lines = file.readlines()
      num_lines = len(lines)

      for i in range(100, num_lines, window_size):
        window_lines = lines[i:i+window_size]

        # print(len(window_lines))
        # print(window_lines[0])

        if len(window_lines) < window_size:
          continue

        # Check timestamps don't have jumps
        valid_ts = True
        curr_ts = float(window_lines[0].split()[0])
        for j in range(1, len(window_lines)):
          next_ts = float(window_lines[j].split()[0])
          if next_ts - curr_ts > 0.1:
            valid_ts = False
            break
          else:
            curr_ts = next_ts

        # print(valid_ts)

        if valid_ts:
          self.seq_dirs.append(seq_dir)
          self.seq_lines.append(window_lines)
      
  def __len__(self):
    return self.seq_dirs.__len__()

  def translation_quat_to_pose3(self, t, q):
    rot3 = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2]) # w, x, y, z
    pose3 = gtsam.Pose3(rot3, gtsam.Point3(t))
    return pose3

  def __getitem__(self, idx):

    seq_dir = self.seq_dirs[idx]
    seq_lines = self.seq_lines[idx]

    rgb_list = []
    depth_list = []
    pose_list = []
    pose_timestamps = []
    for i in range(len(seq_lines)):
      line_list = seq_lines[i].split()

      # RGB - Full res
      rgb_ts = float(line_list[0])
      rgb_filename = line_list[1]
      bgr_np = cv2.imread(seq_dir + rgb_filename)
      rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
      rgb_list.append(rgb_np)

      # Depth - Full res
      depth_ts = float(line_list[2])
      depth_filename = line_list[3]
      depth_np = cv2.imread(seq_dir + depth_filename, cv2.IMREAD_ANYDEPTH)
      depth_np = depth_np.astype(np.float32) / 5000.0
      depth_np[depth_np<=0.0] = float('nan')
      depth_np = np.expand_dims(depth_np, 2)
      depth_list.append(depth_np)
          
      # Pose
      gt_ts = float(line_list[4])
      t_q = np.array(line_list[5:]).astype(np.double)
      t = t_q[:3] # tx, ty, tz
      q = t_q[3:] # qx, qy, qz, qw
      q = q/np.linalg.norm(q)
      pose3 = self.translation_quat_to_pose3(t, q)
      pose_list.append(pose3)
      pose_timestamps.append(gt_ts)

    # TUM1
    # cal3_s2 = gtsam.Cal3_S2(517.306408, 516.469215, 0.0, 318.643040, 255.313989)
    # dist_coeffs = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    # TUM2
    # cal3_s2 = gtsam.Cal3_S2(520.9, 521.0, 0.0, 325.1, 249.7)
    # dist_coeffs = np.array([0.2312,  -0.7849, -0.0033, -0.0001 ,  0.9172 ])

    cal3_s2 = gtsam.Cal3_S2( 535.4 ,  539.2 , 0.0, 320.1,  247.6 )
    dist_coeffs = np.array([0.0,  -0.0, -0.0, -0.0 ,  0.0 ])

    return rgb_list, depth_list, pose_list, pose_timestamps, cal3_s2, dist_coeffs