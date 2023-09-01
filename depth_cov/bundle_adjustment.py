import torch
import torchvision.transforms.functional as TF

import cv2
import numpy as np

import open3d as o3d

from collections import defaultdict

import gtsam
import gtsam_gpdepth

P = gtsam.symbol_shorthand.P
X = gtsam.symbol_shorthand.X
S = gtsam.symbol_shorthand.S

from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.utils.utils import normalize_coordinates, swap_coords_xy
from depth_cov.utils.o3d_utils import frustum_lineset, torch_to_open3d_rgb, torch_to_open3d_depth
import depth_cov.data.depth_resize as depth_resize
from depth_cov.odom.odom_geometry import resize_intrinsics
from depth_cov.data.bundle_adj_dataset import BundleAdjDataset

def translation_quat_to_pose3(t, q):
  rot3 = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2]) # w, x, y, z
  pose3 = gtsam.Pose3(rot3, gtsam.Point3(t))
  return pose3

def filter_frames(start, end, pose_ids, pose_timestamps, poses, point_ids, points, kf_pt_ids, obs, rgb_list):
  pose_ids = pose_ids[start:end]
  pose_timestamps = pose_timestamps[start:end]
  poses = poses[start:end]
  rgb_list = rgb_list[start:end]

  new_obs = np.empty((0,2))
  new_kf_pt_ids = np.empty((0,2), dtype=np.int)
  for i in range(len(pose_ids)):
    obs_mask = (kf_pt_ids[:,0] == pose_ids[i])

    new_obs = np.concatenate((new_obs, obs[obs_mask,:]), axis=0)
    new_kf_pt_ids = np.concatenate((new_kf_pt_ids, kf_pt_ids[obs_mask,:]), axis=0)

  unique_pt_ids, inds, counts = np.unique(new_kf_pt_ids[:,1], return_inverse=True, return_counts=True)

  new_point_ids = []
  new_points = []
  new_obs2 = np.empty((0,2))
  new_kf_pt_ids2 = np.empty((0,2), dtype=np.int)
  for i in range(unique_pt_ids.shape[0]):
    pt_idx = point_ids.index(unique_pt_ids[i])

    new_point_ids.append(point_ids[pt_idx])
    new_points.append(points[pt_idx])

    obs_mask = (new_kf_pt_ids[:,1] == point_ids[pt_idx])

    new_obs2 = np.concatenate((new_obs2, new_obs[obs_mask,:]), axis=0)
    new_kf_pt_ids2 = np.concatenate((new_kf_pt_ids2, new_kf_pt_ids[obs_mask,:]), axis=0)

  return pose_ids, pose_timestamps, poses, new_point_ids, new_points, new_kf_pt_ids2, new_obs2, rgb_list


def get_depth_correspondences(pose_timestamps, depth_dir):
  depth_list = [None] * len(pose_timestamps)

  pairs_file = open(depth_dir + "rgb_depth_pairs.txt")
  lines = pairs_file.readlines()
  for i in range(len(lines)): 
    line = lines[i]
    line_list = line.split()
    rgb_ts = float(line_list[0])

    if rgb_ts in pose_timestamps:
      depth_filename = depth_dir + line_list[3]
      depth_np = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
      depth_np = depth_np.astype(np.float32) / 5000.0
      depth_np[depth_np<=0.0] = float('nan')
      depth_np = np.expand_dims(depth_np, 2)

      depth_torch = torch.permute(torch.from_numpy(depth_np), (2,0,1))
      depth_r = depth_resize.resize_depth(depth_torch, mode="nearest_neighbor", size=[192, 256])

      ind = pose_timestamps.index(rgb_ts)
      depth_list[ind] = depth_r 

  return depth_list 

def load_model(model_path, device):
    model = NonstationaryGpModule.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)
    return model

def run_model(model, rgb_list):
  device = model.device

  gaussian_covs_list = []
  rgb_torch_list = []
  for i in range(len(rgb_list)):
    rgb_float = rgb_list[i].astype(np.float32)/255.0
    rgb_torch = torch.from_numpy(rgb_float).to(device)
    rgb_torch = torch.permute(rgb_torch, (2,0,1)).unsqueeze(0)
    rgb_torch = TF.resize(rgb_torch, [192,256], interpolation = TF.InterpolationMode.BILINEAR, antialias = True)

    with torch.no_grad():
      gaussian_covs = model(rgb_torch)

    gaussian_covs_list.append(gaussian_covs)
    rgb_torch_list.append(rgb_torch[0,...].cpu())

  return gaussian_covs_list, rgb_torch_list

def get_obs(rgb_list, cal3_s2, dist_coeffs):
  num_imgs = len(rgb_list)

  feature_params = dict( maxCorners = 500,
                       qualityLevel = 5e-2,
                       minDistance = 9,
                       blockSize = 7,
                       useHarrisDetector = False,
                       k = 0.04 )
  
  corner_refine_params = dict( winSize = (5, 5),
                              zeroZone = (-1, -1),
                              criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001))

  lk_params = dict( winSize  = (31,31),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  minEigThreshold = 1e-3)

  grays = []
  kps = []
  ids = []
  for i in range(num_imgs):
    gray_curr = cv2.cvtColor(rgb_list[i], cv2.COLOR_RGB2GRAY)

    # Matching if not first frame
    if i == 0:
      pts = cv2.goodFeaturesToTrack(gray_curr, mask=None, **feature_params)
      kps_curr = cv2.cornerSubPix(gray_curr, pts, **corner_refine_params)
      ids_curr = np.arange(kps_curr.shape[0])
    else:
      # Tracking
      pts_tracked, st, err = cv2.calcOpticalFlowPyrLK(grays[-1], gray_curr, kps[-1], None, **lk_params)
      pts_curr = pts_tracked[st[:,0]==1,:,:]
      pts_prev = kps[-1][st[:,0]==1,:,:]
      ids_curr = ids[-1][st[:,0]==1]

      # Outlier rejection
      u1 = cv2.undistortPoints(pts_prev, cameraMatrix=cal3_s2.K(), distCoeffs=dist_coeffs)
      u2 = cv2.undistortPoints(pts_curr, cameraMatrix=cal3_s2.K(), distCoeffs=dist_coeffs)
      E, mask = cv2.findEssentialMat(u1, u2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1e-3)
      mask = np.squeeze(mask.astype(bool))
      kps_curr = pts_curr[mask,:]
      ids_curr = ids_curr[mask]

      # Add new points
      feature_mask = 255*np.ones(gray_curr.shape, dtype=np.uint8)
      for j in range(kps_curr.shape[0]):
        cv2.circle(feature_mask, (kps_curr[j,0,0], kps_curr[j,0,1]), feature_params["minDistance"], 0, -1)
      new_feature_params = feature_params.copy()
      new_feature_params["maxCorners"] = feature_params["maxCorners"] - kps_curr.shape[0]
      if new_feature_params["maxCorners"] > 0:
        new_pts = cv2.goodFeaturesToTrack(gray_curr, mask=feature_mask, **new_feature_params)
        if new_pts is not None:
          new_pts_ref = cv2.cornerSubPix(gray_curr, new_pts, **corner_refine_params)
          max_id = np.max(ids_curr)
          num_new_pts = new_pts_ref.shape[0]
          new_ids = np.arange(max_id+1,max_id+1+num_new_pts)

          kps_curr = np.concatenate((kps_curr, new_pts_ref), axis=0)
          ids_curr = np.concatenate((ids_curr, new_ids), axis=0)


    # print(kps_curr.shape)

    kps.append(kps_curr)
    ids.append(ids_curr)
    grays.append(gray_curr)
  
  return kps, ids

def undistort_keypoints(keypoints_list, cal3_s2, dist_coeffs):
  keypoints_list_u = []
  for i in range(len(keypoints_list)):
    kp_u = cv2.undistortPoints(keypoints_list[i], cameraMatrix=cal3_s2.K(), distCoeffs=dist_coeffs, P=cal3_s2.K())
    keypoints_list_u.append(kp_u)
  return keypoints_list_u

def get_normalized_coordinates(rgb_list, keypoints_list, device):
  coords_norm_list = []
  for i in range(len(rgb_list)):
    shape = rgb_list[i].shape[0:2]
    coords_torch = torch.from_numpy(keypoints_list[i]).to(device)
    coords_torch = torch.permute(coords_torch, (1,0,2))
    coords_torch = swap_coords_xy(coords_torch)
    coords_norm = normalize_coordinates(coords_torch, shape).unsqueeze(0)
    coords_norm_list.append(coords_norm)
  return coords_norm_list

def get_img_correspondence(img_list, coords_norm_list, interp_mode, device):
  sampled_vals_list = []
  for i in range(len(img_list)):
    img_torch = torch.from_numpy(img_list[i])
    img_torch = torch.permute(img_torch, (2, 0, 1)).unsqueeze(0)
    img_torch = img_torch.to(device)
    sampled_vals = torch.nn.functional.grid_sample(img_torch, coords_norm_list[i], 
      mode=interp_mode, padding_mode='zeros', align_corners=False)
    sampled_vals = torch.permute(sampled_vals.squeeze(2), (0,2,1))
    sampled_vals_list.append(sampled_vals)

  return sampled_vals_list

def create_depth_priors(model, gaussian_covs_list, coords_norm_list, ids_list):
  factors = []
  for i in range(len(gaussian_covs_list)):
    with torch.no_grad():
      L, _ = model.get_covariance_chol(gaussian_covs_list[i], -1, coords_norm_list[i][0,...])
      L = L[0,...].double()

    base_noise_model = gtsam.noiseModel.Isotropic.Sigma(ids_list[i].shape[0], 1e0)
    point_keys = [P(ind) for ind in ids_list[i]]
    factor = gtsam_gpdepth.GpDepthFactor3D(base_noise_model, X(i), point_keys, S(i), L.cpu().numpy())
    factors.append(factor)
  
  return factors

def init_landmarks(poses_init, keypoints_list, ids_list, cal3_s2):
  landmark_values = gtsam.Values()
  
  # First pass: collect all observations for each id
  obs_dict = defaultdict(list)
  for i in range(len(keypoints_list)):
    for j in range(len(keypoints_list[i])):
      id = ids_list[i][j]
      keypoint = keypoints_list[i][j,0,:]
      obs_dict[id].append([i, keypoint]) # Cam id, keypoint pair

  # Second pass: Triangulate
  valid_count = 0
  invalid_count = 0
  for landmark_id, camid_pt_list in obs_dict.items():
    cameras = []
    measurements = []
    for i in range(len(camid_pt_list)):
      cam_id, pt = camid_pt_list[i]
      camera = gtsam.PinholeCameraCal3_S2(poses_init.atPose3(X(cam_id)), cal3_s2)
      cameras.append(camera)
      measurements.append(pt)
    camera_set = gtsam.CameraSetCal3_S2(cameras)
    meas_set = gtsam.Point2Vector(measurements)
    try:
      point3_init = gtsam.triangulatePoint3(camera_set, meas_set, rank_tol=1e-1, optimize=True, model=None)
      key = P(landmark_id)
      landmark_values.insert(key, point3_init)
      valid_count += 1
    except RuntimeError:
      invalid_count += 1

  print("Valid landmarks: ", valid_count, " Invalid landmarks: ", invalid_count)

  return landmark_values

def init_poses(pose_list, noise):
  pose_values = gtsam.Values()
  for i in range(len(pose_list)):
    rot_delta = noise[0] * np.random.randn(3, 1)
    t_delta = noise[1] * np.random.randn(3, 1)
    delta = np.concatenate((rot_delta, t_delta), axis=0)

    pose_values.insert(X(i), pose_list[i].retract(delta))
  return pose_values

def filter_observations(landmarks_init, keypoints_list, coords_norm_list, ids_list):
  valid_inds = []  
  for i in range(len(ids_list)):
    valid_inds.append([])
    for j in range(len(ids_list[i])):
      id = ids_list[i][j]
      key = P(id)
      if landmarks_init.exists(key):
        valid_inds[i].append(j)

  constrained = True
  keypoints_filt_list = []
  coords_norm_filt_list = []
  ids_filt_list = []
  for i in range(len(valid_inds)):
    inds = valid_inds[i]
    keypoints_filt_list.append(keypoints_list[i][inds,:,:])
    coords_norm_filt_list.append(coords_norm_list[i][:,:,inds,:])
    ids_filt_list.append(ids_list[i][inds])

    if keypoints_filt_list[-1].shape[0]  < 10:
      constrained = False

  return constrained, keypoints_filt_list, coords_norm_filt_list, ids_filt_list

def generate_depth_imgs(model, gaussian_covs_list, coords_norm_list, depth_prior_factors, depth_img_list, values, img_size):
  device = model.device

  pred_depth_list = [None] * len(depth_img_list)

  for i in range(len(depth_prior_factors)):
    depth_pred = depth_prior_factors[i].getDepthVector(values)
    depth_pred = torch.from_numpy(depth_pred).to(device)
    depth_pred = depth_pred.unsqueeze(0).unsqueeze(-1)

    depth_pred = torch.clamp(depth_pred, 1e-2)

    log_depth_pred = torch.log(depth_pred)
    log_depth_pred = log_depth_pred.to(device).float()

    # NOTE: Bundle adjustment without prior does not have scale variable, so we will use the median of the sparse reprojected depths
    mean_log_depth = torch.log(torch.median(depth_pred, dim=1, keepdim=True).values).float()


    with torch.no_grad():
      log_depth_img, _, _, _ = model.condition_level(gaussian_covs_list[i], -1, coords_norm_list[i][0,...], log_depth_pred, mean_log_depth, img_size)

    pred_depth_list[i] = torch.exp(log_depth_img).squeeze(0)

  return pred_depth_list
  
def bundle_adjustment(poses, poses_init, landmarks_init, keypoints_list, ids_list, cal3_s2, depth_prior_factors, add_gp_depth_factors):

  graph = gtsam.NonlinearFactorGraph()
  values_init = gtsam.Values()

  values_init.insert(poses_init)
  values_init.insert(landmarks_init)

  pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-8)

  # Pose prior
  prior_factor0 = gtsam.PriorFactorPose3(X(0), poses[0], pose_noise)
  graph.push_back(prior_factor0)

  # Range for scale prior
  range_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]))
  last_pose_ind = len(poses)-1
  range_factor = gtsam.RangeFactorPose3(X(0), X(last_pose_ind), poses[0].range(poses[last_pose_ind]), range_noise)
  graph.push_back(range_factor)

  # Measurements
  measurement_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.Isotropic.Sigma(2, 1.0))

  for i in range(len(keypoints_list)):
    pose_key = X(i)
    for j in range(len(keypoints_list[i])):
      id = ids_list[i][j]
      landmark_key = P(id)
      if landmarks_init.exists(landmark_key):
        obs = keypoints_list[i][j,0,:]
        factor = gtsam.GenericProjectionFactorCal3_S2(
                    obs, measurement_noise, pose_key, landmark_key, cal3_s2)
        graph.push_back(factor)

  # Depth factors and scale variables
  if add_gp_depth_factors:
    for i in range(len(poses)):
      graph.push_back(depth_prior_factors[i])
      values_init.insert(S(i), np.array([0.0]))
      depths = depth_prior_factors[i].getDepthVector(values_init)
      median_log_depth = np.array([np.log(np.median(depths))])
      values_init.update(S(i), median_log_depth)

      if i == 0:
        scale_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]))
        scale_prior = gtsam.PriorFactorVector(S(i), median_log_depth, scale_noise)
        graph.push_back(scale_prior)

  # Optimize
  params = gtsam.LevenbergMarquardtParams()
  params.setVerbosity('ERROR') # TERMINATION, ERROR
  params.setlambdaInitial(1e+6)
  optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values_init, params)
  success = True
  try:
    result = optimizer.optimize()
    poses_extract = gtsam.utilities.extractPose3(result)
    poses_opt = np.zeros((poses_extract.shape[0], 4, 4))
    poses_opt[:,0,0:3] = poses_extract[:,0:3]
    poses_opt[:,1,0:3] = poses_extract[:,3:6]
    poses_opt[:,2,0:3] = poses_extract[:,6:9]
    poses_opt[:,0:3,3] = poses_extract[:,9:12]
    poses_opt[:,3,3] = 1.0
    landmarks_opt = gtsam.utilities.extractPoint3(result)[:landmarks_init.size(),:]
  except RuntimeError: # Factor graph disjoint due to no triangulated points?
    success = False
    result = None
    poses_opt = None
    landmarks_opt = None

  return success, result, poses_opt, landmarks_opt

def plot_results(poses, landmarks, cal3_s2, rgb_torch_list, depth_list):

  intrinsics = torch.from_numpy(cal3_s2.K()).float()

  image_scale_factors = [192.0/480.0, 256.0/640.0]
  intrinsics_depth_img = resize_intrinsics(intrinsics, image_scale_factors)
  o3d_intrinsics = o3d.core.Tensor(intrinsics_depth_img.numpy())

  rgbd_imgs = []
  for i in range(len(rgb_torch_list)):
    if depth_list[i] is not None:
      rgb_o3d = torch_to_open3d_rgb(rgb_torch_list[i])
      depth_o3d = torch_to_open3d_depth(depth_list[i].cpu())
      rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d.to_legacy(), depth_o3d.to_legacy(), depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
      rgbd_imgs.append(rgbd_img)

  # Bundle adjustment point cloud
  ba_pcd = o3d.geometry.PointCloud()
  ba_pcd.points = o3d.utility.Vector3dVector(landmarks)
  ba_pcd.paint_uniform_color((1.0, 0.0, 0.0))

  o3d.visualization.gui.Application.instance.initialize()
  vis = o3d.visualization.O3DVisualizer(title="cloud", width=640, height=480)
  vis.show_menu(True)
  o3d.visualization.gui.Application.instance.add_window(vis)

  mesh_mat = o3d.visualization.rendering.MaterialRecord()
  mesh_mat.shader = "defaultLit"

  volume = o3d.pipelines.integration.ScalableTSDFVolume(
      voxel_length=0.01,
      sdf_trunc=0.1,
      color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
  )

  intrinsics2 = o3d.camera.PinholeCameraIntrinsic(256, 192, intrinsics_depth_img.numpy())
  for i in range(len(rgbd_imgs)):
    volume.integrate(rgbd_imgs[i], intrinsics2, np.linalg.inv(poses[i,:,:]))

  mesh = volume.extract_triangle_mesh()
  mesh.compute_vertex_normals()

  vis.add_geometry("mesh", mesh, mesh_mat)

  pcd_mat = o3d.visualization.rendering.MaterialRecord()
  pcd_mat.shader = 'defaultUnlit'
  pcd_mat.sRGB_color = True
  pcd_mat.point_size = 3.0
  vis.add_geometry('ba_cloud', ba_pcd, pcd_mat)

  pcd_plane_mat = o3d.visualization.rendering.MaterialRecord()
  pcd_plane_mat.shader = 'defaultUnlit'
  pcd_plane_mat.sRGB_color = True
  pcd_plane_mat.point_size = 1.0

  # Cameras
  img_size = [480, 640]
  for i in range(poses.shape[0]):
    frustum = frustum_lineset(intrinsics, img_size, poses[i,:,:], scale=0.1)
    vis.add_geometry('f' + str(i), frustum)

  base_pose = poses[0,:,:]
  eye = base_pose[:3,3]
  center = eye + base_pose[0:3,0:3] @ np.array([0,0,5.0])
  up = -base_pose[:3,1]
  vis.setup_camera(60, center, eye, up)
  o3d.visualization.gui.Application.instance.run()

def run_ba(data, model, add_gp_depth_factors, pose_noise):

  rgb_list, depth_list, pose_list, pose_timestamps, cal3_s2, dist_coeffs = data

  # Frontned
  keypoints_list, ids_list = get_obs(rgb_list, cal3_s2, dist_coeffs)
  keypoints_undistored_list = undistort_keypoints(keypoints_list, cal3_s2, dist_coeffs)

  # Correspondence for evaluation
  coords_norm_list = get_normalized_coordinates(rgb_list, keypoints_list, device)

  # Values init and filtering of observations
  poses_init = init_poses(pose_list, pose_noise)
  landmarks_init = init_landmarks(poses_init, keypoints_list, ids_list, cal3_s2)
  constrained, keypoints_filt_list, coords_norm_filt_list, ids_filt_list = filter_observations(landmarks_init, keypoints_undistored_list, coords_norm_list, ids_list)

  if constrained:

    # Setup prior
    gaussian_covs_list, rgb_torch_list = run_model(model, rgb_list)
    depth_prior_factors = create_depth_priors(model, gaussian_covs_list, coords_norm_filt_list, ids_filt_list)

    # Bundle adjustment
    success, values_opt, poses_opt, landmarks_opt = bundle_adjustment(pose_list, poses_init, landmarks_init, keypoints_filt_list, ids_filt_list, cal3_s2, depth_prior_factors, add_gp_depth_factors)

    # Evaluation
    pred_depth_imgs = generate_depth_imgs(model, gaussian_covs_list, coords_norm_filt_list, depth_prior_factors, depth_list, values_opt, rgb_torch_list[0].shape[-2:])

    plot_results(poses_opt, landmarks_opt, cal3_s2, rgb_torch_list, pred_depth_imgs)

  return



if __name__ == "__main__":

  np.random.seed(1)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  window_size = 100
  add_gp_depth_factors = True

  # Setup input data
  base_dir = "path/tum/"

  seqs = [
    "rgbd_dataset_freiburg3_long_office_household/"
  ]

  dataset = BundleAdjDataset(base_dir, seqs, window_size=window_size)

  # Setup model
  model_path = "models/scannet.ckpt"
  model = load_model(model_path, device)

  pose_noise = [1e-2, 1e-2]
  # pose_noise = [0.0, 0.0]

  # Main loop
  num_seqs = dataset.__len__()
  num_imgs = window_size * num_seqs


  for i in range(num_seqs):
    # if i != 1:
    if i != num_seqs - 5:
      continue
    # Get data
    data = dataset.__getitem__(i)
    
    run_ba(data, model, add_gp_depth_factors, pose_noise)