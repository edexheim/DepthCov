import open3d as o3d
import torch
import numpy as np
import open3d.visualization.rendering as rendering
import cv2

from depth_cov.utils.utils import unnormalize_coordinates, get_coord_img
from depth_cov.utils.image_processing import ImageGradientModule

def enable_widget(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

def torch_to_open3d_rgb(rgb):
  rgb_np = torch.permute(rgb, (1,2,0)).numpy()
  rgb_np_uint8 = np.ascontiguousarray((rgb_np * 255).astype(np.uint8))
  rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
  return rgb_img

def torch_to_open3d_rgb_with_points(rgb, points_norm, radius, color):
  coords = unnormalize_coordinates(points_norm, rgb.shape[-2:])
  rgb_np = torch.permute(rgb, (1,2,0)).numpy()
  rgb_np_circles = rgb_np.copy()

  for i in range(coords.shape[1]):
    pt = (int(coords[0,i,1].item()), int(coords[0,i,0].item()))
    cv2.circle(rgb_np_circles, pt, radius, color, thickness=-1)

  rgb_np_uint8 = np.ascontiguousarray((rgb_np_circles * 255).astype(np.uint8))
  rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
  return rgb_img

def torch_to_open3d_depth(depth):
  depth_filt = depth.clone()
  depth_filt[depth.isnan()] = 0.0
  depth_np = depth_filt[0,...].numpy()
  depth_img = o3d.t.geometry.Image(depth_np)
  return depth_img

def torch_to_open3d_normal_color(normals):
  normals_color = 0.5*(1.0 + normals)
  rgb_np_uint8 = np.ascontiguousarray((normals_color.numpy() * 255).astype(np.uint8))
  rgb_img = o3d.t.geometry.Image(rgb_np_uint8)
  return rgb_img

def get_points_and_normals(depth, K):
  device = depth.device

  coords = get_coord_img(depth.shape[-2:], device, batch_size=1)
  tmp1 = (coords[...,0] - K[1,2])/K[1,1]
  tmp2 = (coords[...,1] - K[0,2])/K[0,0]
  rays = torch.empty((depth.shape[-2], depth.shape[-1], 3), device=device)
  rays[...,0] = tmp2
  rays[...,1] = tmp1
  rays[...,2] = 1.0
  z = torch.permute(depth, (1,2,0))
  P = z*rays

  grad_module = ImageGradientModule(3, device=device, dtype=P.dtype)
  gx, gy = grad_module(torch.permute(P, (2,0,1)).unsqueeze(0))
  gx = torch.permute(gx.squeeze(0), (1,2,0))
  gy = torch.permute(gy.squeeze(0), (1,2,0))
  n_dir = torch.cross(gx, gy)
  n_dir = torch.div(n_dir, torch.linalg.norm(n_dir, dim=-1, keepdim=True))
  normals = n_dir

  # Set edges since undefined
  default_normal = torch.tensor([[0.0, 0.0, 1.0]])
  normals[0,:,:] = default_normal
  normals[-1,:,:] = default_normal
  normals[:,0,:] = default_normal
  normals[:,-1,:] = default_normal

  return P, normals, rays

def setup_cloud(rgb, P, normals, rays, pose):

  colors = torch.permute(rgb, (1,2,0))

  colors_np = colors.numpy()
  points_np = P.numpy()
  normals_np = normals.numpy()
  rays_np = rays.numpy()

  # Trim edges since gradient undefined
  points_img = points_np[1:-1,1:-1,:]
  new_size = points_img.shape[0]*points_img.shape[1]
  colors_np = np.resize(colors_np[1:-1,1:-1,:], (new_size,3)).astype(np.float64)
  points_np = np.resize(points_img, (new_size,3)).astype(np.float64)
  normals_np = np.resize(normals_np[1:-1,1:-1,:], (new_size,3)).astype(np.float64)
  rays_np = np.resize(rays_np[1:-1,1:-1,:], (new_size,3)).astype(np.float64)

  # Transform
  pose_np = pose.numpy().astype(np.float64)
  points_np = points_np @ pose_np[0:3,0:3].T + pose_np[0:3,3]
  # normals_np = normals_np @ pose_np[0:3,0:3].T  

  return colors_np, points_np, normals_np, rays_np

# TODO: Can precalculate lots of this including gradient, then multiply by depth at end?
def rgb_depth_to_pcd(rgb, depth, pose, K):
  P, normals, rays = get_points_and_normals(depth, K)
  colors_np, points_np, normals_np, rays_np = setup_cloud(rgb, P, normals, rays, pose)

  # Construct t PointCloud
  pcd = o3d.t.geometry.PointCloud()
  pcd.point.positions = o3d.core.Tensor(points_np)
  pcd.point.normals = o3d.core.Tensor(normals_np)
  pcd.point.colors = o3d.core.Tensor(colors_np)

  return pcd

def precompute_mesh_topology(height, width):
  rows = height-2
  cols = width-2
  num_triangles = 2 * (rows-1) * (cols-1)
  t1 = np.arange((rows-1)*cols, dtype=np.int64)
  t2 = t1 + 1
  t3 = t2 + (cols-1)
  triangle_np1 = np.stack((t1,t2,t3), axis=1)
  triangle_np2 = np.stack((t3,t2,t3+1), axis=1)
  delete_ind = np.arange(cols-1, (rows-1)*cols, cols)
  # print(triangle_np1.shape, delete_ind.shape)
  triangle_np1 = np.delete(triangle_np1, delete_ind, axis=0)
  triangle_np2 = np.delete(triangle_np2, delete_ind, axis=0)
  triangles_np = np.concatenate((triangle_np1, triangle_np2), axis=0)
  return triangles_np

def rgb_depth_to_mesh(rgb, depth, pose, K, triangles_np):
  P, normals, rays = get_points_and_normals(depth, K)
  colors_np, points_np, normals_np, rays_np = setup_cloud(rgb, P, normals, rays, pose)

  rays_norm = rays_np/np.linalg.norm(rays_np, axis=1, keepdims=True)
  # Check dot product between ray and normal
  invalid_mask = np.abs(np.sum(normals_np*rays_np, axis=1)) < 0.10
  invalid_vertex_ids = np.argwhere(invalid_mask)

  triangle_mesh = o3d.geometry.TriangleMesh()
  triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles_np)
  triangle_mesh.vertices = o3d.utility.Vector3dVector(points_np)
  triangle_mesh.vertex_normals = o3d.utility.Vector3dVector(normals_np)
  triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

  triangle_mesh.remove_vertices_by_index(invalid_vertex_ids)

  return triangle_mesh, normals

def create_empty_pcd(max_points):
  pcd = o3d.t.geometry.PointCloud(
      o3d.core.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
  pcd.point["colors"] = o3d.core.Tensor(
      np.zeros((max_points, 3), dtype=np.float32))
  return pcd

def create_empty_lineset(max_points, max_lines, color):
  lineset = o3d.geometry.LineSet()
  np_points = 0.01*np.random.rand(max_points, 3)
  lineset.points = o3d.utility.Vector3dVector(np_points)
  np_lines = np.zeros((max_lines, 2), dtype=np.int32)
  np_lines[:,1] = 1
  lineset.lines = o3d.utility.Vector2iVector(np_lines)
  lineset.paint_uniform_color(color)
  return lineset

def frustum_lineset(intrinsics, img_size, pose, scale=0.2):
  frustum = o3d.geometry.LineSet.create_camera_visualization(
    img_size[1], img_size[0], intrinsics.numpy(), 
    np.linalg.inv(pose), scale=scale)
  return frustum

def poses_to_traj_lineset(poses):
  n = poses.shape[0]
  points = poses[:,:3,3]
  lines = np.stack((np.arange(0,n-1,dtype=np.int32), np.arange(1,n,dtype=np.int32)), axis=1)
  
  lineset = o3d.geometry.LineSet()
  lineset.points = o3d.utility.Vector3dVector(points)
  lineset.lines = o3d.utility.Vector2iVector(lines)
  return lineset

def get_reference_frame_lineset(poses, one_way_poses, ind_pairs):
  n = len(ind_pairs[0])
  points_np = np.concatenate((poses[ind_pairs[0],:3,3], one_way_poses[ind_pairs[1],:3,3]), axis=0)
  lines_ind1 = np.arange(0,n,dtype=np.int32)
  lines_np = np.stack((lines_ind1, lines_ind1+n), axis=1)

  lineset = o3d.geometry.LineSet()
  lineset.points = o3d.utility.Vector3dVector(points_np)
  lineset.lines = o3d.utility.Vector2iVector(lines_np)
  return lineset

def get_one_way_lineset(poses, one_way_poses, ref_inds, intrinsics, img_size, color, frustum_scale):
  lineset = get_reference_frame_lineset(poses, one_way_poses, ref_inds)
  for i in range(one_way_poses.shape[0]):
    lineset += frustum_lineset(intrinsics, img_size, one_way_poses[i,...], frustum_scale)
  lineset.paint_uniform_color(color)
  return lineset

def get_traj_lineset(poses, intrinsics, img_size, color, frustum_scale, frustum_mode="last"):
  lineset = poses_to_traj_lineset(poses)
  if frustum_mode == "all":
    for i in range(poses.shape[0]):
      lineset += frustum_lineset(intrinsics, img_size, poses[i,...], frustum_scale)
  elif frustum_mode == "last":
    lineset += frustum_lineset(intrinsics, img_size, poses[-1,...], frustum_scale)
  elif frustum_mode == "none":
    pass

  lineset.paint_uniform_color(color)
  return lineset

# NOTE: Assumes canonical world frame direction from first pose
def pose_to_camera_setup(pose, pose_init, scale):

  # Assume original negative y axis is up
  up_global = -pose_init[:3,1]
  # up_global = np.array([0, 0, 1.0])

  # Camera coordinates
  center = scale*np.array([0, 0.0, 0.5]) # Point camera is looking at
  eye = scale*np.array([0, -0.0, -0.5]) # Camera location

  def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

  def eul2rot(theta):
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R

  # Transform into world coordinates ()
  R = pose[:3,:3]
  t = pose[:3,3]

  zyx = rot2eul(R)
  # zyx[0] = 0.0 # Roll
  # zyx[2] = 0.0 # Pitch
  R = eul2rot(zyx)

  center = R @ center + t
  eye = R @ eye + t

  return center, eye, up_global