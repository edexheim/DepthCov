import torch # Must import before Open3D when using CUDA!
from torch.utils.data import DataLoader

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

import time
import threading

from depth_cov.odom.odom_datasets import odom_collate_fn
from depth_cov.odom.DepthOdometry import DepthOdometry
from depth_cov.odom.DepthMapper import DepthMapper
from depth_cov.odom.multiprocessing import TupleTensorQueue, release_data
from depth_cov.utils.o3d_utils import enable_widget, rgb_depth_to_pcd, rgb_depth_to_mesh, torch_to_open3d_rgb, torch_to_open3d_depth, torch_to_open3d_normal_color, get_traj_lineset, pose_to_camera_setup, create_empty_pcd, create_empty_lineset, torch_to_open3d_rgb_with_points, frustum_lineset, get_one_way_lineset, precompute_mesh_topology
from depth_cov.utils.io import save_traj
from depth_cov.utils.utils import init_gpu, str_to_dtype


class OdomWindow():
  def __init__(self, is_live, viz_cfg, slam_cfg, dataset):
    self.is_live = is_live
    self.cfg = viz_cfg
    self.viz_device = self.cfg["device"]
    self.viz_dtype = str_to_dtype(self.cfg["dtype"])

    self.num_frames = slam_cfg["mapping"]["graph"]["num_keyframes"]

    self.window = gui.Application.instance.create_window(
        'Monocular Odometry', width=1920, height=1080) # 1280 x 720 for recording
    em = 10

    # em = self.window.theme.font_size

    spacing = int(np.round(0.25 * em))
    vspacing = int(np.round(0.5 * em))

    margins = gui.Margins(left=spacing, top=vspacing, right=spacing, bottom=vspacing)

    self.ctrl_panel = gui.Vert(spacing, margins)

    ## Application control

    # Resume/pause
    resume_button = gui.ToggleSwitch("Resume/Pause")
    resume_button.set_on_clicked(self._on_pause_switch)
    resume_button.is_on = True

    # Next frame
    self.idx_panel = gui.Horiz(em)
    self.idx_label = gui.Label('Idx: {:20d}'.format(0))
    next_frame_button = gui.Button("Next frame")
    next_frame_button.vertical_padding_em = 0.0
    next_frame_button.set_on_clicked(self._on_press)
    self.idx_panel.add_child(self.idx_label)
    self.idx_panel.add_child(next_frame_button)

    # Point cloud viz
    normals_button = gui.ToggleSwitch("Show Normals")
    normals_button.set_on_clicked(self._on_normals_switch)
    normals_button.is_on = False

    # Point cloud control
    self.follow_lv = gui.ListView()
    cloud_viz_options = ["Estimated", "None"]
    self.follow_lv.set_items(cloud_viz_options)
    self.follow_lv.selected_index = self.follow_lv.selected_index + 1  # initially is -1, so now 1
    self.follow_lv.set_max_visible_items(2)
    self.follow_lv.set_on_selection_changed(self._on_list)
    self.follow_val = cloud_viz_options[self.follow_lv.selected_index]

    # Reference frame index slider
    self.adjustable_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))
    ref_frame_label = gui.Label('')
    self.ref_frame_slider = gui.Slider(gui.Slider.INT)
    self.ref_frame_slider.set_limits(0, self.num_frames-1)
    self.ref_frame_slider.int_value = int(0)
    self.ref_frame_slider.set_on_value_changed(self._on_ref_frame_slider)
    self.adjustable_prop_grid.add_child(ref_frame_label)
    self.adjustable_prop_grid.add_child(self.ref_frame_slider)

    ## Tabs
    tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
    tabs = gui.TabControl()

    ### Data tab
    tab_data = gui.Vert(0, tab_margins)
    self.curr_rgb_w = gui.ImageWidget()
    self.kf_rgb_w = gui.ImageWidget()
    self.kf_depth_w = gui.ImageWidget()
    self.kf_normals_w = gui.ImageWidget()
    tab_data.add_child(self.curr_rgb_w)
    tab_data.add_fixed(vspacing)
    tab_data.add_child(self.kf_rgb_w)
    tab_data.add_fixed(vspacing)
    tab_data.add_child(self.kf_depth_w)
    tab_data.add_fixed(vspacing)
    tab_data.add_child(self.kf_normals_w)

    ### Add panel children
    self.ctrl_panel.add_child(resume_button)
    self.ctrl_panel.add_fixed(vspacing)
    self.ctrl_panel.add_child(self.idx_panel)
    self.ctrl_panel.add_fixed(vspacing)
    self.ctrl_panel.add_child(self.follow_lv)
    self.ctrl_panel.add_fixed(vspacing)
    self.ctrl_panel.add_child(normals_button)
    self.ctrl_panel.add_fixed(vspacing)

    self.ctrl_panel.add_child(tab_data)

    self.widget3d = gui.SceneWidget()

    self.fps_panel = gui.Vert(spacing, margins)
    self.output_fps = gui.Label('FPS: 0.0')
    self.fps_panel.add_child(self.output_fps)

    self.num_inducing_panel = gui.Vert(spacing, margins)
    self.num_inducing_pts_label = gui.Label('Keyframe # Inducing Points: 0')
    self.num_inducing_panel.add_child(self.num_inducing_pts_label)

    self.window.add_child(self.ctrl_panel)
    self.window.add_child(self.widget3d)
    self.window.add_child(self.fps_panel)
    self.window.add_child(self.num_inducing_panel)

    self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
    self.widget3d.scene.set_background([1, 1, 1, 0])

    # self.widget3d.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, 
        # (0.577, -0.577, -0.577))

    self.window.set_on_layout(self._on_layout)
    self.window.set_on_close(self._on_close)

    # Application variables
    self.is_running = resume_button.is_on
    self.is_done = False
    self.advance_one_frame = False
    
    # Visualization variables
    self.update_keyframe_render_viz = False
    self.normalize_est_depth = True

    # Point cloud mat
    self.pcd_mat = rendering.MaterialRecord()
    self.pcd_mat.point_size = 3.0
    self.pcd_mat.shader = self.cfg["pcd_shader"]
    # self.pcd_mat.sRGB_color = True

    # Line mat
    self.line_mat = rendering.MaterialRecord()
    self.line_mat.shader = "unlitLine"
    self.line_mat.line_width = 2.0
    self.line_mat.transmission = 1.0

    self.idx = 0

    self.scale = 1.0
    self.pose_init = torch.eye(4)

    # Start processes
    torch.multiprocessing.set_start_method("spawn")
    
    self.dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False, collate_fn=odom_collate_fn)

    self.setup_slam_processes(slam_cfg)

    img_size = self.get_img_size()
    self.triangles = precompute_mesh_topology(height=img_size[0], width=img_size[1])

    # Start running
    threading.Thread(name='UpdateMain', target=self.update_main).start()
    # self.update_main()

  def setup_slam_processes(self, slam_cfg):

    # Setup SLAM processes
    intrinsics = self.get_intrinsics()
    img_size = self.get_img_size()
    self.waitev = torch.multiprocessing.Event()
    self.tracking = DepthOdometry(slam_cfg["tracking"], intrinsics, img_size, self.waitev)
    self.mapping = DepthMapper(slam_cfg["mapping"], intrinsics, self.waitev)
    # Setup queues
    rgb_queue = TupleTensorQueue(self.tracking.device, self.tracking.dtype, maxsize=1)
    pose_viz_queue = TupleTensorQueue(self.viz_device, self.viz_dtype) # Only want recent
    frame_queue = TupleTensorQueue(self.mapping.device, self.mapping.dtype, maxsize=1)
    kf_ref_queue = TupleTensorQueue(self.tracking.device, self.tracking.dtype) # Only want recent
    kf_viz_queue = TupleTensorQueue(self.viz_device, self.viz_dtype) # Only want recent
    
    self.rgb_queue = rgb_queue
    self.tracking.rgb_queue = rgb_queue
    self.tracking.tracking_pose_queue = pose_viz_queue
    self.tracking_pose_queue = pose_viz_queue
    self.tracking.frame_queue = frame_queue
    self.mapping.frame_queue = frame_queue
    self.mapping.kf_ref_queue = kf_ref_queue
    self.tracking.kf_ref_queue = kf_ref_queue
    self.mapping.kf_viz_queue = kf_viz_queue
    self.kf_viz_queue = kf_viz_queue

    # Warmup GPU for main process (others handle their own in run)
    init_gpu(self.viz_device)
    
  def start_slam_processes(self):

    self.tracking_done = False
    self.mapping_done = False

    print("Starting tracking and mapping processes...")
    # self.dataloader.start()
    self.tracking.start()
    self.mapping.start()
    print("Done.")

  def shutdown_slam_processes(self):
    self.waitev.set()
    print("Joining mapping...")
    self.mapping.join()
    print("Joining tracking...")
    self.tracking.join()
    print("Done.")

  def _on_press(self):
    self.advance_one_frame = True

  def _on_list(self, new_val, is_dbl_click):
    self.follow_val = new_val

  def _on_layout(self, ctx):
    em = ctx.theme.font_size
    panel_width = 20 * em
    rect = self.window.content_rect

    self.ctrl_panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

    x = self.ctrl_panel.frame.get_right()
    self.widget3d.frame = gui.Rect(x, rect.y,
                                  rect.get_right() - x, rect.height)

    fps_panel_width = 7 * em
    fps_panel_height = 2 * em
    self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                    rect.y, fps_panel_width,
                                    fps_panel_height)

    inducing_panel_width = 14 * em
    inducing_panel_height = 2 * em
    self.num_inducing_panel.frame = gui.Rect(self.ctrl_panel.frame.get_right(),
                                    rect.y, inducing_panel_width,
                                    inducing_panel_height)


  # Toggle callback: application's main controller
  def _on_pause_switch(self, is_on):
    self.is_running = is_on

  def _on_normals_switch(self, is_on):
    if is_on:
      self.pcd_mat.shader = "normals"
    else:
      self.pcd_mat.shader = self.cfg["pcd_shader"]
    self.widget3d.scene.modify_geometry_material("est_points", self.pcd_mat)

  def save_traj(self):
    if self.dataloader.dataset.seq_path is not None:
      tmp = self.dataloader.dataset.seq_path.rsplit("/", 3)
      tracked_filename = "./results/" + tmp[1] + "_" + tmp[2] + ".txt"
      save_traj(tracked_filename, self.timestamps, self.est_poses)
      print("Saved trajectory.")

  def _on_ref_frame_slider(self, val):
    self.update_keyframe_render_viz = True

  def _on_start(self):
    enable_widget(self.adjustable_prop_grid, True)

  def _on_close(self):
    self.is_done = True

    self.save_traj()

    return True

  def setup_camera_view(self, pose, base_pose):
    center, eye, up = pose_to_camera_setup(pose, base_pose, self.scale)
    self.widget3d.look_at(center, eye, up)

  def get_intrinsics(self):
    return self.dataloader.dataset.intrinsics

  def get_img_size(self):
    return self.dataloader.dataset.img_size

  def init_render(self, rgb):

    rgb_o3d = torch_to_open3d_rgb(rgb[0,...])
    self.curr_rgb_w.update_image(rgb_o3d.to_legacy())

    rgb_o3d = torch_to_open3d_rgb(torch.ones_like(rgb[0,...]))
    self.kf_rgb_w.update_image(rgb_o3d.to_legacy())

    kf_depth_img = torch_to_open3d_depth(torch.ones_like(rgb[0,0:1,...]))
    kf_depth_color = kf_depth_img.colorize_depth(
      self.cfg["depth_scale"], self.cfg["depth_min"], self.cfg["depth_max"])
    self.kf_depth_w.update_image(kf_depth_color.to_legacy())

    kf_normals_img = 0.577*torch.ones((rgb.shape[2], rgb.shape[3], rgb.shape[1]))
    kf_normals_color = torch_to_open3d_normal_color(kf_normals_img)
    self.kf_normals_w.update_image(kf_normals_color.to_legacy())

    self.window.set_needs_layout()

    # Way to set FoV of rendering
    fov = 60.0
    bounds = self.widget3d.scene.bounding_box
    self.widget3d.setup_camera(fov, bounds, bounds.get_center())

    self.frumstum_scale = self.cfg["frustum_const"]

  def update_idx_text(self):
    self.idx_label.text = 'Idx: {:8d}'.format(self.idx)

  def update_curr_image_render(self, rgb):
    rgb_o3d = torch_to_open3d_rgb(rgb[0,...])
    self.curr_rgb_w.update_image(rgb_o3d.to_legacy())

  def update_keyframe_render(self, kf_timestamps, kf_rgb, kf_poses, kf_depth, kf_sparse_coords_norm, one_way_poses, kf_pairs, one_way_pairs):

    kf_rgb = kf_rgb.squeeze(0)
    kf_depth = kf_depth.squeeze(0)

    kf_rgb_o3d = torch_to_open3d_rgb_with_points(kf_rgb, kf_sparse_coords_norm, 
        radius=self.cfg["inducing_point_radius"], color=self.cfg["inducing_point_color"])

    self.kf_rgb_w.update_image(kf_rgb_o3d.to_legacy())

    self.scale = torch.median(kf_depth).item()

    kf_depth_img = torch_to_open3d_depth(kf_depth.float())
    kf_depth_color = kf_depth_img.colorize_depth(
        self.scale, self.cfg["depth_min"], self.cfg["depth_max"])
    self.kf_depth_w.update_image(kf_depth_color.to_legacy())

    # Show points
    o3d_intrinsics = o3d.core.Tensor(self.get_intrinsics().numpy())
    est_rgbd_img = o3d.t.geometry.RGBDImage(kf_rgb_o3d, kf_depth_img)
    triangle_mesh, kf_normals = rgb_depth_to_mesh(kf_rgb, kf_depth, kf_poses[-1,:,:], self.get_intrinsics(), self.triangles)
    self.widget3d.scene.remove_geometry("est_points")
    self.widget3d.scene.add_geometry("est_points", triangle_mesh, self.pcd_mat)

    kf_normals_color = torch_to_open3d_normal_color(kf_normals)
    self.kf_normals_w.update_image(kf_normals_color.to_legacy())

    # frustum_scale = self.cfg["frustum_const"]
    self.frumstum_scale = self.cfg["frustum_const"] * self.scale
    # Keyframe window
    kf_geo = get_one_way_lineset(kf_poses.numpy(), kf_poses.numpy(), kf_pairs,
        self.get_intrinsics(), self.get_img_size(),  self.cfg["kf_color"],
        frustum_scale=self.frumstum_scale)
    self.widget3d.scene.remove_geometry("kf_frames")
    self.widget3d.scene.add_geometry("kf_frames", kf_geo, self.line_mat)
    # One-way frames
    if one_way_poses.shape[0] > 0:
      one_way_geo = get_one_way_lineset(kf_poses.numpy(), one_way_poses.numpy(), one_way_pairs,
          self.get_intrinsics(), self.get_img_size(),  self.cfg["one_way_color"],
          frustum_scale=self.frumstum_scale)
      self.widget3d.scene.remove_geometry("one_way_frames")
      self.widget3d.scene.add_geometry("one_way_frames", one_way_geo, self.line_mat)

    num_inducing_pts = kf_sparse_coords_norm.shape[1]
    self.num_inducing_pts_label.text = 'Keyframe # Inducing Points: ' + str(num_inducing_pts)

    self.kf_timestamps = kf_timestamps
    self.kf_poses = kf_poses.numpy()

  def update_pose_render(self, est_poses):

    est_traj_geo = frustum_lineset(self.get_intrinsics(), self.get_img_size(), est_poses[-1], 
      scale=self.frumstum_scale)
    est_traj_geo.paint_uniform_color(self.cfg["tracking_color"])
    self.widget3d.scene.remove_geometry("est_traj")
    self.widget3d.scene.add_geometry("est_traj", est_traj_geo, self.line_mat)

    if self.follow_val == "Estimated" and est_poses.shape[0] > 0:
      self.setup_camera_view(est_poses[-1], self.pose_init)
    elif self.follow_val == "None":
      pass

  def update_main(self):

    # Initialize processes
    self.start_slam_processes()

    # Initialize rendering
    gui.Application.instance.post_to_main_thread(
        self.window, self._on_start)
    it = iter(self.dataloader)
    timestamp, rgb = next(it)
    self.idx += 1
    reproj_depth = torch.ones_like(rgb[:,0:1,:,:])
    gui.Application.instance.post_to_main_thread(
        self.window, lambda: self.init_render(rgb))

    # Record data
    self.timestamps = []
    self.est_poses = np.array([]).reshape(0,4,4)

    # Rendering helper functions
    def update_curr_image_render_helper():
      self.update_curr_image_render(rgb)
      return

    def update_keyframe_render_helper():
      self.update_keyframe_render_viz = False
      self.update_keyframe_render(kf_timestamps, kf_rgb, kf_poses, kf_depth, kf_sparse_coords_norm, one_way_poses, kf_pairs, one_way_pairs)
      return
    
    def update_pose_render_helper():
      self.update_pose_render(self.est_poses)
      return

    # Main loop
    fps_interval_len = 30
    start_fps = time.time()
    start_data_time = time.time()
    while not self.tracking_done or not self.mapping_done:

      if not self.is_running and not self.advance_one_frame:
        time.sleep(0.01)
        continue
      self.advance_one_frame = False

      # RGB data
      if self.idx < self.dataloader.dataset.__len__():
        timestamp_next, rgb = next(it)
        
        # Real-time handling
        end_data_time = time.time()
        real_diff = end_data_time - start_data_time
        ts_diff = (timestamp_next - timestamp)
        sleep_time = max(0.0, ts_diff - real_diff)
        timestamp = timestamp_next
        # print(sleep_time)
        if not self.is_live:
          time.sleep(sleep_time)
        start_data_time = time.time()

        data = (timestamp, rgb.clone())
        self.rgb_queue.push(data) # Blocking until queue has spot
        gui.Application.instance.post_to_main_thread(
            self.window, update_curr_image_render_helper)
      elif self.idx == self.dataloader.dataset.__len__():
        # Send end signal to queues but don't exit loop yet
        self.rgb_queue.push(("end",))
      end = time.time()

      # Receive pose from tracking
      tracking_data = self.tracking_pose_queue.pop_until_latest(block=False, timeout=0.01)
      if tracking_data is not None:
        if tracking_data[0] == "end":
          self.tracking_done = True
        else:
          tracked_timestamp, tracked_pose = tracking_data

          # Record data
          # TODO: This is specifically for TUM
          self.timestamps.append(tracked_timestamp)
          self.est_poses = np.concatenate((self.est_poses, tracked_pose))
          # Visualize data
          gui.Application.instance.post_to_main_thread(
              self.window, update_pose_render_helper)
      release_data(tracking_data)

      # Receive keyframe visualization
      kf_viz_data = self.kf_viz_queue.pop_until_latest(block=False, timeout=0.01)
      if kf_viz_data is not None:
        if kf_viz_data[0] == "end":
          self.mapping_done = True
        else:
          kf_timestamps, kf_rgb, kf_poses, kf_depth, kf_sparse_coords_norm, one_way_poses, kf_pairs, one_way_pairs = kf_viz_data
          gui.Application.instance.post_to_main_thread(
              self.window, update_keyframe_render_helper)
      release_data(kf_viz_data)

      # Update text
      self.update_idx_text()
      if (self.idx % fps_interval_len == 0):
        end_fps = time.time()
        elapsed = end_fps - start_fps
        start_fps = time.time()
        self.output_fps.text = 'FPS: {:.3f}'.format(fps_interval_len / elapsed)

      self.idx += 1

    self.shutdown_slam_processes()

    # self._on_close()
    # gui.Application.instance.quit()