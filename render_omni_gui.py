# Copyright (C) 2024, OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning
# https://github.com/OceanYing/OmniSeg3D-GS
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting, and Gaussian-Grouping
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA

from scene.gaussian_model import GaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy.spatial.transform import Rotation as R

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 2   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    # ckpt TODO: load from gui window.

    ply_path = "output/360_counter_omni_1/sem_hi/point_cloud/iteration_15000/point_cloud.ply"


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = gaussian_model

        self.proj_mat = None

        self.load_model = False
        print("loading model file...")
        self.engine.load_ply(self.opt.ply_path)
        self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False                 # clear all the click prompts
        self.binary_threshold_button = False    # binary segmentation mode
        self.segment3d_flag = False
        self.reload_flag = False        # reload the whole scene / point cloud
        self.object_seg_id = 0          # to store the segmented object with increasing index order (path at: ./)

    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)

    def register_dpg(self):
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        def callback_depth(sender, app_data):
            self.img_mode = (self.img_mode + 1) % 3
            
        # --- interactive mode switch --- #
        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button
        def clickmode_multi_callback(sender):
            self.clickmode_multi_button = dpg.get_value(sender)
            print("clickmode_multi_button = ", self.clickmode_multi_button)
        def binary_threshold_callback(sender):
            self.binary_threshold_button = dpg.get_value(sender)
            print("binary_threshold_button = ", self.binary_threshold_button)
        def clear_edit():
            self.clear_edit = True
        @torch.no_grad()
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_reload():
            self.reload_flag = True

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=300, pos=[self.window_width+10, 0]):
            
            dpg.add_slider_float(label="ScoreThres", default_value=0.0,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")
            dpg.add_button(label="render_option", tag="_button_depth",
                            callback=callback_depth)
            dpg.add_text("Mouse position: click anywhere to start. ", tag="pos_item")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data")
            dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_checkbox(label="binary_threshold", callback=binary_threshold_callback, user_data="Some Data")
            dpg.add_button(label="clear_edit", callback=clear_edit, user_data="Some Data")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            dpg.add_button(label="reload_data", callback=callback_reload, user_data="Some Data")

            def callback(sender, app_data, user_data):
                self.load_model = False
                file_data = app_data["selections"]
                file_names = []
                for key in file_data.keys():
                    file_names.append(key)

                self.opt.ply_file = file_data[file_names[0]]

                # if not self.load_model:
                print("loading model file...")
                self.engine.load_ply(self.opt.ply_file)
                self.do_pca()   # calculate new self.proj_mat after loading new .ply file
                print("loading model file done.")
                self.load_model = True

            with dpg.file_dialog(directory_selector=False, show=False, callback=callback, id="file_dialog_id", width=700, height=400,
            ):
                dpg.add_file_extension(".*")
                dpg.add_file_extension("", color=(150, 255, 150, 255))
                dpg.add_file_extension("Ply (*.ply){.ply}", color=(0, 255, 255, 255))
            dpg.add_button(label="File Selector", callback=lambda: dpg.show_item("file_dialog_id"))


        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        

        def toggle_moving_left():
            self.moving = not self.moving


        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle


        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            
            self.mouse_pos = pos


        def change_pos(sender, app_data):
            # if not dpg.is_item_focused("_primary_window"):
            #     return
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.clickmode_button and app_data == 1:     # in the click mode and right click
                print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(callback=change_pos)
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        return cam
    

    def pca(self, X, n_components=3):
        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16, 
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]
        
        return proj_mat
    

    def do_pca(self):
        sems = self.engine._objects_dc.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.proj_mat = self.pca(sem_chosen, n_components=3)
        print("project mat initialized !")


    @torch.no_grad()
    def fetch_data(self, view_camera):
        
        outputs = render(view_camera, self.engine, self.opt, self.bg_color)

        # --- RGB image --- #
        img = outputs["render"].permute(1, 2, 0)  #
        rgb_score = img.clone()
        depth_score = rgb_score.cpu().numpy().reshape(-1)

        # --- semantic image --- #
        sems = outputs["render_object"].permute(1, 2, 0)
        H, W, C = sems.shape
        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        sem_transed = sems @ self.proj_mat
        sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1)

        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
        
        if self.reload_flag:
            self.reload_flag = False
            self.engine.load_ply(self.opt.ply_path)
            print("Reload original 3D Gaussians from: ", self.opt.ply_path)


        if len(self.new_click_xy) > 0:

            featmap = sems.reshape(H, W, -1)
            
            if self.new_click:
                xy = self.new_click_xy
                new_feat = featmap[int(xy[1])%H, int(xy[0])%W, :].reshape(featmap.shape[-1], -1)
                if (self.prompt_num == 0) or (self.clickmode_multi_button == False):
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)    # extend to get more prompt features
                self.prompt_num += 1
                self.new_click = False
            
            score_map = featmap @ self.chosen_feature

            score_map = (score_map + 1.0) / 2
            score_binary = score_map > dpg.get_value('_ScoreThres')
            
            score_map[~score_binary] = 0.0
            score_map = torch.max(score_map, dim=-1).values
            score_norm = (score_map - dpg.get_value('_ScoreThres')) / (1 - dpg.get_value('_ScoreThres'))

            if self.binary_threshold_button:
                rgb_score = img * torch.max(score_binary, dim=-1, keepdim=True).values    # option: binary
            else:
                rgb_score = img * score_norm[:, :, None]
            depth_score = 1 - torch.clip(score_norm, 0, 1)
            depth_score = depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0

            if self.segment3d_flag:
                """ gaussian point cloud core params
                self.engine._xyz            # (N, 3)
                self.engine._features_dc    # (N, 1, 3)
                self.engine._features_rest  # (N, 15, 3)
                self.engine._opacity        # (N, 1)
                self.engine._scaling        # (N, 3)
                self.engine._rotation       # (N, 4)
                self.engine._objects_dc     # (N, 1, 16)
                """
                self.segment3d_flag = False
                feat_pts = self.engine._objects_dc.squeeze()
                feat_pts /= (torch.norm(feat_pts, dim=-1, keepdim=True) + 1e-6)
                score_pts = feat_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2
                score_pts_binary = (score_pts > dpg.get_value('_ScoreThres')).sum(1) > 0
                save_path = "./debug_robot_{:0>3d}.ply".format(self.object_seg_id)
                self.engine.save_ply_partial(save_path, idx=score_pts_binary)
                self.engine.load_ply(save_path)
                self.object_seg_id += 1
                print("Segmented 3D Gaussians saved to: ", save_path)

        if self.img_mode == 0:
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
        elif self.img_mode == 1:
            self.render_buffer = depth_score
        elif self.img_mode == 2:
            self.render_buffer = sem_transed_rgb.cpu().numpy()
        
        dpg.set_value("_texture", self.render_buffer)


if __name__ == "__main__":

    opt = CONFIG()
    gs_model = GaussianModel(opt.sh_degree)
    gui = GaussianSplattingGUI(opt, gs_model)

    gui.render()