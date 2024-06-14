#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from torch import nn
from SLAM.utils import downscale_img

from utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2
from utils.general_utils import devF


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        depth,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        pose_gt=np.eye(4),
        cx=-1,
        cy=-1,
        timestamp=0,
        depth_scale=1.0,
        preload=True,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.preload = preload
        self.timestamp = timestamp
        self.depth_scale = depth_scale
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        if not self.preload:
            self.data_device = torch.device("cpu")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if depth is not None:
            self.original_depth = depth.to(self.data_device)
        else:
            self.original_depth = torch.ones(1, self.image_height, self.image_width).to(
                self.data_device
            )

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_depth *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )
            self.original_depth *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # for evaluation, unchange
        self.pose_gt = pose_gt
        self.cx = cx
        self.cy = cy

        self.world_view_transform.share_memory_()
        self.full_proj_transform.share_memory_()

    def updatePose(self, pose_c2w):
        pose_w2c = np.linalg.inv(pose_c2w)
        self.update(pose_w2c[:3, :3].transpose(), pose_w2c[:3, 3])

    def update(self, R, T):
        self.R = R
        self.T = T
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, self.trans, self.scale))
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    def get_w2c(self):
        return self.world_view_transform.transpose(0, 1)

    @property
    def get_c2w(self):
        return self.world_view_transform.transpose(0, 1).inverse()

    # TODO: only work for Repulica dataset, need to add load local depth intrinsic for ScanNet
    @property
    def get_intrinsic(self):
        w, h = self.image_width, self.image_height
        fx, fy = fov2focal(self.FoVx, w), fov2focal(self.FoVy, h)
        cx = self.cx if self.cx > 0 else w / 2
        cy = self.cy if self.cy > 0 else h / 2
        intrinstic = devF(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
        return intrinstic

    def get_focal_length(self):
        w, h = self.image_width, self.image_height
        fx, fy = fov2focal(self.FoVx, w), fov2focal(self.FoVy, h)
        return (fx + fy) / 2.0

    def get_uv(self, xyz_w):
        intrinsic = self.get_intrinsic
        w2c = self.get_w2c()
        xyz_c = xyz_w @ w2c[:3, :3].T + w2c[:3, 3]
        uv = xyz_c @ intrinsic.T
        uv = uv[:, :2] / uv[:, 2:]
        uv = uv.long()
        return uv

    def move_to_cpu_clone(self):
        new_cam = Camera(
            colmap_id=self.colmap_id,
            R=self.R,
            T=self.T,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            image=self.original_image.detach(),
            depth=self.original_depth.detach(),
            gt_alpha_mask=None,
            image_name=self.image_name,
            uid=self.uid,
            data_device=self.data_device,
            pose_gt=self.pose_gt,
            cx=self.cx,
            cy=self.cy,
            timestamp=self.timestamp,
            preload=self.preload,
            depth_scale=self.depth_scale,
        )
        new_cam.original_depth = new_cam.original_depth.to("cpu")
        new_cam.original_image = new_cam.original_image.to("cpu")
        return new_cam


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.cx = -1
        self.cy = -1
