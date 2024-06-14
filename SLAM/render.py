import numpy as np
import math
import torch

from scene.cameras import Camera
from SLAM.utils import devF, devI

from diff_gaussian_rasterization_depth import (
    GaussianRasterizationSettings as GaussianRasterizationSettings_depth,
)
from diff_gaussian_rasterization_depth import (
    GaussianRasterizer as GaussianRasterizer_depth,
)

from utils.general_utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid
)


class Renderer:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.raster_settings = None
        self.rasterizer = None
        self.bg_color = devF(torch.tensor([0, 0, 0]))
        self.renderer_opaque_threshold = args.renderer_opaque_threshold
        self.renderer_normal_threshold = np.cos(
            np.deg2rad(args.renderer_normal_threshold)
        )
        self.scaling_modifier = 1.0
        self.renderer_depth_threshold = args.renderer_depth_threshold
        self.max_sh_degree = args.max_sh_degree
        self.color_sigma = args.color_sigma
        if args.active_sh_degree < 0:
            self.active_sh_degree = self.max_sh_degree
        else:
            self.active_sh_degree = args.active_sh_degree
        self.setup_functions()

    def get_scaling(self, scaling):
        return self.scaling_activation(scaling)

    def get_rotation(self, rotaion):
        return self.rotation_activation(rotaion)

    def get_covariance(self, scaling, rotaion, scaling_modifier=1):
        return self.covariance_activation(scaling, scaling_modifier, rotaion)

    def render(
        self,
        viewpoint_camera: Camera,
        gaussian_data,
        tile_mask=None,
    ):
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        self.raster_settings = GaussianRasterizationSettings_depth(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            opaque_threshold=self.renderer_opaque_threshold,
            depth_threshold=self.renderer_depth_threshold,
            normal_threshold=self.renderer_normal_threshold,
            color_sigma=self.color_sigma,
            prefiltered=False,
            debug=False,
            cx=viewpoint_camera.cx,
            cy=viewpoint_camera.cy,
            T_threshold=0.0001,
        )
        self.rasterizer = GaussianRasterizer_depth(
            raster_settings=self.raster_settings
        )

        means3D = gaussian_data["xyz"]
        opacity = gaussian_data["opacity"]
        scales = gaussian_data["scales"]
        rotations = gaussian_data["rotations"]
        shs = gaussian_data["shs"]
        normal = gaussian_data["normal"]
        cov3D_precomp = None
        colors_precomp = None
        if tile_mask is None:
            tile_mask = devI(
                torch.ones(
                    (viewpoint_camera.image_height + 15) // 16,
                    (viewpoint_camera.image_width + 15) // 16,
                    dtype=torch.int32,
                )
            )
        
        render_results = self.rasterizer(
            means3D=means3D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            normal_w=normal,
            tile_mask=tile_mask,
        )

        rendered_image = render_results[0]
        rendered_depth = render_results[1]
        color_index_map = render_results[2]
        depth_index_map = render_results[3]
        color_hit_weight = render_results[4]
        depth_hit_weight = render_results[5]
        T_map = render_results[6]

        render_normal = devF(torch.zeros_like(rendered_image))
        render_normal[:, depth_index_map[0] > -1] = normal[
            depth_index_map[depth_index_map > -1].long()
        ].permute(1, 0)
        
        results = {
            "render": rendered_image,
            "depth": rendered_depth,
            "normal": render_normal,
            "color_index_map": color_index_map,
            "depth_index_map": depth_index_map,
            "color_hit_weight": color_hit_weight,
            "depth_hit_weight": depth_hit_weight,
            "T_map": T_map,
        }
        return results
