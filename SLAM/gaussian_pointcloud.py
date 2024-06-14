import os
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from SLAM.utils import *
from utils.general_utils import (
    build_rotation,
    build_covariance_from_scaling_rotation,
    inverse_sigmoid
)
from utils.sh_utils import RGB2SH, SH2RGB


class GaussianPointCloud(object):
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args) -> None:
        # gaussian optimize parameters
        self._xyz = devF(torch.empty(0))
        self._features_dc = devF(torch.empty(0))
        self._features_rest = devF(torch.empty(0))
        self._scaling = devF(torch.empty(0))
        self._rotation = devF(torch.empty(0))
        self._opacity = devF(torch.empty(0))
        # map management paramters
        self._normal = devF(torch.empty(0))
        self._confidence = devF(torch.empty(0))
        self._add_tick = devI(torch.empty(0))
        # error counter
        self._depth_error_counter = devI(torch.empty(0))
        self._color_error_counter = devI(torch.empty(0))

        self.init_opacity = args.init_opacity
        self.scale_factor = args.scale_factor
        self.min_radius = args.min_radius
        self.max_radius = args.max_radius
        self.max_sh_degree = args.max_sh_degree
        self.active_sh_degree = args.active_sh_degree
        assert self.active_sh_degree <= self.max_sh_degree
        self.xyz_factor = devF(torch.tensor(args.xyz_factor))
        self.setup_functions()

    def densify(self, sigma, circle_num, levels):
        means3D = self._xyz
        normal = self.get_normal
        normal = normal.cpu()
        plane0, plane1, axis0, axis1 = self.get_plane
        plane0 = plane0.cpu()
        plane1 = plane1.cpu()
        axis0 = axis0.cpu()
        axis1 = axis1.cpu()
        P = normal.shape[0]

        # generate theta
        theta = torch.rand(1, circle_num) * torch.pi * 2
        theta = theta.repeat(1, levels * sigma)

        a_random = None
        b_random = None
        a_random_ = torch.ones(P, circle_num * levels) * axis0 * sigma
        b_random_ = torch.ones(P, circle_num * levels) * axis1 * sigma
        # normal = normal.repeat([sample_num])
        for level in range(levels):
            a_random_[:, level * circle_num : (level + 1) * circle_num] *= (
                level + 0.5
            ) / levels
            b_random_[:, level * circle_num : (level + 1) * circle_num] *= (
                level + 0.5
            ) / levels

        for sigma_ in range(sigma):
            if a_random is None:
                a_random = a_random_
                b_random = b_random_
            else:
                # print(a_random.shape, a_random_.shape)
                a_random = torch.concat([a_random, a_random_ + axis0 * sigma_], dim=1)
                b_random = torch.concat([b_random, b_random_ + axis1 * sigma_], dim=1)

        x = a_random * torch.cos(theta)
        z = b_random * torch.sin(theta)

        xyz = torch.concat(
            [x[..., None], torch.zeros_like(x)[..., None], z[..., None]], dim=-1
        ).unsqueeze(-1)
        rotation = (
            torch.stack([plane0, normal, plane1], dim=-1)
            .permute(0, 2, 1)
            .unsqueeze(1)
            .repeat(1, circle_num * levels * sigma, 1, 1)
        )

        xyz = xyz.cpu()
        rotation = rotation.cpu()
        xyz = torch.matmul(rotation, xyz).squeeze(-1)
        means3D = means3D.cpu()
        xyz += means3D.unsqueeze(1).repeat(1, circle_num * levels * sigma, 1)
        normal = normal[:, None, :].repeat([1, circle_num * levels * sigma, 1])
        xyz = xyz.reshape(-1, 3)
        normal = normal.reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(normal.cpu().numpy())

        return pcd

    def load(self, ply_path):
        plydata = PlyData.read(ply_path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        P = xyz.shape[0]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        if "confidence" in plydata.elements[0]:
            confidences = np.asarray(plydata.elements[0]["confidence"])[..., np.newaxis]
        else:
            confidences = np.zeros((P, 1))

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
        )

        self._features_rest = (
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
        )
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        self._normal = self.get_normal
        self._confidence = torch.tensor(confidences, dtype=torch.float, device="cuda")

        self._add_tick = torch.zeros([P, 1], dtype=torch.int32, device="cuda")
        self._depth_error_counter = torch.zeros(
            [P, 1], dtype=torch.int32, device="cuda"
        )
        self._color_error_counter = torch.zeros(
            [P, 1], dtype=torch.int32, device="cuda"
        )

    def delete(self, delte_mask):
        self._xyz = self._xyz[~delte_mask]
        self._features_dc = self._features_dc[~delte_mask]
        self._features_rest = self._features_rest[~delte_mask]
        self._scaling = self._scaling[~delte_mask]
        self._rotation = self._rotation[~delte_mask]
        self._opacity = self._opacity[~delte_mask]
        self._normal = self._normal[~delte_mask]
        self._confidence = self._confidence[~delte_mask]
        self._add_tick = self._add_tick[~delte_mask]
        self._depth_error_counter = self._depth_error_counter[~delte_mask]
        self._color_error_counter = self._color_error_counter[~delte_mask]

    def remove(self, remove_mask):
        xyz = self._xyz[remove_mask]
        features_dc = self._features_dc[remove_mask]
        features_rest = self._features_rest[remove_mask]
        scaling = self._scaling[remove_mask]
        rotation = self._rotation[remove_mask]
        opacity = self._opacity[remove_mask]
        normal = self._normal[remove_mask]
        confidence = self._confidence[remove_mask]
        add_tick = self._add_tick[remove_mask]
        depth_error_counter = self._depth_error_counter[remove_mask]
        color_error_counter = self._color_error_counter[remove_mask]

        gaussian_params = {
            "xyz": xyz,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "normal": normal,
            "confidence": confidence,
            "add_tick": add_tick,
            "depth_error_counter": depth_error_counter,
            "color_error_counter": color_error_counter,
        }
        self.delete(remove_mask)
        return gaussian_params

    def detach(self):
        self._xyz = self._xyz.detach()
        self._features_dc = self._features_dc.detach()
        self._features_rest = self._features_rest.detach()
        self._scaling = self._scaling.detach()
        self._rotation = self._rotation.detach()
        self._opacity = self._opacity.detach()

    def parametrize(self, update_args):
        self._xyz = nn.Parameter(self._xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity.requires_grad_(True))
        l = [
            {
                "params": [self._xyz],
                "lr": update_args.position_lr,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": update_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": update_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": update_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": update_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": update_args.rotation_lr,
                "name": "rotation",
            },
        ]
        return l

    def cat(self, paramters):
        self._xyz = torch.cat([self._xyz, paramters["xyz"]])
        self._features_dc = torch.cat([self._features_dc, paramters["features_dc"]])
        self._features_rest = torch.cat(
            [self._features_rest, paramters["features_rest"]]
        )
        self._scaling = torch.cat([self._scaling, paramters["scaling"]])
        self._rotation = torch.cat([self._rotation, paramters["rotation"]], dim=0)
        self._opacity = torch.cat([self._opacity, paramters["opacity"]])
        self._confidence = torch.cat([self._confidence, paramters["confidence"]])
        self._normal = self.get_normal
        self._add_tick = torch.cat([self._add_tick, paramters["add_tick"]])
        self._depth_error_counter = torch.cat(
            [self._depth_error_counter, paramters["depth_error_counter"]]
        )
        self._color_error_counter = torch.cat(
            [self._color_error_counter, paramters["color_error_counter"]]
        )

    def add_empty_points(self, xyz, normal, color, time):
        """
        :param xyz: [N, 3]
        :param normal: [N, 3]
        :param color: [N, 3]
        """
        # preprocess
        assert xyz.shape[0] == color.shape[0] and color.shape[0] == normal.shape[0]
        if xyz.shape[0] < 1:
            return
        mag = l2_norm(normal)
        normal = normal / (mag + 1e-8)
        valid_normal_mask = normal.sum(dim=-1) != 0
        xyz = xyz[valid_normal_mask]
        normal = normal[valid_normal_mask]
        color = color[valid_normal_mask]
        points_num = xyz.shape[0]
        # compute SH feature
        features = devF(torch.zeros((points_num, 3, (self.max_sh_degree + 1) ** 2)))
        sh_color = RGB2SH(color)
        features[:, :3, 0] = sh_color
        features[:, 3:, 1:] = 0.0
        # init scale and rot
        raw_scales = devF(torch.ones(points_num, 3)) * 1e-6
        scales = torch.log(raw_scales)
        if (
            self.xyz_factor[0] == 1
            and self.xyz_factor[1] == 1
            and self.xyz_factor[2] == 1
        ):
            rots = devF(torch.zeros((points_num, 4)))
            rots[:, 0] = 1
        else:
            z_axis = devF(torch.tensor([0, 0, 1]).repeat(points_num, 1))
            rots = compute_rot(z_axis, normal)
        # init opacity
        opacities = inverse_sigmoid(
            self.init_opacity * devF(torch.ones((points_num, 1)))
        )
        # init other flags
        confidence = devF(torch.zeros([points_num, 1]))
        add_tick = time * devI(torch.ones([points_num, 1]))

        depth_error_counter = devI(torch.zeros([points_num, 1]))
        color_error_counter = devI(torch.zeros([points_num, 1]))

        add_params = {
            "xyz": xyz,
            "features_dc": features[..., 0:1].transpose(1, 2).contiguous(),
            "features_rest": features[..., 1:].transpose(1, 2).contiguous(),
            "scaling": scales,
            "rotation": rots,
            "opacity": opacities,
            "normal": normal,
            "confidence": confidence,
            "add_tick": add_tick,
            "depth_error_counter": depth_error_counter,
            "color_error_counter": color_error_counter,
        }
        self.cat(add_params)

    def update_geometry(self, extra_xyz, extra_radius):
        xyz = self.get_xyz
        radius = self.get_radius
        points_num = self.get_points_num
        if torch.numel(extra_xyz) > 0:
            inbbox_mask = bbox_filter(xyz, extra_xyz)
            extra_xyz = extra_xyz[inbbox_mask]
            extra_radius = extra_radius[inbbox_mask]
        total_xyz = torch.cat([xyz, extra_xyz])
        total_radius = torch.cat([radius, extra_radius])
        _, knn_indices = distCUDA2(total_xyz.float().cuda())
        knn_indices = knn_indices[:points_num].long()
        dist_0 = (
            torch.norm(xyz - total_xyz[knn_indices[:, 0]], p=2, dim=1)
            - 3 * total_radius[knn_indices[:, 0]]
        )
        dist_1 = (
            torch.norm(xyz - total_xyz[knn_indices[:, 1]], p=2, dim=1)
            - 3 * total_radius[knn_indices[:, 1]]
        )
        dist_2 = (
            torch.norm(xyz - total_xyz[knn_indices[:, 2]], p=2, dim=1)
            - 3 * total_radius[knn_indices[:, 2]]
        )
        invalid_dist_0 = dist_0 < 0
        invalid_dist_1 = dist_1 < 0
        invalid_dist_2 = dist_2 < 0

        invalid_scale_mask = invalid_dist_0 | invalid_dist_1 | invalid_dist_2
        dist2 = (dist_0**2 + dist_1**2 + dist_2**2) / 3
        scales = torch.sqrt(dist2)
        scales = torch.clip(scales, min=self.min_radius, max=self.max_radius)
        if (~invalid_scale_mask).sum() == 0:
            self.delete(invalid_scale_mask)
        else:
            scales = scales[..., None].repeat(1, 3)
            factor_scales = self.scale_factor * torch.mul(scales, self.xyz_factor)
            log_scales = torch.log(factor_scales)
            self._scaling = log_scales
            self.delete(invalid_scale_mask)

    def construct_list_of_attributes(self, include_confidence=True):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        if include_confidence:
            l.append("confidence")
        return l

    def save_model_ply(self, path, include_confidence=True):
        if self.get_points_num == 0:
            return
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        confidence = self._confidence.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4")
            for attribute in self.construct_list_of_attributes(include_confidence)
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if include_confidence:
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation, confidence),
                axis=1,
            )
        else:
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_color_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        # save color ply
        elements = np.empty(
            xyz.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        color = SH2RGB(f_dc.reshape(-1, 3)) * 255
        attributes = np.concatenate((xyz, color), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        file_name = os.path.basename(path)
        file_base = os.path.dirname(path)
        color_name = file_name.split(".")
        color_name = color_name[0] + "_color" + "." + color_name[1]
        color_path = os.path.join(file_base, color_name)
        PlyData([el]).write(color_path)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_points_num(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_radius(self):
        scales = self.get_scaling
        min_length, _ = torch.min(scales, dim=1)
        radius = (torch.sum(scales, dim=1) - min_length) / 2
        return radius

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_R(self):
        return build_rotation(self.rotation_activation(self._rotation))

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation
        )

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        scales = self.get_scaling
        R = self.get_R
        min_indices = torch.argmin(scales, dim=1)
        normal = torch.gather(
            R.transpose(1, 2),
            1,
            min_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )
        normal = normal[:, 0, :]
        mag = l2_norm(normal)
        return normal / (mag + 1e-8)

    @property
    def get_plane(self):
        scales = self.get_scaling
        R = self.get_R
        plane_indices = scales.argsort(dim=1)[:, 1:]
        plane0 = torch.gather(
            R.transpose(1, 2),
            1,
            plane_indices[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )[:, 0, :]
        plane1 = torch.gather(
            R.transpose(1, 2),
            1,
            plane_indices[:, 1].unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )[:, 0, :]
        plane0 = plane0 / (l2_norm(plane0) + 1e-8)
        plane1 = plane1 / (l2_norm(plane1) + 1e-8)
        axis0 = torch.gather(scales, 1, plane_indices[:, 0:1])
        axis1 = torch.gather(scales, 1, plane_indices[:, 1:])
        return plane0, plane1, axis0, axis1

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_color(self):
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()
        color = SH2RGB(f_dc.reshape(-1, 3))
        return color

    @property
    def get_confidence(self):
        return self._confidence

    @property
    def get_add_tick(self):
        return self._add_tick
