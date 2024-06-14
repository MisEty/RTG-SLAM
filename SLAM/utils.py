import math
import os
import time

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from skimage import filters
from skimage.color import rgb2gray
import numpy
import yaml
from utils.general_utils import devB, devF, devI, quaternion_from_axis_angle


def homogeneous(points: torch.tensor):
    temp = points[..., :1]
    ones = devF(torch.ones_like(temp))
    return torch.cat([points, ones], dim=-1)


def l2_norm(points: torch.tensor):
    return torch.norm(points, p=2, dim=-1, keepdim=True)


def get_rot(transform: torch.tensor):
    res = devF(torch.eye(4))
    res[:3, :3] = transform[:3, :3]
    return res


def get_trans(transform: torch.tensor):
    res = devF(torch.eye(4))
    res[:3, 3] = transform[:3, 3]
    return res


def rot_compare(prev_rot: np.array, curr_rot: np.array):
    rot_diff = prev_rot.T @ curr_rot
    cos_theta = (np.trace(rot_diff) - 1) / 2
    rad_diff = np.arccos(cos_theta)
    theta_diff = np.rad2deg(rad_diff)
    return rad_diff, theta_diff


def trans_compare(prev_trans: np.array, curr_trans: np.array):
    trans_diff = prev_trans - curr_trans
    l1_diff = np.linalg.norm(trans_diff, ord=1)
    l2_diff = np.linalg.norm(trans_diff, ord=2)
    return l1_diff, l2_diff

def transform_map(map: torch.tensor, transform: torch.tensor):
    assert map.shape[-1] == 1 or map.shape[-1] == 3
    H, W, C = map.shape[0], map.shape[1], map.shape[2]
    transform_expand = transform.unsqueeze(0).unsqueeze(0).expand(H, W, -1, -1)
    new_map = torch.matmul(transform_expand, homogeneous(map).unsqueeze(-1)).squeeze()[
        ..., :C
    ]
    return new_map

def compute_vertex_map(depth, K):
    H, W = depth.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    device = depth.device

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)  # [h, w]
    j = j.t().to(device)  # [h, w]

    vertex = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1).to(device) * depth  # [h, w, 3]
    return vertex

def feature_gradient(img, normalize_gradient=True):
    """ Calculate the gradient on the feature space using Sobel operator
    :param the input image
    -----------
    :return the gradient of the image in x, y direction
    """
    H, W, C = img.shape
    # to filter the image equally in each channel
    wx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).type_as(img)
    wy = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).type_as(img)

    img_permuted = img.permute(2, 0, 1).view(-1, 1, H, W)  # [c, 1, h, w]
    img_pad = F.pad(img_permuted, (1, 1, 1, 1), mode='replicate')
    img_dx = F.conv2d(img_pad, wx, stride=1, padding=0).squeeze().permute(1, 2, 0)  # [h, w, c]
    img_dy = F.conv2d(img_pad, wy, stride=1, padding=0).squeeze().permute(1, 2, 0)  # [h, w, c]

    if normalize_gradient:
        mag = torch.sqrt((img_dx ** 2) + (img_dy ** 2) + 1e-8)
        img_dx = img_dx / mag
        img_dy = img_dy / mag

    return img_dx, img_dy  # [h, w, c]

def compute_normal_map(vertex_map):
    """ Calculate the normal map from a depth map
    :param the input depth image
    -----------
    :return the normal map
    """
    H, W, C = vertex_map.shape
    img_dx, img_dy = feature_gradient(vertex_map, normalize_gradient=False)  # [h, w, 3]

    normal = torch.cross(img_dy.view(-1, 3), img_dx.view(-1, 3))
    normal = normal.view(H, W, 3)  # [h, w, 3]

    mag = torch.norm(normal, p=2, dim=-1, keepdim=True)
    normal = normal / (mag + 1e-8)

    # filter out invalid pixels
    depth = vertex_map[:, :, -1]
    # 0.5 and 5.
    invalid_mask = (depth <= depth.min()) | (depth >= depth.max())
    zero_normal = torch.zeros_like(normal)
    normal = torch.where(invalid_mask[..., None], zero_normal, normal)

    return normal


def compute_confidence_map(normal_map, intrinsic):
    H, W, C = normal_map.shape
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    h_grid, w_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    proj_map = torch.ones(H, W, 3).cuda()
    proj_map[..., 0] = (w_grid.cuda() - cx) / fx
    proj_map[..., 1] = (h_grid.cuda() - cy) / fy
    mag = l2_norm(proj_map)
    proj_map = proj_map / (mag + 1e-8)
    view_normal_dist = torch.abs(F.cosine_similarity(normal_map, proj_map, dim=-1))
    return view_normal_dist[..., None]


def sample_pixels(
    vertex_map,
    normal_map,
    color_map,
    uniform_sample_num,
    select_mask=None,
):
    assert uniform_sample_num >= 0
    if uniform_sample_num == 0:
        return devF(torch.empty(0)), devF(torch.empty(0)), devF(torch.empty(0))

    # 1. compute local data

    H, W = vertex_map.shape[0], vertex_map.shape[1]
    coord_y, coord_x = torch.meshgrid(
        torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij"
    )
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()

    if select_mask is None:
        select_mask = devB(torch.ones([H, W, 1]))
    invalid_normal_mask = torch.where(normal_map.sum(dim=-1) == 0)
    select_mask[invalid_normal_mask] = False
    if uniform_sample_num > select_mask.sum():
        uniform_sample_num = select_mask.sum()

    select_mask = select_mask.flatten()
    vertexs = vertex_map.view(-1, 3)[select_mask]
    colors = color_map.view(-1, 3)[select_mask]
    normals = normal_map.view(-1, 3)[select_mask]

    samples = torch.randperm(vertexs.shape[0])[:uniform_sample_num]

    points_uniform = vertexs[samples]
    colors_uniform = colors[samples]
    normals_uniform = normals[samples]

    return (
        points_uniform.view(uniform_sample_num, 3),
        normals_uniform.view(uniform_sample_num, 3),
        colors_uniform.view(uniform_sample_num, 3),
    )


def scale_depth(depth, min_depth=0, max_depth=0):
    if min_depth == 0 and max_depth == 0:
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    else:
        depth = (depth - min_depth) / (max_depth - min_depth)
    return depth


def color_value(
    value, invalid_mask=None, min_depth=0, max_depth=0, color_map=cv2.COLORMAP_JET
):
    if value.shape[0] > 3:
        value = value.permute(2, 0, 1)
    scaled_depth = scale_depth(value, min_depth, max_depth)
    colormap = color_map
    scaled_depth_np = scaled_depth.permute(1, 2, 0).detach().cpu().numpy()

    colored_depth = cv2.applyColorMap(np.uint8(scaled_depth_np * 255), colormap)
    colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
    res = (devF(torch.tensor(colored_depth)) / 255.0).permute(2, 0, 1)
    if invalid_mask is not None:
        invalid_mask = invalid_mask.repeat(3, 1, 1)
        res[invalid_mask] = 0
    return res


def scale_normal(normal):
    return (normal + 1) / 2


def compute_rot(init_vec, target_vec):
    axis = torch.cross(init_vec, target_vec)
    axis = axis / (torch.norm(axis, p=2, dim=-1, keepdim=True) + 1e-8)
    angle = torch.acos(torch.sum(init_vec * target_vec, dim=1)).unsqueeze(-1)
    rots = quaternion_from_axis_angle(axis, angle)
    return rots


def prepare_cfg(
    args
):
    data_path = args.source_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(args, f)

    if args.parent != "None":
        os.system(
            "cp {} {}".format(args.parent, os.path.join(args.save_path, "base.yaml"))
        )
    print(data_path)
    os.system(
        "cp {} {}".format(
            os.path.join(data_path, "cameras.json"),
            os.path.join(save_path, "cameras.json"),
        )
    )
    os.system(
        "cp {} {}".format(
            os.path.join(data_path, "points3d_depth.ply"),
            os.path.join(save_path, "input.ply"),
        )
    )
    os.makedirs(os.path.join(save_path, "point_cloud"), exist_ok=True)
    with open(os.path.join(save_path, "cfg_args"), "w") as f:
        cfg_args = "Namespace(data_device='cuda', eval=False, images='images', model_path='{}', resolution=-1, sh_degree={}, source_path='{}', white_background=False)".format(
            "", args.active_sh_degree, ""
        )
        f.write(cfg_args)


def construct_list_of_attributes(
    _features_dc, _features_rest, _scaling, _rotation, include_confidence=True
):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1] * _features_dc.shape[2]):
        l.append("f_dc_{}".format(i))
    for i in range(_features_rest.shape[1] * _features_rest.shape[2]):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(_scaling.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(_rotation.shape[1]):
        l.append("rot_{}".format(i))
    if include_confidence:
        l.append("confidence")
    return l


def read_ply(ply_path, include_confidence=False):
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
    points_select = np.arange(P)
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
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
    # sh_dgree = int(np.sqrt((len(extra_f_names) + 3) // 3)) - 1
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))
    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
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
    if include_confidence:
        return xyz, features_dc, features_extra, opacities, scales, rots, confidences
    else:
        return xyz, features_dc, features_extra, opacities, scales, rots


def save_ply(
    save_path,
    xyz,
    features_dc,
    features_extra,
    opacities,
    scales,
    rots,
    confidences=None,
):
    if confidences is None:
        dtype_full = [
            (attribute, "f4")
            for attribute in construct_list_of_attributes(
                features_dc, features_extra, scales, rots, False
            )
        ]
    else:
        dtype_full = [
            (attribute, "f4")
            for attribute in construct_list_of_attributes(
                features_dc, features_extra, scales, rots, True
            )
        ]
    normals = np.zeros_like(xyz)
    P = xyz.shape[0]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    features_dc = features_dc.reshape(P, -1)
    features_extra = features_extra.reshape(P, -1)
    if confidences is not None:
        attributes = np.concatenate(
            (
                xyz,
                normals,
                features_dc,
                features_extra,
                opacities,
                scales,
                rots,
                confidences,
            ),
            axis=1,
        )
    else:
        attributes = np.concatenate(
            (xyz, normals, features_dc, features_extra, opacities, scales, rots), axis=1
        )

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(save_path)


def merge_ply(ply0_path, ply1_path, save_path, include_confidence):
    ply0_data = read_ply(ply0_path, include_confidence)
    ply1_data = read_ply(ply1_path, include_confidence)
    merge_data = []
    merge_data.append(save_path)
    for i, data in enumerate(ply0_data):
        merge_data.append(np.concatenate([ply0_data[i], ply1_data[i]], axis=0))
    if not include_confidence:
        merge_data.append(None)
    save_ply(*merge_data)


def downscale_img(img, scale_factor):
    assert img.shape[2] == 1 or img.shape[2] == 3
    img_downscale = (
        F.interpolate(
            img.permute(2, 0, 1).unsqueeze(0),
            scale_factor=[scale_factor, scale_factor],
            mode="nearest",
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    return img_downscale


def point2plane_loss(p_t0, p_t1, n_t0, reduce="mean"):
    loss = ((p_t1 - p_t0) * n_t0).sum(dim=-1)
    if reduce == "mean":
        loss = (loss * loss).mean()
    else:
        loss = (loss * loss).sum()
    return loss


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(
        numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)
    ).A[0]

    return rot, trans, trans_error


def eval_ate(pose_estimate, pose_gt):
    
    def align(model, data):
        """Align two trajectories using the method of Horn (closed-form).

        Args:
            model -- first trajectory (3xn)
            data -- second trajectory (3xn)

        Returns:
            rot -- rotation matrix (3x3)
            trans -- translation vector (3x1)
            trans_error -- translational error per point (1xn)

        """
        numpy.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1)
        data_zerocentered = data - data.mean(1)

        W = numpy.zeros((3, 3))
        for column in range(model.shape[1]):
            W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
        S = numpy.matrix(numpy.identity(3))
        if numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0:
            S[2, 2] = -1
        rot = U * S * Vh
        trans = data.mean(1) - rot * model.mean(1)

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = numpy.sqrt(
            numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)
        ).A[0]

        return rot, trans, trans_error
    if isinstance(pose_estimate, torch.Tensor):
        pose_estimate = pose_estimate.cpu().numpy()
    if isinstance(pose_gt, torch.Tensor):
        pose_gt = pose_gt.cpu().numpy()

    pose_estimate = np.matrix(pose_estimate.transpose())
    pose_gt = np.matrix(pose_gt.transpose())
    _, _, trans_error = align(pose_estimate, pose_gt)
    ate_rmse = numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)) * 100
    return ate_rmse


def build_image_pyramid(image, downscales):
    pyramid = []
    for downscale in downscales:
        pyramid.append(downscale_img(image, downscale))
    return pyramid


def build_vertex_pyramid(depth, pyramid_builder, K):
    H, W = depth.shape[:2]
    depth_pyramid = pyramid_builder(depth.view(1,1,H,W))
    vertex_pyramid = []
    for i in range(len(depth_pyramid)):
        H_sub, W_sub = depth_pyramid[i].shape[2:4]
        downscale = 1 / 2**(len(depth_pyramid)-i-1)
        K_downscale = K * downscale
        K_downscale[2,2] = 1.0
        vertex_pyramid.append(compute_vertex_map(depth_pyramid[i].reshape(H_sub, W_sub, 1), K_downscale))
    return vertex_pyramid

def build_normal_pyramid(vertex_pyramid):
    noraml_pyramid = []
    for i in range(len(vertex_pyramid)):
        noraml_pyramid.append(compute_normal_map(vertex_pyramid[i]))
    return noraml_pyramid

def move_to_gpu(frame):
    frame.original_depth = devF(frame.original_depth)
    frame.original_image = devF(frame.original_image)


def move_to_cpu(frame):
    frame.original_depth = frame.original_depth.to("cpu")
    frame.original_image = frame.original_image.to("cpu")


def move_to_gpu_map(keymap):
    keymap["color_map"] = devF(keymap["color_map"])
    keymap["depth_map"] = devF(keymap["depth_map"])
    keymap["normal_map"] = devF(keymap["normal_map"])


def move_to_cpu_map(keymap):
    keymap["color_map"] = keymap["color_map"].detach().cpu()
    keymap["depth_map"] = keymap["depth_map"].detach().cpu()
    keymap["normal_map"] = keymap["normal_map"].detach().cpu()

def bilateralFilter_torch(
    depth,
    radius,
    sigma_color,
    sigma_space,
):
    h, w = depth.shape[:2]
    if len(depth.shape) == 3:
        depth = depth.reshape(h, w)
    depth_pad = torch.nn.functional.pad(depth, (radius, radius, radius, radius))
    weight_sum = torch.zeros_like(depth, dtype=torch.float32)
    pixel_sum = torch.zeros_like(depth, dtype=torch.float32)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if (i**2 + j**2) > radius**2:
                continue
            spatial_weight = -(i**2 + j**2) / (2 * (sigma_space**2))
            color_weight = -(
                (
                    depth
                    - depth_pad[
                        radius + i : radius + i + h, radius + j : radius + j + w
                    ]
                )
                ** 2
            ) / (2 * (sigma_color**2))
            weight = torch.exp(spatial_weight + color_weight)
            weight_mask = (
                depth_pad[radius + i : radius + i + h, radius + j : radius + j + w] != 0
            )
            weight = weight * weight_mask
            weight_sum += weight
            pixel_sum += (
                weight
                * depth_pad[radius + i : radius + i + h, radius + j : radius + j + w]
            )
    # weight_sum[weight_sum == 0] = 1
    depth_smooth = pixel_sum / weight_sum
    depth_smooth[weight_sum == 0] = 0
    return depth_smooth.reshape(h, w, 1)



def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation
    Args:
      v0: Starting vector
      v1: Final vector
      t: Float value between 0.0 and 1.0
      DOT_THRESHOLD: Threshold for considering the two vectors as
                              colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    """
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm = torch.norm(v0, dim=-1)
    v1_norm = torch.norm(v1, dim=-1)

    v0_normed = v0 / v0_norm.unsqueeze(-1)
    v1_normed = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot = (v0_normed * v1_normed).sum(-1)
    dot_mag = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp = ~gotta_lerp

    t_batch_dim_count: int = (
        max(0, t.dim() - v0.dim()) if isinstance(t, torch.Tensor) else 0
    )
    t_batch_dims = (
        t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
    )
    out = torch.zeros_like(v0.expand(*t_batch_dims, *[-1] * v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped = torch.lerp(v0, v1, t)

        out = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():
        # Calculate initial angle between v0 and v1
        theta_0 = dot.arccos().unsqueeze(-1)
        sin_theta_0 = theta_0.sin()
        # Angle at timestep t
        theta_t = theta_0 * t
        sin_theta_t = theta_t.sin()
        # Finish the slerp algorithm
        s0 = (theta_0 - theta_t).sin() / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        slerped = s0 * v0 + s1 * v1

        out = slerped.where(can_slerp.unsqueeze(-1), out)

    return out


def maxpool(matrix, stride, padding_value=0):
    height, width = matrix.shape[:2]
    pad_h = (height + stride - 1) // stride * stride - height
    pad_w = (width + stride - 1) // stride * stride - width
    matrix_pad = F.pad(matrix, (0, pad_w, 0, pad_h), value=padding_value)
    matrix_maxpool = F.max_pool2d(
        matrix_pad.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=stride,
        stride=stride,
    )
    return matrix_maxpool.squeeze(0).squeeze(0)


def meanpool(matrix, stride, padding_value=0):
    height, width = matrix.shape[:2]
    pad_h = (height + stride - 1) // stride * stride - height
    pad_w = (width + stride - 1) // stride * stride - width
    matrix_pad = F.pad(matrix, (0, pad_w, 0, pad_h), value=padding_value)
    matrix_avgpool = F.avg_pool2d(
        matrix_pad.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=stride,
        stride=stride,
    )
    return matrix_avgpool.squeeze(0).squeeze(0)


def pixelmask2tilemask(pixelmask, stride):
    height, width = pixelmask.shape[:2]
    pad_h = (height + stride - 1) // stride * stride - height
    pad_w = (width + stride - 1) // stride * stride - width
    pixelmask_pad = F.pad(pixelmask, (0, pad_w, 0, pad_h))
    # return pixelmask_pad
    tilemask = F.max_pool2d(
        pixelmask_pad.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=stride,
        stride=stride,
    )
    return tilemask.squeeze(0).squeeze(0).int()


def transmission2tilemask(pixelmask, stride, tile_mask_ratio=0.5):
    height, width = pixelmask.shape[:2]
    pad_h = (height + stride - 1) // stride * stride - height
    pad_w = (width + stride - 1) // stride * stride - width
    pixelmask_pad = F.pad(pixelmask, (0, pad_w, 0, pad_h))
    tilemask = F.avg_pool2d(
        pixelmask_pad.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=stride,
        stride=stride,
    )
    return (tilemask > tile_mask_ratio).int().squeeze(0).squeeze(0)


def colorerror2tilemask(color_error, stride, top_ratio=0.4):
    height, width = color_error.shape[:2]
    pad_h = (height + stride - 1) // stride * stride - height
    pad_w = (width + stride - 1) // stride * stride - width
    color_error_pad = F.pad(color_error, (0, pad_w, 0, pad_h), value=0)
    
    color_error_downscale = (
        F.avg_pool2d(
            color_error_pad.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=stride,
            stride=stride,
        )
        .squeeze(0)
        .squeeze(0)
    )
    sample_num = int(torch.numel(color_error_downscale) * top_ratio)
    _, color_error_index = torch.topk(color_error_downscale.view(-1), k=sample_num)
    top_values_indices = torch.stack(
        [
            color_error_index // color_error_downscale.shape[1],
            color_error_index % color_error_downscale.shape[1],
        ],
        dim=1,
    )
    tile_mask = devI(torch.zeros_like(color_error_downscale))
    tile_mask[top_values_indices[:, 0], top_values_indices[:, 1]] = 1
    return tile_mask


def bbox_filter(local_xyz, total_xyz, padding=0.05):
    local_min = local_xyz.min(dim=0)[0] - padding
    local_max = local_xyz.max(dim=0)[0] + padding
    inbbox_mask = (total_xyz > local_min).all(dim=-1) & (total_xyz < local_max).all(
        dim=-1
    )

    return inbbox_mask
