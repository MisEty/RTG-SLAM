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

import glob
import json
import os
import sys
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import yaml
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import (
    focal2fov,
    fov2focal,
    getK,
    getWorld2View,
    getWorld2View2,
)
from utils.sh_utils import SH2RGB


def get_split_list(a, num):
    if num == 1:
        return a[0:1]
    elif num == -1:
        return a
    else:
        a_num = len(a)
        return a[0 : a_num : a_num // num][:num]


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    depth_path: str
    pose_gt: np.array = np.eye(4)
    cx: float = -1
    cy: float = -1
    depth_scale: float = 1
    timestamp: float = -1


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud  # position, color, normal pcd
    train_cameras: list  # list[CameraInfo]
    test_cameras: list  # list[CameraInfo]
    nerf_normalization: dict  # Cameras center and radius extent
    ply_path: str  # ply file path
    mesh_path: str  # check mesh path


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    if radius == 0:
        radius = 1
    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        depth = Image.fromarray(np.zeros(image.width, image.height))
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            depth=depth,
            depth_path="",
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(
    path,
    images,
    eval,
    pts_num,
    llffhold=8,
    head_frames=-1,
    frame_num=300,
    init_mode="colmap",
):
    if os.path.exists(os.path.join(path, "sparse/0", "images.bin")):
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    elif os.path.exists(os.path.join(path, "sparse/0", "images.txt")):
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    elif os.path.exists(os.path.join(path, "sparse", "images.txt")):
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    elif os.path.exists(os.path.join(path, "sparse", "images.bin")):
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    cam_infos = cam_infos[:head_frames]
    cam_infos = get_split_list(cam_infos, frame_num)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_mode == "random":
        ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
        # Since this data set has no colmap data, we start with random points

        print(f"Generating random point cloud ({pts_num})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = sample_ply_random(train_cam_infos, pts_num)
        shs = np.random.random((pts_num, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((pts_num, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)

            storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readTumCameras(
    cam_extrinsics,
    cam_intrinsics,
    image_paths,
    depth_paths,
    associations,
    poses,
    indices,
    timestamps,
):
    cam_infos = []
    for idx_ in range(len(indices)):
        # idx = indices[idx_]
        idx = idx_
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(indices)))
        sys.stdout.flush()
        (i, j, k) = associations[idx]
        image_path = image_paths[idx]
        depth_path = depth_paths[idx]
        c2w = cam_extrinsics[idx]

        width, height = cam_intrinsics["W"], cam_intrinsics["H"]
        uid = 0
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        fx, fy = cam_intrinsics["fx"], cam_intrinsics["fy"]
        cx, cy = cam_intrinsics["cx"], cam_intrinsics["cy"]
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        depth_scale = cam_intrinsics["depth_scale"]
        image_name = "_".join(os.path.basename(image_path).split(".")[:2])
        image = Image.open(image_path)
        K = getK(fx, fy, cx, cy)
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / depth_scale
        depth_filter = cv2.bilateralFilter(depth, 5, 2, 2)
        print(depth_filter[depth_filter != 0].min())
        depth_filter = depth_filter * (depth != 0)
        depth = depth_filter * (depth_filter > 0.2)
        if "crop_edge" in cam_intrinsics and cam_intrinsics["crop_edge"] > 0:
            crop_edge = cam_intrinsics["crop_edge"]
            image = np.asarray(image)[
                crop_edge : -crop_edge + 1, crop_edge : -crop_edge + 1, :
            ]
            depth = depth[crop_edge : -crop_edge + 1, crop_edge : -crop_edge + 1]
            width = image.shape[1]
            height = image.shape[0]
            cx = cx - crop_edge
            cy = cy - crop_edge
            image = Image.fromarray(image)
            depth = Image.fromarray(depth)

        cam_info = CameraInfo(
            uid=idx_,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            depth=depth,
            depth_path=depth_path,
            pose_gt=poses[idx],
            cx=cx,
            cy=cy,
            timestamp=timestamps[idx],
            depth_scale=depth_scale,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def sample_ply_random(cam_infos: List[CameraInfo], sample_nums=10000):
    bbox_min = np.ones(3) * float("inf")
    bbox_max = np.ones(3) * float("-inf")
    for cam in cam_infos:
        R, t = cam.R, cam.T
        w2c = getWorld2View(R, t)
        c2w = np.linalg.inv(w2c)
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        bbox_min = np.minimum(bbox_min, t)
        bbox_max = np.maximum(bbox_max, t)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_min = bbox_min - (bbox_center - bbox_min) * 0.3
    bbox_max = bbox_max + (bbox_max - bbox_center) * 0.3
    xyz = np.random.random((sample_nums, 3))
    xyz = bbox_min + (bbox_max - bbox_min) * xyz
    return xyz


def sample_ply_from_depth(
    cam_infos: List[CameraInfo], sample_nums=10000, intrinsic_depth=None
):
    points = []
    points_color = []
    frame_nums = len(cam_infos)
    sample_frame_num = sample_nums // frame_nums
    to_do = sample_nums
    # 将所有的深度反投影至世界空间
    for idx in range(len(cam_infos)):
        if idx == frame_nums - 1:
            samples = to_do
        else:
            samples = sample_frame_num
        cam = cam_infos[idx]
        depth_np = np.asarray(cam.depth)
        image_np = cam.image
        j, i = np.where(depth_np != 0)
        i, j = i[:, None], j[:, None]
        if intrinsic_depth is None:
            w, h = cam.width, cam.height
            fx, fy = fov2focal(cam.FovX, w), fov2focal(cam.FovY, h)
            cx, cy = cam.width / 2, cam.height / 2
        else:
            cx, cy = intrinsic_depth[0, 2], intrinsic_depth[1, 2]
            w, h = cx * 2, cy * 2
            fx, fy = intrinsic_depth[0, 0], intrinsic_depth[1, 1]
        R, t = cam.R, cam.T
        w2c = getWorld2View(R, t)
        c2w = np.linalg.inv(w2c)
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        point = (
            np.concatenate([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)
            * depth_np[j, i].flatten()[:, None]
        )
        point = point @ R.transpose() + t
        P = point.reshape(-1, 3).shape[0]
        random_indices = np.random.choice(np.arange(P), samples, replace=False)
        points.append(point[random_indices, :])
        point_color = np.asarray(image_np)[j, i, :].reshape(-1, 3) / 255.0
        point_color = point_color.reshape(-1, 3)[random_indices, :]
        points_color.append(point_color)
        to_do -= samples
    # 随机采样sample_nums个点
    points = np.concatenate(points, axis=0)
    points_color = np.concatenate(points_color, axis=0)
    return points, points_color


def readTumSceneInfo(
    datapath, eval, llffhold, frame_start=0, frame_num=-1, frame_step=1
):
    def parse_list(filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def pose_matrix_from_quaternion(pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    """ read video data in tum-rgbd format """
    if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
        pose_list = os.path.join(datapath, "groundtruth.txt")
    elif os.path.isfile(os.path.join(datapath, "pose.txt")):
        pose_list = os.path.join(datapath, "pose.txt")

    config_path = os.path.join(datapath, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read config
    depth_scale = config["depth_scale"]
    intrinsic = np.array(
        [[config["fx"], 0, config["cx"]], [0, config["fy"], config["cy"]], [0, 0, 1]]
    )
    crop_edge = config["crop_edge"]

    image_list = os.path.join(datapath, "rgb.txt")
    depth_list = os.path.join(datapath, "depth.txt")

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:, 1:].astype(np.float64)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_depth = depth_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)
    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

    
    indicies = [0]
    frame_rate = 32
    for i in range(1, len(associations)):
        t0 = tstamp_image[associations[indicies[-1]][0]]
        t1 = tstamp_image[associations[i][0]]
        if t1 - t0 > 1.0 / frame_rate:
            indicies += [i]

    n_img = len(indicies)
    if frame_num == -1:
        indexs = list(range(n_img))
    else:
        indexs = list(range(frame_num))
    indicies = [frame_start + i * (frame_step + 1) for i in indexs]
    indicies = [i for i in indicies if i < n_img]
    
    color_paths, poses, depth_paths, timestamps = [], [], [], []
    inv_pose = None
    rgbd_pose_tupe = []
    for idx in range(len(indicies)):
        ix = indicies[idx]
        (i, j, k) = associations[ix]
        color_paths += [os.path.join(datapath, image_data[i, 1])]
        depth_paths += [os.path.join(datapath, depth_data[j, 1])]
        rgbd_pose_tupe.append([image_data[i, 1], depth_data[j, 1], tstamp_pose[k]])
        c2w = pose_matrix_from_quaternion(pose_vecs[k])
        if inv_pose is None:
            inv_pose = np.linalg.inv(c2w)
            c2w = np.eye(4)
        else:
            c2w = inv_pose @ c2w
        poses += [c2w]
        timestamps.append(tstamp_image[i])

    with open(os.path.join(datapath, "associations_nerf.txt"), "w") as f:
        for cuple in rgbd_pose_tupe:
            f.write("{} {} {}\n".format(cuple[0], cuple[1], cuple[2]))

    np.save(os.path.join(datapath, "pose_gt_nerf.npy"), poses)
    
    cam_infos_unsorted = readCameras(
        color_paths=color_paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsic=intrinsic,
        indices=range(len(color_paths)),
        depth_scale=depth_scale,
        timestamps=timestamps,
        crop_edge=crop_edge,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    saveCameraJson(
        poses,
        {
            "h": cam_infos[0].height,
            "w": cam_infos[0].width,
            "fx": intrinsic[0, 0],
            "fy": intrinsic[1, 1],
        },
        datapath,
    )

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
        mesh_path=None,
    )
    return scene_info


def readReplicaCameras(color_paths, depth_paths, poses, config, indices):
    cam_infos = []
    for idx_ in range(len(indices)):
        idx = indices[idx_]
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx_ + 1, len(indices)))
        sys.stdout.flush()

        depth_scale = config["scale"]
        image_color = Image.open(color_paths[idx])
        image_depth = Image.fromarray(
            np.asarray(Image.open(depth_paths[idx])) / depth_scale
        )
        c2w = poses[idx]
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        fx, fy = config["fx"], config["fy"]
        cx, cy = config["cx"], config["cy"]
        width, height = config["w"], config["h"]
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        image_name = "_".join(os.path.basename(color_paths[idx]).split(".")[0])
        cam_info = CameraInfo(
            uid=idx_,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image_color,
            image_path=color_paths[idx],
            image_name=image_name,
            width=width,
            height=height,
            depth=image_depth,
            depth_path=depth_paths[idx],
            pose_gt=c2w,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def saveCfg(poses, config, save_path):
    cameras = []
    for idx in range(len(poses)):
        width, height = config["w"], config["h"]
        c2w = poses[idx]

        R = c2w[:3, :3]
        T = c2w[:3, 3]
        position = T.tolist()
        rotation = [x.tolist() for x in R]
        id = idx
        img_name = "frame_%04d" % idx
        fx, fy = config["fx"], config["fy"]
        cameras.append(
            {
                "id": id,
                "img_name": img_name,
                "width": width,
                "height": height,
                "position": position,
                "rotation": rotation,
                "fx": fx,
                "fy": fy,
            }
        )
    with open(os.path.join(save_path, "cameras.json"), "w") as file:
        json.dump(cameras, file)


def readReplicaSceneInfo(
    datapath, eval, llffhold, frame_start=0, frame_num=1, frame_step=1
):
    def load_poses(path, n_img):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        pose_w_t0 = np.eye(4)
        for i in range(n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            if i == 0:
                pose_w_t0 = np.linalg.inv(c2w)
            c2w = pose_w_t0 @ c2w
            poses.append(c2w)
        return poses

    color_paths = sorted(glob.glob(f"{datapath}/results/frame*.jpg"))
    depth_paths = sorted(glob.glob(f"{datapath}/results/depth*.png"))
    n_img = len(color_paths)
    timestamps = [i / 30.0 for i in range(n_img)]

    poses = load_poses(f"{datapath}/traj.txt", n_img)
    if frame_num == -1:
        indicies = list(range(n_img))
    else:
        frame_num = min(n_img, frame_num)
        indicies = list(range(frame_num))
    indicies = [frame_start + i * (frame_step + 1) for i in indicies]
    with open(os.path.join(datapath, "../cam_params.json"), "r") as f:
        config = json.load(f)["camera"]
    intrinsic = np.eye(3)
    intrinsic[0, 0] = config["fx"]
    intrinsic[1, 1] = config["fx"]
    intrinsic[0, 2] = config["cx"]
    intrinsic[1, 2] = config["cy"]

    cam_infos = readCameras(
        color_paths=color_paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsic=intrinsic,
        indices=indicies,
        depth_scale=config["scale"],
        timestamps=timestamps,
        crop_edge=0,
    )
    saveCfg(poses, config, datapath)
    if eval:
        train_cam_infos = [
            c for idx, c in enumerate(cam_infos) if (idx + 1) % llffhold != 0
        ]
        test_cam_infos = [
            c for idx, c in enumerate(cam_infos) if (idx + 1) % llffhold == 0
        ]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene = os.path.basename(datapath)
    mesh_path = "{}.ply".format(scene)
    mesh_path = os.path.join(datapath, mesh_path)
    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
        mesh_path=mesh_path,
    )
    return scene_info


def readCameras(
    color_paths,
    depth_paths,
    poses,
    intrinsic,
    indices,
    depth_scale,
    timestamps,
    crop_edge=0,
    eval_=False,
):
    cam_infos = []
    pose_w_t0 = np.eye(4)
    for idx_ in range(len(indices)):
        idx = indices[idx_]
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx_ + 1, len(indices)))
        sys.stdout.flush()

        c2w = poses[idx]
        if idx_ == 0:
            pose_w_t0 = np.linalg.inv(c2w)
        # pass invalid pose
        if np.isinf(c2w).any():
            continue
        if not eval_:
            c2w = pose_w_t0 @ c2w
        poses[idx] = c2w
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_color = Image.open(color_paths[idx])
        image_depth = (
            np.asarray(Image.open(depth_paths[idx]), dtype=np.float32) / depth_scale
        )
        image_color = np.asarray(
            image_color.resize((image_depth.shape[1], image_depth.shape[0]))
        )
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        if crop_edge > 0:
            image_color = image_color[
                crop_edge:-crop_edge,
                crop_edge:-crop_edge,
                :,
            ]
            image_depth = image_depth[
                crop_edge:-crop_edge,
                crop_edge:-crop_edge,
            ]
            cx -= crop_edge
            cy -= crop_edge

        height, width = image_color.shape[:2]
        # print("image size:", height, width)
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        image_name = os.path.basename(color_paths[idx]).split(".")[0]

        cam_info = CameraInfo(
            uid=idx_,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=Image.fromarray(image_color),
            image_path=color_paths[idx],
            image_name=image_name,
            width=width,
            height=height,
            depth=Image.fromarray(image_depth),
            depth_path=depth_paths[idx],
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            timestamp=timestamps[idx],
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos

def saveCameraJson(poses, config, save_path):
    cameras = []
    for idx in range(len(poses)):
        width, height = config["w"], config["h"]
        c2w = poses[idx]
        # pass invalid pose
        if np.isinf(c2w).any():
            print("get inf at frame {:d}".format(idx))
            continue
        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        position = T.tolist()
        rotation = [x.tolist() for x in R]
        id = idx
        img_name = "frame_%04d" % idx
        fx, fy = config["fx"], config["fy"]
        cameras.append(
            {
                "id": id,
                "img_name": img_name,
                "width": width,
                "height": height,
                "position": position,
                "rotation": rotation,
                "fx": fx,
                "fy": fy,
            }
        )
    with open(os.path.join(save_path, "cameras.json"), "w") as file:
        json.dump(cameras, file)


def readOursSceneInfo(
    datapath, eval_, llffhold, frame_start=0, frame_num=100, frame_step=0, isscannetpp=False
):
    def load_poses(datapaths, n_img):
        poses = []
        for i in range(n_img):
            pose_file = datapaths[i]
            pose = np.loadtxt(pose_file)
            poses.append(pose)
        return poses

    color_path = "color"
    depth_path = "depth"
    pose_path = "pose"
    if eval_:
        color_path += "_eval"
        depth_path += "_eval"
        pose_path += "_eval"

    color_paths = sorted(
        glob.glob(f"{datapath}/{color_path}/*.jpg"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    depth_paths = sorted(
        glob.glob(f"{datapath}/{depth_path}/*.png"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    n_img = len(color_paths)
    timestamps = [(i+1) / 30.0 for i in range(n_img)]

    crop_edge = 0

    pose_paths = sorted(
        glob.glob(f"{datapath}/{pose_path}/*.txt"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    
    poses = load_poses(pose_paths, n_img)
    if eval_:
        eval_list = os.path.join(datapath, "eval_list.txt")
        if os.path.exists(eval_list):
            eval_list = list(np.loadtxt(eval_list, dtype=np.int32))
            print("eval_list:", eval_list)
            color_paths = [
                color_paths[i] for i in range(len(color_paths)) if i in eval_list
            ]
            depth_paths = [
                depth_paths[i] for i in range(len(depth_paths)) if i in eval_list
            ]
            poses = [poses[i] for i in range(len(poses)) if i in eval_list]
            n_img = len(poses)
    if eval_:
        pose_t0_c2w_fake = load_poses(f"{datapath}/pose", 1)[0]
        pose_t0_w2c = np.linalg.inv(pose_t0_c2w_fake)
        for i in range(len(poses)):
            poses[i] = pose_t0_w2c @ poses[i]
    if frame_num == -1:
        indicies = list(range(n_img))
    else:
        indicies = list(range(frame_num))
    indicies = [frame_start + i * (frame_step + 1) for i in indicies]
    indicies = [i for i in indicies if i < n_img]

    if eval_:
        indicies = list(range(n_img))

    intrinsic = np.loadtxt(os.path.join(datapath, "intrinsic", "intrinsic_depth.txt"))

    cam_infos = readCameras(
        color_paths,
        depth_paths,
        poses,
        intrinsic,
        indicies,
        depth_scale=1000.0,
        timestamps=timestamps,
        crop_edge=crop_edge,
        eval_=eval_,
    )
    saveCameraJson(
        poses,
        {
            "h": cam_infos[0].height,
            "w": cam_infos[0].width,
            "fx": intrinsic[0, 0],
            "fy": intrinsic[1, 1],
        },
        datapath,
    )

    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    mesh_path = None
    if isscannetpp:
        mesh_path = os.path.join(datapath, "mesh_aligned_cull.ply")
    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
        mesh_path=mesh_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Tum": readTumSceneInfo,
    "Replica": readReplicaSceneInfo,
    "ours": readOursSceneInfo,
    "Scannetpp": readOursSceneInfo,
}
