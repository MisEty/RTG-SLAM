import os
from argparse import ArgumentParser


from utils.config_utils import read_config

parser = ArgumentParser(description="Eval script parameters")
parser.add_argument("--config", type=str)
parser.add_argument("--load_frame", type=int, default=-1)
parser.add_argument("--eval_frames", type=int, default=-1)
parser.add_argument("--load_iter", nargs="+", type=int, default=[])
parser.add_argument("--eval_merge", action="store_true")
parser.add_argument("--save_pic", action="store_true")

eval_args = parser.parse_args()
config_path = eval_args.config
args = read_config(config_path)
# set visible devices
device_list = args.device_list
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in device_list)

from utils.camera_utils import loadCam
import pandas as pd
import torch
from tqdm import tqdm
from arguments import DatasetParams, MapParams, OptimizationParams
from scene import Dataset
from SLAM.multiprocess.mapper import Mapping
from SLAM.utils import *
from SLAM.eval import eval_frame
from utils.general_utils import safe_state


torch.set_printoptions(4, sci_mode=False)


def filter_models(frame_path, eval_merge, load_iter):
    if eval_merge:
        exclud_ = "stable"
        include_ = "merge"
    else:
        exclud_ = "merge"
        include_ = "stable"
    total_models = [
        i for i in os.listdir(frame_path) if "sibr" not in i and exclud_ not in i
    ]
    select_models = []
    if len(load_iter) > 0:
        for eval_iter in load_iter:
            model_iter = [i for i in total_models if "%04d" % eval_iter in i]
            merge_models = [i for i in model_iter if include_ in i]
            if len(merge_models) > 0:
                select_models.extend(merge_models)
            else:
                select_models.extend(model_iter)
    else:
        max_iter = sorted([i[5:9] for i in total_models], reverse=True)[0]
        total_models = [i for i in total_models if max_iter in i]
        merge_models = [i for i in total_models if include_ in i]
        if len(merge_models) > 0:
            select_models.extend(merge_models)
        else:
            select_models.extend(total_models)
    return select_models


def move_to_gpu(frame):
    frame.original_depth = devF(frame.original_depth)
    frame.original_image = devF(frame.original_image)


def move_to_cpu(frame):
    frame.original_depth = frame.original_depth.to("cpu")
    frame.original_image = frame.original_image.to("cpu")


def read_pose_t0(args):
    data_type = args.type
    if data_type == "Replica":
        pose_t0_c2w = np.loadtxt(os.path.join(args.source_path, "traj.txt"))[0].reshape(
            4, 4
        )
    elif data_type == "Scannetpp":
        pose_t0_c2w = np.loadtxt(os.path.join(args.source_path, "pose", "0000.txt")).reshape(4,4)
    else:
        pose_t0_c2w = np.eye(4)
    return pose_t0_c2w


def main():
    if not os.path.exists(os.path.join(args.save_path, "eval_metric")):
        os.system("rm -r {}".format(os.path.join(args.save_path, "eval_metric")))

    load_iter = eval_args.load_iter
    load_frame = eval_args.load_frame
    eval_merge = eval_args.eval_merge
    eval_frames = eval_args.eval_frames
    model_base = os.path.join(args.save_path, "save_model")
    frames = [i for i in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, i))]
    frames = sorted(frames)
    if load_frame < 0:
        check_frame = frames[-1]
    else:
        check_frame = [i for i in frames if "%04d" % load_frame in i][0]
    print("check frame: ", check_frame)
    if eval_frames < 0:
        max_cams = int(check_frame.split("_")[-1])
    else:
        max_cams = min(eval_frames, int(check_frame.split("_")[-1]))
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser, sentinel=True)
    map_params = MapParams(parser)

    safe_state(args.quiet)
    save_pic = eval_args.save_pic
    optimization_params = optimization_params.extract(args)
    dataset_params = dataset_params.extract(args)
    dataset_params.frame_num = max_cams
    map_params = map_params.extract(args)

    # read pose_es
    if not args.use_gt_pose:
        pose_es = np.load(os.path.join(args.save_path, "save_traj", "pose_es.npy")).reshape(
            -1, 4, 4
        )[args.frame_start :, ...]

    # Initialize dataset
    dataset = Dataset(
        dataset_params,
        shuffle=False,
        resolution_scales=dataset_params.resolution_scales,
    )

    pose_t0_c2w = read_pose_t0(args)
    pose_t0_w2c = np.linalg.inv(pose_t0_c2w)

    # evaluate depth map opaque
    args.renderer_opaque_threshold = args.renderer_opaque_threshold_eval
    pcd_densify = args.pcd_densify
    
    gaussian_map = Mapping(args)

    frame_id = int(check_frame.split("_")[1])
    gaussian_map.time = frame_id
    frame_path = os.path.join(model_base, check_frame)
    select_models = filter_models(frame_path, eval_merge, load_iter)

    print("select models", select_models)
    select_model = select_models[0]
    test_iter = select_model[5:9]
    print("test_iter: ", test_iter)
    
    select_ply = os.path.join(frame_path, select_model)
    gaussian_map.pointcloud.load(select_ply)

    if pcd_densify:
        pcd_path = os.path.join(model_base, "pcd_densify.ply")
        if not os.path.exists(pcd_path):
            pcd_path = select_ply
    else:
        pcd_path = select_ply
    
    print("geometry eval ply: ", pcd_path)

    gaussian_map.iter = int(test_iter)
    total_loss = []
    run_pcd = False
    for cam_id, frame_info in tqdm(
        enumerate(dataset.scene_info.train_cameras),
        desc="Evaluating",
        total=len(dataset.scene_info.train_cameras),
    ):
        test_frame = loadCam(
            dataset_params,
            frame_id,
            frame_info,
            dataset_params.resolution_scales[0],
        )
        move_to_gpu(test_frame)
        if not args.use_gt_pose:
            test_frame.updatePose(pose_es[cam_id])
        gaussian_map.time = cam_id
        if cam_id == len(dataset.scene_info.train_cameras) - 1:
            run_pcd = True
        move_to_gpu(test_frame)

        losses = eval_frame(
            gaussian_map,
            test_frame,
            os.path.join(gaussian_map.save_path, "eval_metric"),
            run_picture=True,
            run_pcd=run_pcd,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            pcd_path=pcd_path,
            gt_mesh_path=dataset.mesh_path,
            dist_threshs=[0.03],
            sample_nums=1000000,
            pcd_transform=pose_t0_c2w,
            save_picture= save_pic,
        )
        
        losses["frame"] = gaussian_map.time
        losses["iter"] = gaussian_map.iter
        total_loss.append(losses)
        move_to_cpu(test_frame)


    df = pd.DataFrame(total_loss)
    print(df.mean())
    mean_row = df.mean().to_frame().T
    mean_row["frame"] = "mean"
    df = pd.concat([df, mean_row], ignore_index=True)
    df.to_csv(
        os.path.join(
            args.save_path,
            "statis_frame_{}_iter_{}.csv".format(frame_id, test_iter),
        )
    )


if __name__ == "__main__":
    main()
