import os
import json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_base", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    data_base = args.data_base
    output_path = args.output_path

    data_path = os.path.join(data_base, "dslr")
    mesh_path = os.path.join(data_base, "scans")
    
    scene_name = os.path.basename(os.path.dirname(data_path))
    save_path = os.path.join(output_path, scene_name)
    img_save_path = os.path.join(save_path, "color")
    img_eval_save_path = os.path.join(save_path, "color_eval")
    intrinsic_save_path = os.path.join(save_path, "intrinsic")
    depth_save_path = os.path.join(save_path, "depth")
    depth_eval_save_path = os.path.join(save_path, "depth_eval")
    intrinsic_save_path = os.path.join(save_path, "intrinsic")
    pose_save_path = os.path.join(save_path, "pose")
    pose_eval_save_path = os.path.join(save_path, "pose_eval")

    # os.system("rm -r {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(depth_save_path, exist_ok=True)
    os.makedirs(intrinsic_save_path, exist_ok=True)
    os.makedirs(pose_save_path, exist_ok=True)
    os.makedirs(img_eval_save_path, exist_ok=True)
    os.makedirs(depth_eval_save_path, exist_ok=True)
    os.makedirs(pose_eval_save_path, exist_ok=True)
    
    
    ply_files = [i for i in os.listdir(mesh_path) if ".ply" in i]
    for ply_file in ply_files:
        os.system("cp {} {}".format(os.path.join(mesh_path, ply_file),
                                    os.path.join(save_path, ply_file)))


    img_read_path = os.path.join(data_path, "undistorted_images")
    depth_read_path = os.path.join(data_path, "undistorted_depths")
    pose_read_path = os.path.join(
        data_path, "nerfstudio", "transforms_undistorted.json"
    )

    with open(pose_read_path, "r") as pose_file:
        pose_intrinsic = json.load(pose_file)

    pose_intrinsic["frames"] = sorted(
        pose_intrinsic["frames"], key=lambda x: x["file_path"]
    )

    pose_intrinsic["test_frames"] = sorted(
        pose_intrinsic["test_frames"], key=lambda x: x["file_path"]
    )
    
    fx = pose_intrinsic["fl_x"]
    fy = pose_intrinsic["fl_y"]
    cx = pose_intrinsic["cx"]
    cy = pose_intrinsic["cy"]

    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    np.savetxt(
        os.path.join(intrinsic_save_path, "intrinsic_color.txt"), intrinsic, fmt="%f"
    )
    np.savetxt(
        os.path.join(intrinsic_save_path, "intrinsic_depth.txt"), intrinsic, fmt="%f"
    )

    frame_id = 0

    for frame in tqdm(pose_intrinsic["frames"]):
        is_bad = frame["is_bad"]
        if is_bad:
            continue
        color_file = frame["file_path"]
        depth_file = color_file.replace(".JPG", ".png")
        pose_file = color_file.replace(".JPG", ".txt")
        pose_c2w = np.array(frame["transform_matrix"]).reshape(4, 4)
        pose_c2w[:, 1:3] *= -1

        # copy image
        os.system(
            "cp {} {}".format(
                os.path.join(img_read_path, color_file),
                os.path.join(img_save_path, "%04d.jpg" % frame_id),
            )
        )

        os.system(
            "cp {} {}".format(
                os.path.join(depth_read_path, depth_file),
                os.path.join(depth_save_path, "%04d.png" % frame_id),
            )
        )

        np.savetxt(
            os.path.join(pose_save_path, "%04d.txt" % frame_id),
            pose_c2w.tolist(),
            fmt="%f",
        )
        frame_id += 1

    frame_id = 0

    for frame in tqdm(pose_intrinsic["test_frames"]):
        is_bad = frame["is_bad"]
        if is_bad:
            continue
        color_file = frame["file_path"]
        depth_file = color_file.replace(".JPG", ".png")
        pose_file = color_file.replace(".JPG", ".txt")
        pose_c2w = np.array(frame["transform_matrix"]).reshape(4, 4)
        pose_c2w[:, 1:3] *= -1

        # copy image
        os.system(
            "cp {} {}".format(
                os.path.join(img_read_path, color_file),
                os.path.join(img_eval_save_path, "%04d.jpg" % frame_id),
            )
        )

        os.system(
            "cp {} {}".format(
                os.path.join(depth_read_path, depth_file),
                os.path.join(depth_eval_save_path, "%04d.png" % frame_id),
            )
        )

        np.savetxt(
            os.path.join(pose_eval_save_path, "%04d.txt" % frame_id),
            pose_c2w.tolist(),
            fmt="%f",
        )
        frame_id += 1
