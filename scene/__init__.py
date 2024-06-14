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

from arguments import DatasetParams
from scene.dataset_readers import sceneLoadTypeCallbacks


class Dataset:
    def __init__(
        self,
        args: DatasetParams,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        self.train_cameras = {}
        self.test_cameras = {}
        if args.type == "TUM":
            print("Assuming TUM data set!")
            scene_info = sceneLoadTypeCallbacks["Tum"](
                args.source_path,
                args.eval,
                args.eval_llff,
                args.frame_start,
                args.frame_num,
                args.frame_step,
            )
        elif args.type == "Replica":
            print("Assuming Replica data set!")
            scene_info = sceneLoadTypeCallbacks["Replica"](
                args.source_path,
                args.eval,
                args.eval_llff,
                args.frame_start,
                args.frame_num,
                args.frame_step,
            )
        elif args.type == "Ours":
            print("Assuming Ours dataset!")
            scene_info = sceneLoadTypeCallbacks["ours"](
                args.source_path,
                args.eval,
                args.eval_llff,
                args.frame_start,
                args.frame_num,
                args.frame_step,
            )
        elif args.type == "Scannetpp":
            print("Assuming Scannetpp dataset!")
            scene_info = sceneLoadTypeCallbacks["Scannetpp"](
                args.source_path,
                args.eval,
                args.eval_llff,
                args.frame_start,
                args.frame_num,
                args.frame_step,
                isscannetpp=True
            )
        else:
            print("scene dataset path:", args.source_path)
            assert False, "Could not recognize scene type!"

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.mesh_path = scene_info.mesh_path
        self.scene_info = scene_info


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
