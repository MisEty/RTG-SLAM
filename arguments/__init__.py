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

import os
import sys
from argparse import ArgumentParser, Namespace

import numpy as np


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                if "densify_until_iter" == arg[0]:
                    print(arg[0], arg[1])
                setattr(group, arg[0], arg[1])
        return group

    def extract_dict(self, config):
        group = GroupParams()
        for k, v in config.items():
            if k in vars(self) or ("_" + k) in vars(self):
                setattr(group, k, v)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.init_mode = "random"
        self.frame_num = -1
        self.eval_llff = 8
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.debug_gaussian_id = -1
        self.use_network_gui = False
        self.render_mode = "raw"
        self.fix_position = False
        self.fix_opacity = False
        self.init_opacity = 0.5
        self.fix_sh = False
        self.fix_cov = False
        self.fix_density = False
        self.opaque_threshold = 0.9

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.train_iterations = 30_000
        self.position_lr = 0.0016
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.color_weight = 0.8
        self.depth_weight = 1
        self.ssim_weight = 0.2
        self.history_weight = 0.1
        self.normal_weight = 0.1
        super().__init__(parser, "Optimization Parameters")


class DatasetParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.type = "ours"
        self.data_device = "cuda"
        self.eval = False
        self.init_mode = "random"
        self.frame_num = -1
        self.frame_start = 0
        self.frame_step = 0
        self.eval_llff = 8
        self.sh_degree = 3
        self.preload = False
        self.resolution_scales = [1.0]
        super().__init__(parser, "Dataset Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class MapParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.init_opacity = 0.999
        self.max_sh_degree = 4
        self.active_sh_degree = -1
        self.uniform_sample_num = 5000
        self.gaussian_update_iter = 300
        self.gaussian_update_frame = 1
        self.KNN_num = 15
        self.KNN_threshold = 0.005

        self.spatial_lr_scale = 1
        self.save_path = "output/slam_test"
        self.min_depth = 0
        self.max_depth = 0
        self.renderer_opaque_threshold = 0.7
        self.renderer_normal_threshold = 80
        self.renderer_depth_threshold = 1.0
        self.render_mode = "ours"
        
        self.memory_length = 10
        self.xyz_factor = [1, 1, 1]
        self.use_tensorboard = True
        self.add_depth_thres = 0.05
        self.add_normal_thres = 0.1
        self.add_color_thres = 0.1
        self.add_transmission_thres = 0.1
        self.transmission_sample_ratio = 0.5
        self.error_sample_ratio = 0.3
        self.save_step = 1
        self.stable_confidence_thres = 200
        self.unstable_time_window = 50
        self.min_radius = 0.01
        self.max_radius = 0.10
        self.scale_factor = 0.5
        self.color_sigma = 1.0
        self.depth_filter = False
        self.verbose = False


        self.keyframe_trans_thes = 0.3
        self.keyframe_theta_thes = 20
        self.global_keyframe_num = 3
        self.sync_tracker2mapper_method = "strict"
        self.sync_tracker2mapper_frames = 5
        super().__init__(parser, "Map Parameters", sentinel)


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
