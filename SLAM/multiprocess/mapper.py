import os
import shutil
import time
import random
import copy
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque
from scene.cameras import Camera
from SLAM.gaussian_pointcloud import *
from SLAM.render import Renderer
from SLAM.utils import merge_ply, rot_compare, trans_compare, bbox_filter
from utils.loss_utils import l1_loss, l2_loss, ssim
from cuda_utils._C import accumulate_gaussian_error
from utils.monitor import Recorder

class Mapping(object):
    def __init__(self, args, recorder=None) -> None:
        self.temp_pointcloud = GaussianPointCloud(args)
        self.pointcloud = GaussianPointCloud(args)
        self.stable_pointcloud = GaussianPointCloud(args)
        self.recorder = recorder

        self.renderer = Renderer(args)
        self.optimizer = None
        self.time = 0
        self.iter = 0
        self.gaussian_update_iter = args.gaussian_update_iter
        self.gaussian_update_frame = args.gaussian_update_frame
        self.final_global_iter = args.final_global_iter
        
        # # history management
        self.memory_length = args.memory_length
        self.optimize_frames_ids = []
        self.processed_frames = deque(maxlen=self.memory_length)
        self.processed_map = deque(maxlen=self.memory_length)
        self.keyframe_ids = []
        self.keyframe_list = []
        self.keymap_list = []
        self.global_keyframe_num = args.global_keyframe_num
        self.keyframe_trans_thes = args.keyframe_trans_thes
        self.keyframe_theta_thes = args.keyframe_theta_thes
        self.KNN_num = args.KNN_num
        self.KNN_threshold = args.KNN_threshold
        self.history_merge_max_weight = args.history_merge_max_weight
        
        
        # points adding parameters
        self.uniform_sample_num = args.uniform_sample_num
        self.add_depth_thres = args.add_depth_thres
        self.add_normal_thres = args.add_normal_thres
        self.add_color_thres = args.add_color_thres
        self.add_transmission_thres = args.add_transmission_thres

        self.transmission_sample_ratio = args.transmission_sample_ratio
        self.error_sample_ratio = args.error_sample_ratio
        self.stable_confidence_thres = args.stable_confidence_thres
        self.unstable_time_window = args.unstable_time_window

        # all map shape is [H, W, C], please note the raw image shape is [C, H, W]
        self.min_depth, self.max_depth = args.min_depth, args.max_depth
        self.depth_filter = args.depth_filter
        self.frame_map = {
            "depth_map": torch.empty(0),
            "color_map": torch.empty(0),
            "normal_map_c": torch.empty(0),
            "normal_map_w": torch.empty(0),
            "vertex_map_c": torch.empty(0),
            "vertex_map_w": torch.empty(0),
            "confidence_map": torch.empty(0),
        }
        self.model_map = {
            "render_color": torch.empty(0),
            "render_depth": torch.empty(0),
            "render_normal": torch.empty(0),
            "render_color_index": torch.empty(0),
            "render_depth_index": torch.empty(0),
            "render_transmission": torch.empty(0),
            "confidence_map": torch.empty(0),
        }

        # parameters for eval
        self.save_path = args.save_path
        self.save_step = args.save_step
        self.verbose = args.verbose
        self.mode = args.mode
        self.dataset_type = args.type
        assert self.mode == "single process" or self.mode == "multi process"
        self.use_tensorboard = args.use_tensorboard
        self.tb_writer = None

        self.feature_lr_coef = args.feature_lr_coef
        self.scaling_lr_coef = args.scaling_lr_coef
        self.rotation_lr_coef = args.rotation_lr_coef
        
    def mapping(self, frame, frame_map, frame_id, optimization_params):
        self.frame_map = frame_map
        self.gaussians_add(frame)
        self.processed_frames.append(frame)
        self.processed_map.append(frame_map)

        if (self.time + 1) % self.gaussian_update_frame == 0 or self.time == 0:
            self.optimize_frames_ids.append(frame_id)
            is_keyframe = self.check_keyframe(frame, frame_id)
            move_to_gpu(frame)
            if self.dataset_type == "Scannetpp":
                self.local_optimize(frame, optimization_params)
                if is_keyframe:
                    self.global_optimization(
                        optimization_params,
                        select_keyframe_num=self.global_keyframe_num
                    )
            else:
                if not is_keyframe or self.get_stable_num <= 0:
                    self.local_optimize(frame, optimization_params)
                else:
                    self.global_optimization(
                        optimization_params,
                        select_keyframe_num=self.global_keyframe_num
                    )
                self.gaussians_delete(unstable=False)
        self.gaussians_fix()
        self.error_gaussians_remove()
        self.gaussians_delete()
        move_to_cpu(frame)

    def gaussians_add(self, frame):
        self.temp_points_init(frame)
        self.temp_points_filter()
        self.temp_points_attach(frame)
        self.temp_to_optimize()

    def update_poses(self, new_poses):
        if new_poses is None:
            return
        for frame in self.processed_frames:
            frame.updatePose(new_poses[frame.uid])

        for frame in self.keyframe_list:
            frame.updatePose(new_poses[frame.uid])

    def local_optimize(self, frame, update_args):
        print("===== map optimize =====")
        l = self.pointcloud.parametrize(update_args)
        history_stat = {
            "opacity": self.pointcloud._opacity.detach().clone(),
            "confidence": self.pointcloud.get_confidence.detach().clone(),
            "xyz": self.pointcloud._xyz.detach().clone(),
            "features_dc": self.pointcloud._features_dc.detach().clone(),
            "features_rest": self.pointcloud._features_rest.detach().clone(),
            "scaling": self.pointcloud._scaling.detach().clone(),
            "rotation": self.pointcloud.get_rotation.detach().clone(),
            "rotation_raw": self.pointcloud._rotation.detach().clone(),
        }
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        gaussian_update_iter = self.gaussian_update_iter
        render_masks = []
        tile_masks = []
        for frame in self.processed_frames:
            render_mask, tile_mask, render_ratio = self.evaluate_render_range(frame)
            render_masks.append(render_mask)
            tile_masks.append(tile_mask)
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)
                print("tile mask ratio: {:f}".format(tile_raito))
        print(
            "unstable gaussian num = {:d}, stable gaussian num = {:d}".format(
                self.get_unstable_num, self.get_stable_num
            )
        )

        with tqdm(total=gaussian_update_iter, desc="map update") as pbar:
            for iter in range(gaussian_update_iter):
                self.iter = iter
                random_index = random.randint(0, len(self.processed_frames) - 1)
                if iter > gaussian_update_iter / 2:
                    random_index = -1
                opt_frame = self.processed_frames[random_index]
                opt_frame_map = self.processed_map[random_index]
                opt_render_mask = render_masks[random_index]
                opt_tile_mask = tile_masks[random_index]
                # compute loss
                render_ouput = self.renderer.render(
                    opt_frame,
                    self.global_params,
                    tile_mask=opt_tile_mask,
                )
                image_input = {
                    "color_map": devF(opt_frame_map["color_map"]),
                    "depth_map": devF(opt_frame_map["depth_map"]),
                    "normal_map": devF(opt_frame_map["normal_map_w"]),
                }
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    history_stat,
                    update_args,
                    render_mask=opt_render_mask,
                    unstable=True,
                )
                pbar.set_postfix({"loss": "{0:1.5f}".format(loss)})
                pbar.update(1)


        self.pointcloud.detach()
        self.iter = 0

        self.history_merge(history_stat, self.history_merge_max_weight)

    def history_merge(self, history_stat, max_weight=0.5):
        if max_weight <= 0:
            return
        history_weight = (
            max_weight
            * history_stat["confidence"]
            / (self.pointcloud.get_confidence + 1e-6)
        )
        if self.verbose:
            print("===== history merge ======")
            print("history weight: {:.2f}".format(history_weight.mean()))
        xyz_merge = (
            history_stat["xyz"] * history_weight
            + (1 - history_weight) * self.pointcloud.get_xyz
        )

        features_dc_merge = (
            history_stat["features_dc"] * history_weight[0]
            + (1 - history_weight[0]) * self.pointcloud._features_dc
        )

        features_rest_merge = (
            history_stat["features_rest"] * history_weight[0]
            + (1 - history_weight[0]) * self.pointcloud._features_rest
        )

        scaling_merge = (
            history_stat["scaling"] * history_weight[0]
            + (1 - history_weight[0]) * self.pointcloud._scaling
        )
        rotation_merge = slerp(
            history_stat["rotation"], self.pointcloud.get_rotation, 1 - history_weight
        )

        self.pointcloud._xyz = xyz_merge
        self.pointcloud._features_dc = features_dc_merge
        self.pointcloud._features_rest = features_rest_merge
        self.pointcloud._scaling = scaling_merge
        self.pointcloud._rotation = rotation_merge

    # Fix the confidence points
    def gaussians_fix(self, mask=None):
        if mask is None:
            confidence_mask = (
                self.pointcloud.get_confidence > self.stable_confidence_thres
            ).squeeze()
            stable_mask = confidence_mask
        else:
            stable_mask = mask.squeeze()
        if self.verbose:
            print("===== points fix =====")
            print(
                "fix gaussian num: {:d}".format(stable_mask.sum()),
            )
        if stable_mask.sum() > 0:
            stable_params = self.pointcloud.remove(stable_mask)
            stable_params["confidence"] = torch.clip(
                stable_params["confidence"], max=self.stable_confidence_thres
            )
            self.stable_pointcloud.cat(stable_params)

    # Release the outlier points
    def gaussians_release(self, mask):
        if mask.sum() > 0:
            unstable_params = self.stable_pointcloud.remove(mask)
            unstable_params["confidence"] = devF(
                torch.zeros_like(unstable_params["confidence"])
            )
            unstable_params["add_tick"] = self.time * devF(
                torch.ones_like(unstable_params["add_tick"])
            )
            self.pointcloud.cat(unstable_params)

    # Remove too small/big gaussians, long time unstable gaussians, insolated_gaussians
    def gaussians_delete(self, unstable=True):
        if unstable:
            pointcloud = self.pointcloud
        else:
            pointcloud = self.stable_pointcloud
        if pointcloud.get_points_num == 0:
            return
        threshold = self.KNN_threshold
        big_gaussian_mask = (
            pointcloud.get_radius > (pointcloud.get_radius.mean() * 10)
        ).squeeze()
        unstable_time_mask = (
            (self.time - pointcloud.get_add_tick) > self.unstable_time_window
        ).squeeze()
        # isolated_gaussian_mask = self.gaussians_isolated(
        #     pointcloud.get_xyz, self.KNN_num, threshold
        # )
        if unstable:
            # delete_mask = (
            #     big_gaussian_mask | unstable_time_mask | isolated_gaussian_mask
            # )
            delete_mask = (
                big_gaussian_mask | unstable_time_mask 
            )
        else:
            # delete_mask = big_gaussian_mask | isolated_gaussian_mask
            delete_mask = big_gaussian_mask 
        if self.verbose:
            print("===== points delete =====")
            print(
                "threshold: {:.1f} cm, big num: {:d}, unstable num: {:d}, delete num: {:d}".format(
                    threshold * 100,
                    big_gaussian_mask.sum(),
                    unstable_time_mask.sum(),
                    delete_mask.sum(),
                ),
            )
        pointcloud.delete(delete_mask)

    # check if current frame is a keyframe
    def check_keyframe(self, frame, frame_id):
        # add keyframe
        if self.time == 0:
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keyframe_ids.append(frame_id)
            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
            }
            self.keymap_list.append(image_input)
            return False
        prev_rot = self.keyframe_list[-1].R.T
        prev_trans = self.keyframe_list[-1].T
        curr_rot = frame.R.T
        curr_trans = frame.T
        _, theta_diff = rot_compare(prev_rot, curr_rot)
        _, l2_diff = trans_compare(prev_trans, curr_trans)
        if self.verbose:
            print("rot diff: {:.2f}, move diff: {:.2f}".format(theta_diff, l2_diff))
        if theta_diff > self.keyframe_theta_thes or l2_diff > self.keyframe_trans_thes:
            print("add key frame at frame {:d}!".format(self.time))
            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
            }
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keymap_list.append(image_input)
            self.keyframe_ids.append(frame_id)
            return True
        else:
            return False

    # update confidence by grad
    def loss_update(
        self,
        render_output,
        image_input,
        init_stat,
        update_args,
        render_mask=None,
        unstable=True,
    ):
        if unstable:
            pointcloud = self.pointcloud
        else:
            pointcloud = self.stable_pointcloud
        opacity = pointcloud.opacity_activation(init_stat["opacity"])
        attach_mask = (opacity < 0.9).squeeze()
        attach_loss = torch.tensor(0)
        if attach_mask.sum() > 0:
            attach_loss = 1000 * (
                l2_loss(
                    pointcloud._scaling[attach_mask],
                    init_stat["scaling"][attach_mask],
                )
                + l2_loss(
                    pointcloud._xyz[attach_mask],
                    init_stat["xyz"][attach_mask],
                )
                + l2_loss(
                    pointcloud._rotation[attach_mask],
                    init_stat["rotation_raw"][attach_mask],
                )
            )
        image, depth, normal, depth_index = (
            render_output["render"].permute(1, 2, 0),
            render_output["depth"].permute(1, 2, 0),
            render_output["normal"].permute(1, 2, 0),
            render_output["depth_index_map"].permute(1, 2, 0),
        )
        ssim_loss = devF(torch.tensor(0))
        normal_loss = devF(torch.tensor(0))
        depth_loss = devF(torch.tensor(0))
        if render_mask is None:
            render_mask = devB(torch.ones(image.shape[:2]))
            ssim_loss = 1 - ssim(image.permute(2,0,1), image_input["color_map"].permute(2,0,1))
        else:
            render_mask = render_mask.bool()
        # render_mask include depth == 0
        if self.dataset_type == "Scannetpp":
            render_mask = render_mask & (image_input["depth_map"] > 0).squeeze()
        color_loss = l1_loss(image[render_mask], image_input["color_map"][render_mask])

        if depth is not None and update_args.depth_weight > 0:
            depth_error = depth - image_input["depth_map"]
            valid_depth_mask = (
                (depth_index != -1).squeeze()
                & (image_input["depth_map"] > 0).squeeze()
                & (depth_error < self.add_depth_thres).squeeze()
                & render_mask
            )
            depth_loss = torch.abs(depth_error[valid_depth_mask]).mean()

        if normal is not None and update_args.normal_weight > 0:
            cos_dist = 1 - F.cosine_similarity(
                normal, image_input["normal_map"], dim=-1
            )
            valid_normal_mask = (
                render_mask
                & (depth_index != -1).squeeze()
                & (~(image_input["normal_map"] == 0).all(dim=-1))
            )
            normal_loss = cos_dist[valid_normal_mask].mean()

        total_loss = (
            update_args.depth_weight * depth_loss
            + update_args.normal_weight * normal_loss
            + update_args.color_weight * color_loss
            + update_args.ssim_weight * ssim_loss
        )
        loss = total_loss
        (loss + attach_loss).backward()
        self.optimizer.step()

        # update confidence by grad
        grad_mask = (pointcloud._features_dc.grad.abs() != 0).any(dim=-1)
        pointcloud._confidence[grad_mask] += 1

        # report train loss
        report_losses = {
            "total_loss": total_loss.item(),
            "depth_loss": depth_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "normal_loss": normal_loss.item(),
            "color_loss": color_loss.item(),
            "scale_loss": attach_loss.item(),
        }
        self.train_report(self.get_total_iter, report_losses)
        self.optimizer.zero_grad(set_to_none=True)
        return loss, report_losses

    def evaluate_render_range(
        self, frame, global_opt=False, sample_ratio=-1, unstable=True
    ):
        if unstable:
            render_output = self.renderer.render(
                frame,
                self.unstable_params
            )
        else:
            render_output = self.renderer.render(
                frame,
                self.stable_params
            )
        unstable_T_map = render_output["T_map"]

        if global_opt:
            if sample_ratio > 0:
                render_image = render_output["render"].permute(1, 2, 0)
                gt_image = frame.original_image.permute(1, 2, 0).cuda()
                image_diff = (render_image - gt_image).abs()
                color_error = torch.sum(
                    image_diff, dim=-1, keepdim=False
                )
                filter_mask = (render_image.sum(dim=-1) == 0)
                color_error[filter_mask] = 0
                tile_mask = colorerror2tilemask(color_error, 16, sample_ratio)
                render_mask = (
                    F.interpolate(
                        tile_mask.float().unsqueeze(0).unsqueeze(0),
                        scale_factor=16,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )[: color_error.shape[0], : color_error.shape[1]]
            # after training, real global optimization
            else:
                render_mask = (unstable_T_map != 1).squeeze(0)
                tile_mask = None
        else:
            render_mask = (unstable_T_map != 1).squeeze(0)
            tile_mask = transmission2tilemask(render_mask, 16, 0.5)

        render_ratio = render_mask.sum() / self.get_pixel_num
        return render_mask, tile_mask, render_ratio

    def error_gaussians_remove(self):
        if self.get_stable_num <= 0:
            return
        # check error by backprojection
        check_frame = self.processed_frames[-1]
        check_map = self.processed_map[-1]
        render_output = self.renderer.render(
            check_frame, self.global_params
        )
        # [unstable, stable]
        unstable_points_num = self.get_unstable_num
        stable_points_num = self.get_stable_num

        color = render_output["render"].permute(1, 2, 0)
        depth = render_output["depth"].permute(1, 2, 0)
        normal = render_output["normal"].permute(1, 2, 0)
        depth_index = render_output["depth_index_map"].permute(1, 2, 0)
        color_index = render_output["color_index_map"].permute(1, 2, 0)

        depth_error = torch.abs(check_map["depth_map"] - depth)
        depth_error[(check_map["depth_map"] - depth) < 0] = 0
        image_error = torch.abs(check_map["color_map"] - color)
        color_error = torch.sum(image_error, dim=-1, keepdim=True)

        normal_error = devF(torch.zeros_like(depth_error))
        invalid_mask = (check_map["depth_map"] == 0) | (depth_index == -1)
        invalid_mask = invalid_mask.squeeze()

        depth_error[invalid_mask] = 0
        color_error[check_map["depth_map"] == 0] = 0
        normal_error[invalid_mask] = 0
        H, W = self.frame_map["color_map"].shape[:2]
        P = unstable_points_num + stable_points_num
        (
            gaussian_color_error,
            gaussian_depth_error,
            gaussian_normal_error,
            outlier_count,
        ) = accumulate_gaussian_error(
            H,
            W,
            P,
            color_error,
            depth_error,
            normal_error,
            color_index,
            depth_index,
            self.add_color_thres,
            self.add_depth_thres,
            self.add_normal_thres,
            True,
        )

        color_filter_thres = 2 * self.add_color_thres
        depth_filter_thres = 2 * self.add_depth_thres

        depth_delete_mask = (gaussian_depth_error > depth_filter_thres).squeeze()
        color_release_mask = (gaussian_color_error > color_filter_thres).squeeze()
        if self.verbose:
            print("===== outlier remove =====")
            print(
                "color outlier num: {:d}, depth outlier num: {:d}".format(
                    (color_release_mask).sum(),
                    (depth_delete_mask).sum(),
                ),
            )

        depth_delete_mask_stable = depth_delete_mask[unstable_points_num:, ...]
        color_release_mask_stable = color_release_mask[unstable_points_num:, ...]

        self.stable_pointcloud._depth_error_counter[depth_delete_mask_stable] += 1
        self.stable_pointcloud._color_error_counter[color_release_mask_stable] += 1

        delete_thresh = 10
        depth_delete_mask = (
            self.stable_pointcloud._depth_error_counter >= delete_thresh
        ).squeeze()
        color_release_mask = (
            self.stable_pointcloud._color_error_counter >= delete_thresh
        ).squeeze()
        # move_to_cpu(keyframe)
        # move_to_cpu_map(keymap)
        self.stable_pointcloud.delete(depth_delete_mask)
        self.gaussians_release(color_release_mask[~depth_delete_mask])

    # update all stable gaussians by keyframes render
    def global_optimization(
        self, update_args, select_keyframe_num=-1, is_end=False
    ):
        print("===== global optimize =====")
        if select_keyframe_num == -1:
            self.gaussians_fix(mask=(self.pointcloud.get_confidence > -1))
        print(
            "keyframe num = {:d}, stable gaussian num = {:d}".format(
                self.get_keyframe_num, self.get_stable_num
            )
        )
        if self.get_stable_num == 0:
            return
        l = self.stable_pointcloud.parametrize(update_args)
        if select_keyframe_num != -1:
            l[0]["lr"] = 0
            for i in range(1, len(l)):
                l[i]["lr"] *= 0.1
        else:
            l[0]["lr"] = 0.0000
            l[1]["lr"] *= self.feature_lr_coef
            l[2]["lr"] *= self.feature_lr_coef
            l[4]["lr"] *= self.scaling_lr_coef
            l[5]["lr"] *= self.rotation_lr_coef
        is_final = False
        init_stat = {
            "opacity": self.stable_pointcloud._opacity.detach().clone(),
            "scaling": self.stable_pointcloud._scaling.detach().clone(),
            "xyz": self.stable_pointcloud._xyz.detach().clone(),
            "rotation_raw": self.stable_pointcloud._rotation.detach().clone(),
        }
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        total_iter = int(self.gaussian_update_iter)
        sample_ratio = 0.4
        if select_keyframe_num == -1:
            total_iter = self.get_keyframe_num * self.final_global_iter
            is_final = True
            select_keyframe_num = self.get_keyframe_num
            update_args.depth_weight = 0
            sample_ratio = -1

        # test random kframes
        random_kframes = False
        
        select_keyframe_num = min(select_keyframe_num, self.get_keyframe_num)
        if random_kframes:
            if select_keyframe_num >= self.get_keyframe_num:
                select_kframe_indexs = list(range(0, self.get_keyframe_num))
            else:
                select_kframe_indexs = np.random.choice(np.arange(1, min(select_keyframe_num * 2, self.get_keyframe_num)),
                                                        select_keyframe_num-1,
                                                        replace=False).tolist() + [0]
        else:
            select_kframe_indexs = list(range(select_keyframe_num))
        
        select_kframe_indexs = [i*-1-1 for i in select_kframe_indexs]
            
            
        select_frame = []
        select_map = []
        select_render_mask = []
        select_tile_mask = []
        select_id = []
        # TODO: only sample some pixels of keyframe for global optimization
        for index in select_kframe_indexs:
            move_to_gpu(self.keyframe_list[index])
            move_to_gpu_map(self.keymap_list[index])
            select_frame.append(self.keyframe_list[index])
            select_map.append(self.keymap_list[index])
            render_mask, tile_mask, _ = self.evaluate_render_range(
                self.keyframe_list[index],
                global_opt=True,
                unstable=False,
                sample_ratio=sample_ratio,
            )
            if select_keyframe_num == -1:
                move_to_cpu(self.keyframe_list[index])
                move_to_cpu_map(self.keymap_list[index])
            select_render_mask.append(render_mask)
            select_tile_mask.append(tile_mask)
            select_id.append(self.keyframe_ids[index])
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)

        with tqdm(total=total_iter, desc="global optimization") as pbar:
            for iter in range(total_iter):
                self.iter = iter
                random_index = random.randint(0, select_keyframe_num - 1)
                frame_input = select_frame[random_index]
                image_input = select_map[random_index]
                if select_keyframe_num == -1:
                    move_to_gpu(frame_input)
                    move_to_gpu_map(image_input)
                if not random_kframes and iter > total_iter / 2 and not is_final:
                    random_index = -1
                render_ouput = self.renderer.render(
                    frame_input,
                    self.stable_params,
                    tile_mask=select_tile_mask[random_index],
                )
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    init_stat,
                    update_args,
                    render_mask=select_render_mask[random_index],
                    unstable=False,
                )
                if select_keyframe_num == -1:
                    move_to_cpu(frame_input)
                    move_to_cpu_map(image_input)
                pbar.set_postfix({"loss": "{0:1.5f}".format(loss)})
                pbar.update(1)

        for index in range(-select_keyframe_num, 0):
            move_to_cpu(self.keyframe_list[index])
            move_to_cpu_map(self.keymap_list[index])
        self.stable_pointcloud.detach()

    # Sample some pixels as the init gaussians
    def temp_points_init(self, frame: Camera):
        # print("===== temp points add =====")
        if self.time == 0:
            depth_range_mask = self.frame_map["depth_map"] > 0
            xyz, normal, color = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                self.uniform_sample_num,
                depth_range_mask,
            )
            self.temp_pointcloud.add_empty_points(xyz, normal, color, self.time)
        else:
            self.get_render_output(frame)
            transmission_sample_mask = (
                self.model_map["render_transmission"] > self.add_transmission_thres
            ) & (self.frame_map["depth_map"] > 0)
            transmission_sample_ratio = (
                transmission_sample_mask.sum() / self.get_pixel_num
            )
            transmission_sample_num = devI(
                self.transmission_sample_ratio
                * transmission_sample_ratio
                * self.uniform_sample_num
            )
            if self.verbose:
                print(
                    "transmission empty num = {:d}, sample num = {:d}".format(
                        transmission_sample_mask.sum(), transmission_sample_num
                    )
                )
            xyz_trans, normal_trans, color_trans = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                transmission_sample_num,
                transmission_sample_mask,
            )
            self.temp_pointcloud.add_empty_points(
                xyz_trans, normal_trans, color_trans, self.time
            )

            depth_error = torch.abs(
                self.frame_map["depth_map"] - self.model_map["render_depth"]
            )
            color_error = torch.abs(
                self.frame_map["color_map"] - self.model_map["render_color"]
            ).mean(dim=-1, keepdim=True)

            depth_sample_mask = (
                (depth_error > self.add_depth_thres)
                & (self.frame_map["depth_map"] > 0)
                & (self.model_map["render_depth_index"] > -1)
            )
            color_sample_mask = (
                (color_error > self.add_color_thres)
                & (self.frame_map["depth_map"] > 0)
                & (self.model_map["render_transmission"] < self.add_transmission_thres)
            )
            sample_mask = color_sample_mask | depth_sample_mask
            sample_mask = sample_mask & (~transmission_sample_mask)
            sample_num = devI(sample_mask.sum() * self.error_sample_ratio)
            if self.verbose:
                print(
                    "wrong depth num = {:d}, wrong color num = {:d}, sample num = {:d}".format(
                        depth_sample_mask.sum(),
                        color_sample_mask.sum(),
                        sample_num,
                    )
                )
            xyz_error, normal_error, color_error = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                sample_num,
                sample_mask,
            )
            self.temp_pointcloud.add_empty_points(
                xyz_error, normal_error, color_error, self.time
            )

    # Remove temp points that fall within the existing unstable Gaussian.
    def temp_points_filter(self, topk=3):
        if self.get_unstable_num > 0:
            temp_xyz = self.temp_pointcloud.get_xyz
            if self.verbose:
                print("init {} temp points".format(self.temp_pointcloud.get_points_num))
            exist_xyz = self.unstable_params["xyz"]
            exist_raidus = self.unstable_params["radius"]
            if torch.numel(exist_xyz) > 0:
                inbbox_mask = bbox_filter(temp_xyz, exist_xyz)
                exist_xyz = exist_xyz[inbbox_mask]
                exist_raidus = exist_raidus[inbbox_mask]

            if torch.numel(exist_xyz) == 0:
                return

            nn_dist, nn_indices, _ = knn_points(
                temp_xyz[None, ...],
                exist_xyz[None, ...],
                norm=2,
                K=topk,
                return_nn=True,
            )
            nn_dist = torch.sqrt(nn_dist).squeeze(0)
            nn_indices = nn_indices.squeeze(0)

            corr_radius = exist_raidus[nn_indices] * 0.6
            inside_mask = (nn_dist < corr_radius).any(dim=-1)
            if self.verbose:
                print("delete {} temp points".format(inside_mask.sum().item()))
            self.temp_pointcloud.delete(inside_mask)

    # Attach gaussians fall with in the stable gaussians. attached gaussians is set to low opacity and fix scale
    def temp_points_attach(self, frame: Camera, unstable_opacity_low=0.1):
        if self.get_stable_num == 0:
            return
        # project unstable gaussians and compute uv
        unstable_xyz = self.temp_pointcloud.get_xyz
        origin_indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_opacity = self.temp_pointcloud.get_opacity
        unstable_opacity_filter = (unstable_opacity > unstable_opacity_low).squeeze(-1)
        unstable_xyz = unstable_xyz[unstable_opacity_filter]
        unstable_uv = frame.get_uv(unstable_xyz)
        indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_mask = (
            (unstable_uv[:, 0] >= 0)
            & (unstable_uv[:, 0] < frame.image_width)
            & (unstable_uv[:, 1] >= 0)
            & (unstable_uv[:, 1] < frame.image_height)
        )
        project_uv = unstable_uv[unstable_mask]

        # get the corresponding stable gaussians
        stable_render_output = self.renderer.render(
            frame, self.stable_params,
        )
        
        stable_index = stable_render_output["color_index_map"].permute(1, 2, 0)
        intersect_mask = stable_index[project_uv[:, 1], project_uv[:, 0]] >= 0
        indices = indices[unstable_mask][intersect_mask[:, 0]]

        # compute point to plane distance
        intersect_stable_index = (
            (stable_index[unstable_uv[indices, 1], unstable_uv[indices, 0]])
            .squeeze(-1)
            .long()
        )
        
        stable_normal_check = self.stable_pointcloud.get_normal[intersect_stable_index]
        stable_xyz_check = self.stable_pointcloud.get_xyz[intersect_stable_index]
        unstable_xyz_check = self.temp_pointcloud.get_xyz[indices]
        point_to_plane_distance = (
            (stable_xyz_check - unstable_xyz_check) * stable_normal_check
        ).sum(dim=-1)
        intersect_check = point_to_plane_distance.abs() < 0.5 * self.add_depth_thres
        indices = indices[intersect_check]
        indices = origin_indices[unstable_opacity_filter][indices]

        # set opacity
        self.temp_pointcloud._opacity[indices] = inverse_sigmoid(
            unstable_opacity_low
            * torch.ones_like(self.temp_pointcloud._opacity[indices])
        )
        if self.verbose:
            print("attach {} unstable gaussians".format(indices.shape[0]))

    # Initialize temp points as unstable gaussian.
    def temp_to_optimize(self):
        self.temp_pointcloud.update_geometry(
            self.global_params["xyz"],
            self.global_params["radius"],
        )
        if self.verbose:
            print("===== points add =====")
            print(
                "add new gaussian num: {:d}".format(self.temp_pointcloud.get_points_num)
            )
        remove_mask = devB(torch.ones(self.temp_pointcloud.get_points_num))
        temp_params = self.temp_pointcloud.remove(remove_mask)
        self.pointcloud.cat(temp_params)

    # detect isolated gaussians by KNN
    def gaussians_isolated(self, points, topk=5, threshold=0.005):
        if threshold < 0:
            isolated_mask = devB(torch.zeros(points.shape[0]))
            return isolated_mask
        nn_dist, nn_indices, _ = knn_points(
            points[None, ...],
            points[None, ...],
            norm=2,
            K=topk + 1,
            return_nn=True,
        )
        dist_mean = nn_dist[0, :, 1:].mean(1)
        isolated_mask = dist_mean > threshold
        return isolated_mask

    def create_workspace(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        print(self.save_path)
        os.makedirs(self.save_path, exist_ok=True)
        render_save_path = os.path.join(self.save_path, "eval_render")
        os.makedirs(render_save_path, exist_ok=True)
        model_save_path = os.path.join(self.save_path, "save_model")
        os.makedirs(model_save_path, exist_ok=True)
        traj_save_path = os.path.join(self.save_path, "save_traj")
        os.makedirs(traj_save_path, exist_ok=True)
        traj_save_path = os.path.join(self.save_path, "eval_metric")
        os.makedirs(traj_save_path, exist_ok=True)

        if self.mode == "single process" and self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.save_path)
        else:
            self.tb_writer = None

    def save_model(self, path=None, save_data=True, save_sibr=True, save_merge=True):
        if path == None:
            frame_name = "frame_{:04d}".format(self.time)
            model_save_path = os.path.join(self.save_path, "save_model", frame_name)
            os.makedirs(model_save_path, exist_ok=True)
            path = os.path.join(
                model_save_path,
                "iter_{:04d}".format(self.iter),
            )
        if save_data:
            self.pointcloud.save_model_ply(path + ".ply", include_confidence=True)
            self.stable_pointcloud.save_model_ply(
                path + "_stable.ply", include_confidence=True
            )
        if save_sibr:
            self.pointcloud.save_model_ply(path + "_sibr.ply", include_confidence=False)
            self.stable_pointcloud.save_model_ply(
                path + "_stable_sibr.ply", include_confidence=False
            )
        if self.get_unstable_num > 0 and self.get_stable_num > 0:
            if save_data and save_merge:
                merge_ply(
                    path + ".ply",
                    path + "_stable.ply",
                    path + "_merge.ply",
                    include_confidence=True,
                )
            if save_sibr and save_merge:
                merge_ply(
                    path + "_sibr.ply",
                    path + "_stable_sibr.ply",
                    path + "_merge_sibr.ply",
                    include_confidence=False,
                )

    def train_report(self, iteration, losses):
        if self.tb_writer is not None:
            for loss in losses:
                self.tb_writer.add_scalar(
                    "train/{}".format(loss), losses[loss], iteration
                )

    def eval_report(self, iteration, losses):
        if self.tb_writer is not None:
            for loss in losses:
                self.tb_writer.add_scalar(
                    "eval/{}".format(loss), losses[loss], iteration
                )

    def get_render_output(self, frame):
        render_output = self.renderer.render(
            frame, self.global_params,
        )
        self.model_map["render_color"] = render_output["render"].permute(1, 2, 0)
        self.model_map["render_depth"] = render_output["depth"].permute(1, 2, 0)
        self.model_map["render_normal"] = render_output["normal"].permute(1, 2, 0)
        self.model_map["render_color_index"] = render_output["color_index_map"].permute(
            1, 2, 0
        )
        self.model_map["render_depth_index"] = render_output["depth_index_map"].permute(
            1, 2, 0
        )
        self.model_map["render_transmission"] = render_output["T_map"].permute(1, 2, 0)

    @property
    def stable_params(self):
        have_stable = self.get_stable_num > 0
        xyz = self.stable_pointcloud.get_xyz if have_stable else torch.empty(0)
        opacity = self.stable_pointcloud.get_opacity if have_stable else torch.empty(0)
        scales = self.stable_pointcloud.get_scaling if have_stable else torch.empty(0)
        rotations = (
            self.stable_pointcloud.get_rotation if have_stable else torch.empty(0)
        )
        shs = self.stable_pointcloud.get_features if have_stable else torch.empty(0)
        radius = self.stable_pointcloud.get_radius if have_stable else torch.empty(0)
        normal = self.stable_pointcloud.get_normal if have_stable else torch.empty(0)
        confidence = (
            self.stable_pointcloud.get_confidence if have_stable else torch.empty(0)
        )
        stable_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
        }
        return stable_params

    @property
    def unstable_params(self):
        have_unstable = self.get_unstable_num > 0
        xyz = self.pointcloud.get_xyz if have_unstable else torch.empty(0)
        opacity = self.pointcloud.get_opacity if have_unstable else torch.empty(0)
        scales = self.pointcloud.get_scaling if have_unstable else torch.empty(0)
        rotations = self.pointcloud.get_rotation if have_unstable else torch.empty(0)
        shs = self.pointcloud.get_features if have_unstable else torch.empty(0)
        radius = self.pointcloud.get_radius if have_unstable else torch.empty(0)
        normal = self.pointcloud.get_normal if have_unstable else torch.empty(0)
        confidence = self.pointcloud.get_confidence if have_unstable else torch.empty(0)
        unstable_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
        }
        return unstable_params

    @property
    def global_params_detach(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params
        for k in unstable_params:
            unstable_params[k] = unstable_params[k].detach()
        for k in stable_params:
            stable_params[k] = stable_params[k].detach()

        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"]]
        )
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
        }
        return global_prams

    @property
    def global_params(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params

        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"]]
        )
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
        }
        return global_prams


    @property
    def get_pixel_num(self):
        return (
            self.frame_map["depth_map"].shape[0] * self.frame_map["depth_map"].shape[1]
        )

    @property
    def get_total_iter(self):
        return self.iter + self.time * self.gaussian_update_iter

    @property
    def get_stable_num(self):
        return self.stable_pointcloud.get_points_num

    @property
    def get_unstable_num(self):
        return self.pointcloud.get_points_num

    @property
    def get_total_num(self):
        return self.get_stable_num + self.get_unstable_num

    @property
    def get_curr_frame(self):
        return self.optimize_frames_ids[-1]

    @property
    def get_keyframe_num(self):
        return len(self.keyframe_list)


class MappingProcess(Mapping):
    def __init__(self, map_params, optimization_params, slam):
        super().__init__(map_params)
        self.recorder = Recorder(map_params.device_list[0])
        print("finish init")

        self.slam = slam
        # tracker 2 mapper
        self._tracker2mapper_call = slam._tracker2mapper_call
        self._tracker2mapper_frame_queue = slam._tracker2mapper_frame_queue

        # mapper 2 system
        self._mapper2system_call = slam._mapper2system_call
        self._mapper2system_map_queue = slam._mapper2system_map_queue
        self._mapper2system_tb_queue = slam._mapper2system_tb_queue
        self._mapper2system_requires = slam._mapper2system_requires

        # mapper 2 tracker
        self._mapper2tracker_call = slam._mapper2tracker_call
        self._mapper2tracker_map_queue = slam._mapper2tracker_map_queue

        self._requests = [False, False]  # [frame process, global optimization]
        self._stop = False
        self.input = {}
        self.output = {}
        self.processed_tick = []
        self.time = 0
        self.optimization_params = optimization_params
        self._end = slam._end
        self.max_frame_id = -1

        self.finish = mp.Condition()

    def set_input(self):
        self.frame_map["depth_map"] = self.input["depth_map"]
        self.frame_map["color_map"] = self.input["color_map"]
        self.frame_map["normal_map_c"] = self.input["normal_map_c"]
        self.frame_map["normal_map_w"] = self.input["normal_map_w"]
        self.frame_map["vertex_map_c"] = self.input["vertex_map_c"]
        self.frame_map["vertex_map_w"] = self.input["vertex_map_w"]
        self.frame = self.input["frame"]
        self.time = self.input["time"]
        self.last_send_time = -1

    def send_output(self):
        self.output = {
            "pointcloud": self.pointcloud,
            "stable_pointcloud": self.stable_pointcloud,
            "time": self.time,
            "iter": self.iter,
        }
        print("send output: ", self.time)
        self._mapper2system_map_queue.put(copy.deepcopy(self.output))
        self._mapper2system_requires[1] = True
        with self._mapper2system_call:
            self._mapper2system_call.notify()

    def release_receive(self):
        while (
            not self._tracker2mapper_frame_queue.empty()
            and self._tracker2mapper_frame_queue.qsize() > 1
        ):
            x = self._tracker2mapper_frame_queue.get()
            self.max_frame_id = max(self.max_frame_id, x["time"])
            print("release: ", x["time"])
            if x["time"] == -1:
                self.input = x
            else:
                del x

    def pack_map_to_tracker(self):
        map_info = {
            "frame": copy.deepcopy(self.frame),
            "global_params": self.global_params_detach,
            "frame_id": self.processed_tick[-1],
        }
        print("mapper send map {} to tracker".format(self.processed_tick[-1]))
        with self._mapper2tracker_call:
            self._mapper2tracker_map_queue.put(map_info)
            self._mapper2tracker_call.notify()

    def run(self):
        while True:
            print("map run...")
            with self._tracker2mapper_call:
                if self._tracker2mapper_frame_queue.qsize() == 0:
                    print("waiting tracker to wakeup")
                    self._tracker2mapper_call.wait()
                self.input = self._tracker2mapper_frame_queue.get()
                self.max_frame_id = max(self.max_frame_id, self.input["time"])

            # TODO: debug input is None
            if "time" in self.input and self.input["time"] == -1:
                del self.input
                break
            
            # run frame map update
            self.set_input()
            self.processed_tick.append(self.time)
            self.mapping(self.frame, self.frame_map, self.input["time"], self.optimization_params)
            self.pack_map_to_tracker()


        # self.release_receive()
        self.global_optimization(self.optimization_params)
        self.time = -1
        self.send_output()
        print("processed frames: ", self.optimize_frames_ids)
        print("keyframes: ", self.keyframe_ids)
        self._end[1] = 1
        with self._mapper2system_call:
            self._mapper2system_call.notify()

        with self.finish:
            print("mapper wating finish")
            self.finish.wait()
        print("map finish")

    def stop(self):
        with self.finish:
            self.finish.notify()