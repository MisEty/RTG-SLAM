import copy
import matplotlib.pyplot as plt
from SLAM.gaussian_pointcloud import *

import torch.multiprocessing as mp
from SLAM.render import Renderer
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from SLAM.icp import IcpTracker
from threading import Thread
from utils.camera_utils import loadCam


def convert_poses(trajs):
    poses = []
    stamps = []
    for traj in trajs:
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
        pose_ = np.eye(4)
        pose_[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
        pose_[:3, 3] = np.array([t0, t1, t2])
        poses.append(pose_)
        stamps.append(stamp)
    return poses, stamps


class Tracker(object):
    def __init__(self, args):
        self.use_gt_pose = args.use_gt_pose
        self.mode = args.mode
        self.K = None

        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.depth_filter = args.depth_filter
        self.verbose = args.verbose

        self.icp_tracker = IcpTracker(args)

        self.status = defaultdict(bool)
        self.pose_gt = []
        self.pose_es = []
        self.timestampes = []
        self.finish = mp.Condition()

        self.icp_success_count = 0

        self.use_orb_backend = args.use_orb_backend
        self.orb_vocab_path = args.orb_vocab_path
        self.orb_settings_path = args.orb_settings_path
        self.orb_backend = None
        self.orb_useicp = args.orb_useicp

        self.invalid_confidence_thresh = args.invalid_confidence_thresh

        if self.mode == "single process":
            self.initialize_orb()

    def get_new_poses_byid(self, frame_ids):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses = convert_poses(self.orb_backend.get_trajectory_points())
            frame_poses = [new_poses[frame_id] for frame_id in frame_ids]
        else:
            frame_poses = [self.pose_es[frame_id] for frame_id in frame_ids]
        return frame_poses

    def get_new_poses(self):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses, _ = convert_poses(self.orb_backend.get_trajectory_points())
        else:
            new_poses = None
        return new_poses

    def save_invalid_traing(self, path):
        if np.linalg.norm(self.pose_es[-1][:3, 3] - self.pose_gt[-1][:3, 3]) > 0.15:
            if self.track_mode == "icp":
                frame_id = len(self.pose_es)
                torch.save(
                    self.icp_tracker.vertex_pyramid_t1,
                    os.path.join(path, "vertex_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.vertex_pyramid_t0,
                    os.path.join(path, "vertex_pyramid_t0_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t1,
                    os.path.join(path, "normal_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t0,
                    os.path.join(path, "normal_pyramid_t0_{}.pt".format(frame_id)),
                )

    def map_preprocess(self, frame, frame_id):
        depth_map, color_map = (
            frame.original_depth.permute(1, 2, 0) * 255,
            frame.original_image.permute(1, 2, 0),
        )  # [H, W, C], the image is scaled by 255 in function "PILtoTorch"
        depth_map_orb = (
            frame.original_depth.permute(1, 2, 0).cpu().numpy()
            * 255
            * frame.depth_scale
        ).astype(np.uint16)
        intrinsic = frame.get_intrinsic
        # depth filter
        if self.depth_filter:
            depth_map_filter = bilateralFilter_torch(depth_map, 5, 2, 2)
        else:
            depth_map_filter = depth_map

        valid_range_mask = (depth_map_filter > self.min_depth) & (depth_map_filter < self.max_depth)
        depth_map_filter[~valid_range_mask] = 0.0
        # update depth map
        frame.original_depth = depth_map_filter.permute(2, 0, 1) / 255.0
        # compute geometry info
        vertex_map_c = compute_vertex_map(depth_map_filter, intrinsic)
        normal_map_c = compute_normal_map(vertex_map_c)
        confidence_map = compute_confidence_map(normal_map_c, intrinsic)

        # confidence_threshold tum: 0.5, others: 0.2
        invalid_confidence_mask = ((normal_map_c == 0).all(dim=-1)) | (
            confidence_map < self.invalid_confidence_thresh
        )[..., 0]

        depth_map_filter[invalid_confidence_mask] = 0
        normal_map_c[invalid_confidence_mask] = 0
        vertex_map_c[invalid_confidence_mask] = 0
        confidence_map[invalid_confidence_mask] = 0

        color_map_orb = (
            (frame.original_image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )

        self.update_curr_status(
            frame,
            frame_id,
            depth_map,
            depth_map_filter,
            vertex_map_c,
            normal_map_c,
            color_map,
            color_map_orb,
            depth_map_orb,
            intrinsic,
        )

        frame_map = {}
        frame_map["depth_map"] = depth_map_filter
        frame_map["color_map"] = color_map
        frame_map["normal_map_c"] = normal_map_c
        frame_map["vertex_map_c"] = vertex_map_c
        frame_map["confidence_map"] = confidence_map
        frame_map["invalid_confidence_mask"] = invalid_confidence_mask
        frame_map["time"] = frame_id

        return frame_map

    def update_curr_status(
        self,
        frame,
        frame_id,
        depth_t1,
        depth_t1_filter,
        vertex_t1,
        normal_t1,
        color_t1,
        color_orb,
        depth_orb,
        K,
    ):
        if self.K is None:
            self.K = K
        self.curr_frame = {
            "K": frame.get_intrinsic,
            "normal_map": normal_t1,
            "depth_map": depth_t1,
            "depth_map_filter": depth_t1_filter,
            "vertex_map": vertex_t1,
            "frame_id": frame_id,
            "pose_gt": frame.get_c2w.cpu().numpy(), # 1
            "color_map": color_t1,
            "timestamp": frame.timestamp, # 1
            "color_map_orb": color_orb, # 1
            "depth_map_orb": depth_orb, # 1
        }
        self.icp_tracker.update_curr_status(depth_t1_filter, self.K)
        
    def update_last_status_v2(
        self, frame, render_depth, frame_depth, render_normal, frame_normal
    ):
        intrinsic = frame.get_intrinsic
        normal_mask = (
            1 - F.cosine_similarity(render_normal, frame_normal, dim=-1)
        ) < self.icp_sample_normal_threshold
        depth_filling_mask = (
            (
                torch.abs(render_depth - frame_depth)
                > self.icp_sample_distance_threshold
            )[..., 0]
            | (render_depth == 0)[..., 0]
            | (normal_mask)
        ) & (frame_depth > 0)[..., 0]

        render_depth[depth_filling_mask] = frame_depth[depth_filling_mask]
        render_depth[(frame_depth == 0)[..., 0]] = 0
        
        self.last_model_vertex = compute_vertex_map(render_depth, intrinsic)
        self.last_model_normal = compute_normal_map(self.last_model_vertex)

    def update_last_status(
        self,
        frame,
        render_depth,
        frame_depth,
        render_normal,
        frame_normal,
    ):
        self.icp_tracker.update_last_status(
            frame, render_depth, frame_depth, render_normal, frame_normal
        )

    def refine_icp_pose(self, pose_t1_t0, tracking_success):
        if tracking_success and self.orb_useicp:
            print("success")
            self.orb_backend.track_with_icp_pose(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                pose_t1_t0.astype(np.float32),
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        else:
            self.orb_backend.track_with_orb_feature(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        traj_history = self.orb_backend.get_trajectory_points()
        pose_es_t1, _ = convert_poses(traj_history[-2:])
        return pose_es_t1[-1]

    def initialize_orb(self):
        if not self.use_gt_pose and self.use_orb_backend and self.orb_backend is None:
            import orbslam2
            print("init orb backend")
            self.orb_backend = orbslam2.System(
                self.orb_vocab_path, self.orb_settings_path, orbslam2.Sensor.RGBD
            )
            self.orb_backend.set_use_viewer(False)
            self.orb_backend.initialize(self.orb_useicp)

    def initialize_tracker(self):
        if self.use_orb_backend:
            self.orb_backend.process_image_rgbd(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                self.curr_frame["timestamp"],
            )
        self.status["initialized"] = True

    def tracking(self, frame, frame_map):
        self.pose_gt.append(self.curr_frame["pose_gt"])
        self.timestampes.append(self.curr_frame["timestamp"])
        p2loss = 0
        tracking_success = True
        if self.use_gt_pose:
            pose_t1_w = self.pose_gt[-1]
        else:
            # initialize
            if not self.status["initialized"]:
                self.initialize_tracker()
                pose_t1_w = np.eye(4)
            else:
                pose_t1_t0, tracking_success = self.icp_tracker.predict_pose(self.curr_frame)
                if self.use_orb_backend:
                    pose_t1_w = self.refine_icp_pose(pose_t1_t0, tracking_success)
                else:
                    pose_t1_w = self.pose_es[-1] @ pose_t1_t0

        self.icp_tracker.move_last_status()
        self.pose_es.append(pose_t1_w)

        frame.updatePose(pose_t1_w)
        frame_map["vertex_map_w"] = transform_map(
            frame_map["vertex_map_c"], frame.get_c2w
        )
        frame_map["normal_map_w"] = transform_map(
            frame_map["normal_map_c"], get_rot(frame.get_c2w)
        )

        return tracking_success

    def eval_total_ate(self, pose_es, pose_gt):
        ates = []
        for i in tqdm(range(1, len(pose_gt) + 1)):
            ates.append(self.eval_ate(pose_es, pose_gt, i))
        ates = np.array(ates)
        return ates

    def save_ate_fig(self, ates, save_path, save_name):
        plt.plot(range(len(ates)), ates)
        plt.ylim(0, max(ates) + 0.1)
        plt.title("ate:{}".format(ates[-1]))
        plt.savefig(os.path.join(save_path, "{}.png".format(save_name)))
    

    def save_keyframe_traj(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_keyframe_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    def save_traj_tum(self, save_file):
        poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
        with open(save_file, "w") as f:
            for pose_id, pose_es_ in enumerate(self.pose_es):
                t = pose_es_[:3, 3]
                q = R.from_matrix(pose_es_[:3, :3])
                f.write(str(stamps[pose_id]) + " ")
                for i in t.tolist():
                    f.write(str(i) + " ")
                for i in q.as_quat().tolist():
                    f.write(str(i) + " ")
                f.write("\n")

    def save_orb_traj_tum(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    def save_traj(self, save_path):
        save_traj_path = os.path.join(save_path, "save_traj")
        if not self.use_gt_pose and self.use_orb_backend:
            traj_history = self.orb_backend.get_trajectory_points()
            self.pose_es, _ = convert_poses(traj_history)
        pose_es = np.stack(self.pose_es, axis=0)
        pose_gt = np.stack(self.pose_gt, axis=0)
        ates_ba = self.eval_total_ate(pose_es, pose_gt)
        print("ate: ", ates_ba[-1])
        np.save(os.path.join(save_traj_path, "pose_gt.npy"), pose_gt)
        np.save(os.path.join(save_traj_path, "pose_es.npy"), pose_es)
        self.save_ate_fig(ates_ba, save_traj_path, "ate")

        plt.figure()
        plt.plot(pose_es[:, 0, 3], pose_es[:, 1, 3])
        plt.plot(pose_gt[:, 0, 3], pose_gt[:, 1, 3])
        plt.legend(["es", "gt"])
        plt.savefig(os.path.join(save_traj_path, "traj_xy.jpg"))
        
        if self.use_orb_backend:
            self.orb_backend.shutdown()
        
    def eval_ate(self, pose_es, pose_gt, frame_id=-1):
        pose_es = np.stack(pose_es, axis=0)[:frame_id, :3, 3]
        pose_gt = np.stack(pose_gt, axis=0)[:frame_id, :3, 3]
        ate = eval_ate(pose_gt, pose_es)
        return ate


class TrackingProcess(Tracker):
    def __init__(self, slam, args):
        args.icp_use_model_depth = False
        super().__init__(args)

        self.args = args
        # online scanner
        self.use_online_scanner = args.use_online_scanner
        self.scanner_finish = False

        # sync mode
        self.sync_tracker2mapper_method = slam.sync_tracker2mapper_method
        self.sync_tracker2mapper_frames = slam.sync_tracker2mapper_frames

        # tracker2mapper
        self._tracker2mapper_call = slam._tracker2mapper_call
        self._tracker2mapper_frame_queue = slam._tracker2mapper_frame_queue

        self.mapper_running = True

        # mapper2tracker
        self._mapper2tracker_call = slam._mapper2tracker_call
        self._mapper2tracker_map_queue = slam._mapper2tracker_map_queue

        self.dataset_cameras = slam.dataset.scene_info.train_cameras
        self.map_process = slam.map_process
        self._end = slam._end
        self.max_fps = args.tracker_max_fps
        self.frame_time = 1.0 / self.max_fps
        self.frame_id = 0
        self.last_mapper_frame_id = 0

        self.last_frame = None
        self.last_global_params = None

        self.track_renderer = Renderer(args)
        self.save_path = args.save_path

    def map_preprocess_mp(self, frame, frame_id):
        self.map_input = super().map_preprocess(frame, frame_id)

    def send_frame_to_mapper(self):
        print("tracker send frame {} to mapper".format(self.map_input["time"]))
        self._tracker2mapper_call.acquire()
        self._tracker2mapper_frame_queue.put(self.map_input)
        self.map_process._requests[0] = True
        self._tracker2mapper_call.notify()
        self._tracker2mapper_call.release()

    def finish_(self):
        if self.use_online_scanner:
            return self.scanner_finish
        else:
            return self.frame_id >= len(self.dataset_cameras)

    def getNextFrame(self):
        frame_info = self.dataset_cameras[self.frame_id]
        frame = loadCam(self.args, self.frame_id, frame_info, 1)
        print("get frame: {}".format(self.frame_id))
        self.frame_id += 1
        return frame


    def run(self):
        self.time = 0
        self.initialize_orb()

        while not self.finish_():
            frame = self.getNextFrame()
            if frame is None:
                break
            frame_id = frame.uid
            print("current tracker frame = %d" % self.time)
            # update current map
            move_to_gpu(frame)

            self.map_preprocess_mp(frame, frame_id)
            self.tracking(frame, self.map_input)
            self.map_input["frame"] = copy.deepcopy(frame)
            self.map_input["frame"] = frame

            self.map_input["poses_new"] = self.get_new_poses()
            # send message to mapper

            self.send_frame_to_mapper()

            wait_begin = time.time()
            if not self.finish_() and self.mapper_running:
                if self.sync_tracker2mapper_method == "strict":
                    if (frame_id + 1) % self.sync_tracker2mapper_frames == 0:
                        with self._mapper2tracker_call:
                            print("wait mapper to wakeup")
                            print(
                                "tracker buffer size: {}".format(
                                    self._tracker2mapper_frame_queue.qsize()
                                )
                            )
                            self._mapper2tracker_call.wait()
                elif self.sync_tracker2mapper_method == "loose":
                    if (
                        frame_id - self.last_mapper_frame_id
                    ) > self.sync_tracker2mapper_frames:
                        with self._mapper2tracker_call:
                            print("wait mapper to wakeup")
                            self._mapper2tracker_call.wait()
                else:
                    pass
            wait_end = time.time()

            self.unpack_map_to_tracker()
            self.update_last_mapper_render(frame)
            self.update_viewer(frame)

            move_to_cpu(frame)

            self.time += 1
        # send a invalid time stamp as end signal
        self.map_input["time"] = -1
        self.send_frame_to_mapper()
        self.save_traj(self.save_path)
        self._end[0] = 1
        with self.finish:
            print("tracker wating finish")
            self.finish.wait()
        print("track finish")

    def stop(self):
        with self.finish:
            self.finish.notify()

    def unpack_map_to_tracker(self):
        self._mapper2tracker_call.acquire()
        while not self._mapper2tracker_map_queue.empty():
            map_info = self._mapper2tracker_map_queue.get()
            self.last_frame = map_info["frame"]
            self.last_global_params = map_info["global_params"]
            self.last_mapper_frame_id = map_info["frame_id"]
            print("tracker unpack map {}".format(self.last_mapper_frame_id))
        self._mapper2tracker_call.notify()
        self._mapper2tracker_call.release()

    def update_last_mapper_render(self, frame):
        pose_t0_w = frame.get_c2w.cpu().numpy()
        if self.last_frame is not None:
            pose_w_t0 = np.linalg.inv(pose_t0_w)
            self.last_frame.update(pose_w_t0[:3, :3].transpose(), pose_w_t0[:3, 3])
            render_output = self.track_renderer.render(
                self.last_frame,
                self.last_global_params,
                None
            )
            self.update_last_status(
                self.last_frame,
                render_output["depth"].permute(1, 2, 0),
                self.map_input["depth_map"],
                render_output["normal"].permute(1, 2, 0),
                self.map_input["normal_map_w"],
            )
