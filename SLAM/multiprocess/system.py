import torch
import time
import torch.multiprocessing as mp
import os
from SLAM.multiprocess.mapper import MappingProcess
from SLAM.multiprocess.tracker import TrackingProcess
from SLAM.utils import merge_ply

sleep_time = 0.01


class SLAM(object):
    def __init__(self, map_params, optimization_params, dataset, args) -> None:
        
        self.verbose = True
        self._end = torch.zeros((2)).int().share_memory_()
        self.dataset = dataset
        
        # strict: mapping : tracker == 1 : sync_tracker2mapper_frames
        # loose: tracker frame_id should be: [mapper_frame_id - sync_tracker2mapper_frames, 
        #                                     mapper_frame_id + sync_tracker2mapper_frames]
        # free: there is no sync
        self.sync_tracker2mapper_method = map_params.sync_tracker2mapper_method
        self.sync_tracker2mapper_frames = map_params.sync_tracker2mapper_frames

        # tracker 2 mapper
        self._tracker2mapper_call = mp.Condition()
        self._tracker2mapper_frame_queue = mp.Queue()
        
        # mapper 2 tracker
        self._mapper2tracker_call = mp.Condition()
        self._mapper2tracker_map_queue = mp.Queue()
        
        # mapper 2 system
        self._mapper2system_call = mp.Condition()
        self._mapper2system_requires = [False, False]  # tb call, save_model call
        self._mapper2system_map_queue = mp.Queue()
        self._mapper2system_tb_queue = mp.Queue()
        
        self.map_process = MappingProcess(args, optimization_params, 
                                          self)
        self.track_process = TrackingProcess(self, args)
        self.save_path = self.map_process.save_path
        
    
    def run(self):
        processes = []
        for rank in range(2):
            if rank == 0:
                print("start mapping process")
                p = mp.Process(target=self.mapping, args=(rank, ))
            elif rank == 1:
                print("start tracking process")
                p = mp.Process(target=self.tracking, args=(rank,))
            p.start()
            processes.append(p)
        while self._end.sum() != 2:
            # process save model task
            with self._mapper2system_call:
                if self._mapper2system_requires.count(True) == 0:
                    self._mapper2system_call.wait()
                
                if self._mapper2system_requires[0]:
                    self._mapper2system_requires[0] = False
                
                # save model
                if self._mapper2system_requires[1]:
                    while not self._mapper2system_map_queue.empty():
                        map_output = self._mapper2system_map_queue.get()
                        self.save_model(map_output)
                        del map_output
                        break
                    self._mapper2system_requires[1] = False
                if self._end[1] == 1:
                    break
        print("system finish")
        while not self._mapper2system_map_queue.empty():
            print("delete model")
            x = self._mapper2system_map_queue.get()
            self.save_model(x)
            del x
        self.release()
        self.track_process.stop()
        self.map_process.stop()
        print("main finish")
        for p in processes:
            p.join()
    
    def tracking(self, rank):
        print("start traking")
        self.track_process.run()

    def mapping(self, rank):
        print("start mapping")
        self.map_process.run()

    def release_mp_queue(self, mp_queue):
        while not mp_queue.empty():
            x = mp_queue.get()
            del x

    def release(self):
        self.release_mp_queue(self._tracker2mapper_frame_queue)
        self.release_mp_queue(self._mapper2system_map_queue)
        self.release_mp_queue(self._mapper2system_tb_queue)
        self.release_mp_queue(self._mapper2tracker_map_queue)
        
    def save_model(self, map_output, save_data=True, save_sibr=True, save_merge=False):
        print("save model")
        self.pointcloud = map_output["pointcloud"]
        self.stable_pointcloud = map_output["stable_pointcloud"]
        self.map_time = map_output["time"]
        self.map_iter = map_output["iter"]
        print("save model:", self.map_time)
        frame_name = "frame_{:04d}".format(self.map_time)
        frame_save_path = os.path.join(self.save_path, "save_model", frame_name)
        os.makedirs(frame_save_path, exist_ok=True)
        path = os.path.join(
            frame_save_path,
            "iter_{:04d}".format(self.map_iter),
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
        print("save finish")