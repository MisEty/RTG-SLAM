import GPUtil
import time
import numpy as np
import os
import json
from os.path import join, exists


class Recorder(object):
    def __init__(self, gpu_id) -> None:
        self._gpu_id = gpu_id
        self._value = {}
        self._counter = {}

    def update_max(self, name, value):
        if name not in self._value:
            self._value[name] = value
            self._counter[name] = 1
        else:
            self._value[name] = max(self._value[name], value)

    def cal_fps(self):
        self._value["fps"] = 1 / self._value["mapping"]
        self._counter["fps"] = 1

    def update_mean(self, name, value, count):
        if count == 0:
            return
        value_mean = value / count
        if name not in self._value:
            self._value[name] = value_mean
            self._counter[name] = count
        else:
            self._value[name] = (self._value[name] * self._counter[name] + value) / (
                self._counter[name] + count
            )
            self._counter[name] = self._counter[name] + count

    def watch_gpu(self):
        # current gpu info
        gpu = GPUtil.getGPUs()[self._gpu_id]
        memory_used = gpu.memoryUsed
        self.update_max("gpu_memory", memory_used)
        return memory_used / 1024.0

    def save(self, dir):
        if not exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, "performance.json"), "w") as f:
            json.dump(self._value, f)

    def display(self):
        print(self._gpu_info)


if __name__ == "__main__":
    monitor = Recorder(0)
    for i in range(5):
        monitor.watch_gpu(i)
        time.sleep(0.5)
        monitor.update_mean("money", 100, 5)
        monitor.update_mean("money", 28, 3)
    monitor.save("./temp")
