from task.detect import detect_yolov8

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import json
import time
import power.AGXPowerLogger as APL
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus

CONFIG_NAME = "YOLOv8-1800"
CPU_CONFIGS = [2201600]
GPU_CONFIGS = [306000000, 408000000, 510000000, 612000000, 714000000, 816000000, 918000000, 1020000000, 1122000000, 1224000000, 1300500000]

if __name__ == "__main__":
    result = {}
    for cpu_config in CPU_CONFIGS:
        for gpu_config in GPU_CONFIGS:
            setCpu(cpu_config)
            setGpu(gpu_config)
            time.sleep(5)
            print(CONFIG_NAME, " CPU: ", getCpuStatus(), " GPU: ", getGpuStatus())
            
            logger = APL.AGXPowerLogger()
            logger.start()
            t0 = time.perf_counter()

            detect_yolov8()

            t1 = time.perf_counter()
            logger.stop()

            latency = t1 - t0
            energy = logger.getTotalEnergy()
            print("Latency: ", latency)
            print("GPU Energy Consumption: ", energy[0])
            print("CPU Energy Consumption: ", energy[1])
            print("Memory Energy Consumption: ", energy[2])
            result[str(cpu_config) + ":" + str(gpu_config)] = (float(latency), float(energy[0]), float(energy[1]), float(energy[2]))
            logger.reset()
    
    with open(CONFIG_NAME + ".json", 'w') as file:
        json.dump(result, file, indent=4)