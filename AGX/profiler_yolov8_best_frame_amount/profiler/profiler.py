from task.detect import detect_yolov8

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import json
import time
import power.AGXPowerLogger as APL
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus, getEmcStatus

CONFIG_NAME = "YOLOv8-BFA-640-2201600-1300500000"
ITERATION = 12
NUM_FRAMES = [120, 240, 360, 600, 900, 1200, 1800, 3600]

if __name__ == "__main__":
    result = {}
    for n in NUM_FRAMES:
        result[n] = []
        for i in range(ITERATION):
            print(f"Frame Amount: {n}; Iteration: {i}")
            print(CONFIG_NAME, " CPU: ", getCpuStatus(), " GPU: ", getGpuStatus(), " EMC: ", getEmcStatus())
                    
            logger = APL.AGXPowerLogger()
            logger.start()
            t0 = time.perf_counter()

            detect_yolov8(source_path=f"/home/jiaxi/cs525/Assets/{n}_1K", image_width=640)

            t1 = time.perf_counter()
            logger.stop()

            energy_log = logger.getDataLog()

            latency = t1 - t0
            energy = logger.getTotalEnergy()
            print("Latency: ", latency)
            print("GPU Energy Consumption: ", energy[0])
            print("CPU Energy Consumption: ", energy[1])
            print("Memory Energy Consumption: ", energy[2])
            logger.reset()

            result[n].append((latency, energy[0], energy[1], energy[2]))

    with open(f'{CONFIG_NAME}.json', 'w') as file:
        json.dump(result, file, indent=4)