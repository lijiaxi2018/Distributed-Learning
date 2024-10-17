from task.detect import detect_yolov8

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import json
import threading
import time
import power.AGXPowerLogger as APL
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus, getEmcStatus
from feature_extraction.edge import detect_edges_in_directory_v2
from feature_extraction.corner import detect_corners_in_directory
from feature_extraction.pixel import calculate_ssim_for_neighbors
from prediction_model.rf.rf_inference import random_forest_inference
from prediction_model.lr.lr_inference import linear_regression_inference
from prediction_model.svm.svm_inference import svm_inference
from prediction_model.clustering.clustering_inference import clustering_inference

def feature_extract():
    # detect_edges_in_directory_v2("/home/jiaxi/cs525/Assets/600_1K")
    # detect_corners_in_directory("/home/jiaxi/cs525/Assets/600_1K", "")
    # calculate_ssim_for_neighbors("/home/jiaxi/cs525/Assets/600_1K", (640, 640))
    # random_forest_inference(600)
    # linear_regression_inference(600)
    svm_inference(600)
    # clustering_inference(600)

CONFIG_NAME = "YOLOv8-1800-640"
ITERATION = 5
CPU_CONFIGS = [
    # 115200, 
    # 192000, 
    # 268800, 
    # 345600, 
    # 422400, 
    # 499200, 
    # 576000, 
    # 652800, 
    # 729600,
    # 806400,
    # 883200, 
    # 960000,
    # 1036800,
    # 1113600, 
    # 1190400,
    # 1267200,
    # 1344000, 
    # 1420800,
    # 1497600,
    # 1574400, 
    # 1651200,
    # 1728000,
    # 1804800, 
    # 1881600,
    # 1958400,
    # 2035200, 
    # 2112000,
    # 2188800,
    # 2201600,
]
GPU_CONFIGS = [
    # 306000000, 
    # 408000000, 
    # 510000000, 
    # 612000000, 
    # 714000000, 
    # 816000000, 
    # 918000000, 
    # 1020000000, 
    # 1122000000, 
    # 1224000000,
    # 1300500000,
]

if __name__ == "__main__":
    result = {}
    for i in range(ITERATION):
        print(CONFIG_NAME, " CPU: ", getCpuStatus(), " GPU: ", getGpuStatus(), " EMC: ", getEmcStatus())
                
        logger = APL.AGXPowerLogger()
        logger.start()
        t0 = time.perf_counter()

        thread = threading.Thread(target=feature_extract)
        thread.start()

        detect_yolov8(source_path="/home/jiaxi/cs525/Assets/600_1K", image_width=640)

        thread.join()

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

        # with open(f'Energy_Log.json', 'w') as file:
        #     json.dump(energy_log, file, indent=4)