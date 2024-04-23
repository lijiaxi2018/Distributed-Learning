import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import json
import time
import power.AGXPowerLogger as APL
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus

from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

def run_bert(model_name="camille/bert-base-pruned-voc-esw0.5-40000-en-de-cased", file_name="/home/jiaxi/cs525/Distributed-Learning/data/tinyshakespeare-10000.txt"):
    pipe = pipeline("feature-extraction", model=model_name, device="cuda")
    dataset = load_dataset('text', data_files=file_name)

    for out in tqdm(pipe(KeyDataset(dataset['train'], "text"))):
        # print(out)
        pass

CONFIG_NAME = "BERT-0.5"
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

            run_bert()

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