import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import time
import power.AGXPowerLogger as APL
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus

from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

def run_bert(model_name="camille/bert-base-pruned-voc-esw0.9-40000-en-de-cased", file_name="/home/jiaxi/cs525/Distributed-Learning/data/tinyshakespeare-500.txt"):
    pipe = pipeline("feature-extraction", model=model_name, device="cuda")
    dataset = load_dataset('text', data_files=file_name)

    for out in tqdm(pipe(KeyDataset(dataset['train'], "text"))):
        # print(out)
        pass

CONFIG_NAME = "GPU Frequency"
CONFIGS = [306000000, 510000000, 714000000, 918000000, 1122000000, 1300500000]

if __name__ == "__main__":
    for config in CONFIGS:
        setGpu(config)
        time.sleep(5)
        print(CONFIG_NAME, ": ", getGpuStatus())
        
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
        logger.reset()