import time
import numpy as np
import power.AGXPowerLogger as APL
from task.ds import run_deepsparse  

def intensive_computation(size=1000, iterations=10):
    # Generate two large matrices with random values
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)

    start_time = time.time()

    for _ in range(iterations):
        # Perform matrix multiplication
        result = np.dot(matrix_a, matrix_b)

    end_time = time.time()
    print(f"Completed {iterations} iterations in {end_time - start_time} seconds.")

logger = APL.AGXPowerLogger()
logger.start()
t0 = time.perf_counter()

run_deepsparse("zoo:llama2-7b-gsm8k_llama2_pretrain-base_quantized")

t1 = time.perf_counter()
logger.stop()

latency = t1 - t0
energy = logger.getTotalEnergy()
print("Latency: ", latency)
print("GPU Energy Consumption: ", energy[0])
print("CPU Energy Consumption: ", energy[1])
print("Memory Energy Consumption: ", energy[2])
logger.reset()