import json
import numpy as np
from skopt import Optimizer
from skopt.space import Categorical

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

RESULT_PATH = '../assets/result/energy/YOLOv8-1800-960.json'
ENERGY_RESULT = load_json(RESULT_PATH)
ENERGY_SCALE = 10**6
NUM_FRAME = 1800
RESAMPLING_INTERVAL = 15

CPU_FREQ = [
    115200, 192000, 268800, 345600, 422400, 499200, 576000, 652800, 729600,
    806400, 883200, 960000, 1036800, 1113600, 1190400, 1267200, 1344000,
    1420800, 1497600, 1574400, 1651200, 1728000, 1804800, 1881600, 1958400,
    2035200, 2112000, 2188800, 2201600
]

GPU_FREQ = [
    306000000, 408000000, 510000000, 612000000, 714000000, 816000000, 918000000,
    1020000000, 1122000000, 1224000000, 1300500000
]

COMBINED_FPS_PATH = "../assets/result/accuracy/Combined-FPS.json"
REQUIRED_THROUGHPUT = load_json(COMBINED_FPS_PATH)

def evaluate_power(cpu_freq, gpu_freq, result_dict=ENERGY_RESULT):
    key = f"{cpu_freq}:{gpu_freq}"
    energy = result_dict[key][1] + result_dict[key][2]
    time = result_dict[key][0]
    power = energy / time / ENERGY_SCALE
    return power

def evaluate_throughput(cpu_freq, gpu_freq, result_dict=ENERGY_RESULT):
    key = f"{cpu_freq}:{gpu_freq}"
    time = result_dict[key][0]
    throughput = NUM_FRAME / time
    return throughput

# Initialize the optimizer with the parameter space for CPU and GPU frequencies
optimizer = Optimizer(
    dimensions=[
        Categorical(CPU_FREQ), 
        Categorical(GPU_FREQ)
    ],
    base_estimator="GP",  # Gaussian Process for surrogate modeling
    acq_func="EI"  # Expected Improvement acquisition function
)

# Degree of exploration vs. exploitation
initial_exploration = 5  # Number of initial exploration steps
# exploration_weight = 0.2  # Adjust this to control the exploration-exploitation trade-off

# Helper function to combine objectives into a single scalar
def combined_objective(throughput, power, required_throughput):
    if throughput < required_throughput:
        penalty = (required_throughput - throughput) * 100  # Large penalty for not meeting the required throughput
    else:
        penalty = 0
    return power + penalty

# Optimization process
selected_frequencies = []

for i, required_throughput in enumerate(REQUIRED_THROUGHPUT):
    if i < initial_exploration:
        # Initial exploration phase
        cpu_freq = np.random.choice(CPU_FREQ)
        gpu_freq = np.random.choice(GPU_FREQ)
    else:
        # Bayesian optimization phase
        next_point = optimizer.ask()
        cpu_freq = next_point[0]
        gpu_freq = next_point[1]
    
    # Evaluate the chosen frequencies
    power = evaluate_power(cpu_freq, gpu_freq)
    throughput = evaluate_throughput(cpu_freq, gpu_freq)
    
    # Combine objectives into a single scalar
    objective_value = combined_objective(throughput, power, required_throughput)
    
    # Record the selected frequencies
    selected_frequencies.append((cpu_freq, gpu_freq))
    
    # Update the optimizer with the new observation
    optimizer.tell([cpu_freq, gpu_freq], objective_value)

selected_frequencies = [(int(x), int(y)) for x, y in selected_frequencies]
with open("../assets/result/optimizer_energy/Real-Frequency" + ".json", 'w') as file:
    json.dump(selected_frequencies, file, indent=4)

# Evaluation
OPTIMIZER_ENERGY_RESULT = selected_frequencies
EFFICIENT_POWER_RESULT_PATH = '../assets/result/energy/Dict-Per-Real-Second-960.json'
EFFICIENT_POWER_RESULT = load_json(EFFICIENT_POWER_RESULT_PATH)

power_overuse_rate_list = []
output_throughput_list = []
output_throughput_surplus_list = []
output_throughput_shortage_list = []
for i, freq_tuple in enumerate(OPTIMIZER_ENERGY_RESULT):
	key = f"{freq_tuple[0]}:{freq_tuple[1]}"
	output_time = ENERGY_RESULT[key][0]
	output_throughput = round(NUM_FRAME / output_time)
	output_power = (ENERGY_RESULT[key][1] + ENERGY_RESULT[key][2]) / output_time / ENERGY_SCALE

	efficient_data = EFFICIENT_POWER_RESULT[str(output_throughput)][1]
	efficient_time = efficient_data[0]
	efficient_power = (efficient_data[1] + efficient_data[2]) / efficient_time / ENERGY_SCALE

	target_throughput = REQUIRED_THROUGHPUT[i]
	output_throughput_surplus_list.append(max((output_throughput - target_throughput), 0) / (0.5 * (output_throughput + target_throughput)))
	output_throughput_shortage_list.append(max((target_throughput - output_throughput), 0) / (0.5 * (output_throughput + target_throughput)))
	
	output_throughput_list.append(output_throughput)
	power_overuse_rate_list.append( (output_power - efficient_power) / (0.5 * (output_power + efficient_power)) )

time = list(np.array(range(len(REQUIRED_THROUGHPUT))) * RESAMPLING_INTERVAL * 1.)

print(f"Average Power Overuse Rate: {np.average(np.array(power_overuse_rate_list))}")
print(f"Average Throughput Surplus Rate: {np.average(np.array(output_throughput_surplus_list))}")
print(f"Average Throughput Shortage Rate: {np.average(np.array(output_throughput_shortage_list))}")