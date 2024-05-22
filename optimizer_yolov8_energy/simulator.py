import json
import numpy as np
from DVFSController import DVFSController 

def interpolate_2d_array(arr, new_shape):
    """
    Reshape a 2D array to a specific shape, retaining the edge values and 
    maintaining the trend for non-edge values.
    
    :param arr: 2D list of numbers
    :param new_shape: tuple of (new_rows, new_cols)
    :return: reshaped 2D list
    """
    arr = np.array(arr)
    orig_rows, orig_cols = arr.shape
    new_rows, new_cols = new_shape

    # Step 1: Interpolate rows
    intermediate_array = np.zeros((orig_rows, new_cols))
    for i in range(orig_rows):
            intermediate_array[i, :] = np.interp(np.linspace(0, orig_cols-1, new_cols), np.arange(orig_cols), arr[i, :])

    # Step 2: Interpolate columns
    new_array = np.zeros((new_rows, new_cols))
    for j in range(new_cols):
            new_array[:, j] = np.interp(np.linspace(0, orig_rows-1, new_rows), np.arange(orig_rows), intermediate_array[:, j])

    # Ensure edge values are exactly as in the original array
    new_array[0, :] = np.interp(np.linspace(0, orig_cols-1, new_cols), np.arange(orig_cols), arr[0, :])
    new_array[-1, :] = np.interp(np.linspace(0, orig_cols-1, new_cols), np.arange(orig_cols), arr[-1, :])
    new_array[:, 0] = np.interp(np.linspace(0, orig_rows-1, new_rows), np.arange(orig_rows), arr[:, 0])
    new_array[:, -1] = np.interp(np.linspace(0, orig_rows-1, new_rows), np.arange(orig_rows), arr[:, -1])

    return new_array.tolist()

def evaluate_controller(selected_frequencies, energy_result, efficient_power_result, num_frame):
    power_overuse_rate_list = []
    output_throughput_list = []
    output_throughput_surplus_list = []
    output_throughput_shortage_list = []
    output_throughput_diff_list = []

    for i, freq_tuple in enumerate(selected_frequencies):
        key = f"{freq_tuple[0]}:{freq_tuple[1]}"
        output_time = energy_result[key][0]
        output_throughput = round(num_frame / output_time)
        output_power = (energy_result[key][1] + energy_result[key][2]) / output_time / ENERGY_SCALE

        efficient_data = efficient_power_result[str(output_throughput)][1]
        efficient_time = efficient_data[0]
        efficient_power = (efficient_data[1] + efficient_data[2]) / efficient_time / ENERGY_SCALE

        target_throughput = REQUIRED_THROUGHPUT[i]
        output_throughput_surplus_list.append(max((output_throughput - target_throughput), 0) / (0.5 * (output_throughput + target_throughput)))
        output_throughput_shortage_list.append(max((target_throughput - output_throughput), 0) / (0.5 * (output_throughput + target_throughput)))
        output_throughput_diff_list.append((output_throughput - target_throughput) / (0.5 * (output_throughput + target_throughput)))
        
        output_throughput_list.append(output_throughput)
        power_overuse_rate_list.append( (output_power - efficient_power) / (0.5 * (output_power + efficient_power)) )

    time = list(np.array(range(len(REQUIRED_THROUGHPUT))) * RESAMPLING_INTERVAL * 1.)

    return time, power_overuse_rate_list, output_throughput_list, output_throughput_surplus_list, output_throughput_shortage_list, output_throughput_diff_list

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

ENERGY_SCALE = 10**6
NUM_FRAME = 1800
RESAMPLING_INTERVAL = 15

RESULT_PATH = '../assets/result/energy/YOLOv8-1800-960.json'
ENERGY_RESULT = load_json(RESULT_PATH)

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

STANDARD_EFFICIENCY_TABLE_PATH = '../assets/result/optimizer_energy/Standard-Efficiency-Table.json'
STANDARD_EFFICIENCY_TABLE = load_json(STANDARD_EFFICIENCY_TABLE_PATH)
real_efficiency_table = interpolate_2d_array(STANDARD_EFFICIENCY_TABLE, (len(GPU_FREQ), len(CPU_FREQ)))

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

# Simulation
if __name__ == "__main__":
    controller = DVFSController(w=0.375, GPU_FREQ=GPU_FREQ, CPU_FREQ=CPU_FREQ, efficiency_array=real_efficiency_table)

    selected_frequencies = []
    for required_throughput in REQUIRED_THROUGHPUT:
        cpu_freq, gpu_freq = controller.suggest_combination(required_throughput)
        selected_frequencies.append((cpu_freq, gpu_freq))
        real_throughput = evaluate_throughput(cpu_freq, gpu_freq)
        controller.update_model(cpu_freq, gpu_freq, real_throughput)

    suggestion_latency = controller.get_suggestion_latency()
    update_latency = controller.get_update_latency()
        
    # Evaluation
    EFFICIENT_POWER_RESULT_PATH = '../assets/result/energy/Dict-Per-Real-Second-960.json'
    EFFICIENT_POWER_RESULT = load_json(EFFICIENT_POWER_RESULT_PATH)

    _, power_overuse_rate_list, _, output_throughput_surplus_list, output_throughput_shortage_list, _ = evaluate_controller(selected_frequencies, ENERGY_RESULT, EFFICIENT_POWER_RESULT, NUM_FRAME)

    print(f"Average Power Overuse Rate: {np.average(np.array(power_overuse_rate_list))}")
    print(f"Average Throughput Surplus Rate: {np.average(np.array(output_throughput_surplus_list))}")
    print(f"Average Throughput Shortage Rate: {np.average(np.array(output_throughput_shortage_list))}")
    print(f"Average Suggestion Latency: {np.average(np.array(suggestion_latency))}")
    print(f"Average Update Latency: {np.average(np.array(update_latency))}")

    # Output
    selected_frequencies = [(int(x), int(y)) for x, y in selected_frequencies]

    result = {}
    result['frequencies'] = selected_frequencies
    result['suggestion_latency'] = suggestion_latency
    result['update_latency'] = update_latency
    with open("../assets/result/optimizer_energy/DVFSController-Simulator-Results" + ".json", 'w') as file:
        json.dump(result, file, indent=4)