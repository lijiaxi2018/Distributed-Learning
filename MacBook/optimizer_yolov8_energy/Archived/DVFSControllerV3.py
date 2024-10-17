import numpy as np
import random
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

class DVFSControllerV3:
    def __init__(self, efficiency_desc_rate, exploration_desc_rate, suggest_randomness, GPU_FREQ, CPU_FREQ, efficiency_array):
        self.efficiency_desc_rate = efficiency_desc_rate
        self.exploration_desc_rate = exploration_desc_rate
        self.suggest_randomness = suggest_randomness
        self.GPU_FREQ = GPU_FREQ
        self.CPU_FREQ = CPU_FREQ
        self.efficiency_array = np.array(efficiency_array)
        self.observed_data = {'frequency': [], 'throughput': [], 'power': []}
        self.suggestion_latency = []
        self.update_latency = []

        self.max_throughput = 0
        self.min_throughput = 0

        self.visited = [[False for _ in range(len(CPU_FREQ))] for _ in range(len(GPU_FREQ))]
        self.throughput_min_power = {}

        self.idx = 0
        
        # Initialize Gaussian Process Regressor
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
        self.throughput_gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
        self.power_gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
        
        # Standard Scaler for input normalization
        self.scaler = StandardScaler()
    
    def get_update_latency(self):
        return self.update_latency
    
    def update_model(self, cpu_freq, gpu_freq, real_throughput, real_power):
        start_time = time.time()
        self.update_model_helper(cpu_freq, gpu_freq, real_throughput, real_power)
        end_time = time.time()

        self.update_latency.append(end_time - start_time)

    def update_model_helper(self, cpu_freq, gpu_freq, real_throughput, real_power):
        if len(self.observed_data['frequency']) == 0:
            self.max_throughput = round(real_throughput)
            self.throughput_min_power[self.max_throughput] = ((cpu_freq, gpu_freq), real_power)

        if len(self.observed_data['frequency']) == 1:
            self.min_throughput = round(real_throughput)
            self.throughput_min_power[self.min_throughput] = ((cpu_freq, gpu_freq), real_power)

            ((max_cpu_freq, max_gpu_freq), max_real_power) = self.throughput_min_power[self.max_throughput]
            for t in range(self.min_throughput + 1, self.max_throughput):
                self.throughput_min_power[t] = ((max_cpu_freq, max_gpu_freq), max_real_power)

        if self.visited[self.GPU_FREQ.index(gpu_freq)][self.CPU_FREQ.index(cpu_freq)] == False:
            self.visited[self.GPU_FREQ.index(gpu_freq)][self.CPU_FREQ.index(cpu_freq)] = True
            
            rounded_throughput = round(real_throughput)
            for t in range(rounded_throughput, self.min_throughput - 1, -1):
                if t not in self.throughput_min_power or self.throughput_min_power[t][1] > real_power:
                    self.throughput_min_power[t] = ((cpu_freq, gpu_freq), real_power)
                else:
                    break

            self.observed_data['frequency'].append([cpu_freq, gpu_freq])
            self.observed_data['throughput'].append(real_throughput)
            self.observed_data['power'].append(real_power)
            
            if len(self.observed_data['frequency']) > 1:
                X = np.array(self.observed_data['frequency'])
                y_throughput = np.array(self.observed_data['throughput'])
                y_power = np.array(self.observed_data['power'])
                X_scaled = self.scaler.fit_transform(X)
                self.throughput_gpr.fit(X_scaled, y_throughput)
                self.power_gpr.fit(X_scaled, y_power)
    
    def get_suggestion_latency(self):
        return self.suggestion_latency
    
    def suggest_combination(self, required_throughput):
        start_time = time.time()
        suggestion_frequency = self.suggest_combination_helper(required_throughput)
        end_time = time.time()

        self.suggestion_latency.append(end_time - start_time)
        self.idx += 1
        return suggestion_frequency
    
    def suggest_combination_helper(self, required_throughput):
        if len(self.observed_data['frequency']) == 0:
            return self.CPU_FREQ[-1], self.GPU_FREQ[-1]
        
        if len(self.observed_data['frequency']) == 1:
            return self.CPU_FREQ[0], self.GPU_FREQ[0]
        
        if required_throughput >= self.max_throughput:
            return self.CPU_FREQ[-1], self.GPU_FREQ[-1]
        
        if required_throughput <= self.min_throughput:
            return self.CPU_FREQ[0], self.GPU_FREQ[0]

        if required_throughput not in self.throughput_min_power:
            return self.suggest_by_gpr(required_throughput)
        else:
            exploration_boundary = max(1. - self.idx * self.exploration_desc_rate, 0.2)
            if random.random() > exploration_boundary:
                ((cpu_freq_pq, gpu_freq_pq), _) = self.throughput_min_power[required_throughput]
                return cpu_freq_pq, gpu_freq_pq
            else:
                cpu_freq_gpr, gpu_freq_gpr = self.suggest_by_gpr(required_throughput)
                if self.visited[self.GPU_FREQ.index(gpu_freq_gpr)][self.CPU_FREQ.index(cpu_freq_gpr)] == True:
                    ((cpu_freq_pq, gpu_freq_pq), _) = self.throughput_min_power[required_throughput]
                    return cpu_freq_pq, gpu_freq_pq
                else:
                    return cpu_freq_gpr, gpu_freq_gpr

    def suggest_by_gpr(self, required_throughput):
        search_space = np.array([[cf, gf] for cf in self.CPU_FREQ for gf in self.GPU_FREQ])
        search_space_scaled = self.scaler.transform(search_space)
        predicted_throughputs, _ = self.throughput_gpr.predict(search_space_scaled, return_std=True)
        predicted_powers, _ = self.power_gpr.predict(search_space_scaled, return_std=True)
                
        valid_combinations = search_space[predicted_throughputs >= required_throughput]

        # If no valid combination, return the most efficient combination
        if len(valid_combinations) == 0:
            max_efficiency_index = np.unravel_index(np.argmax(self.efficiency_array, axis=None), self.efficiency_array.shape)
            return self.CPU_FREQ[max_efficiency_index[1]], self.GPU_FREQ[max_efficiency_index[0]]

        # Normalize predicted powers between 0 and 1
        min_power = np.min(predicted_powers)
        max_power = np.max(predicted_powers)
        normalized_powers = (predicted_powers - min_power) / (max_power - min_power)

        # Calculate efficiency, distance, and power score for each valid combination
        efficiencies = [self.efficiency_array[self.GPU_FREQ.index(gf), self.CPU_FREQ.index(cf)] for cf, gf in valid_combinations]
        distances = [min((predicted_throughput - required_throughput) / required_throughput, 1) for predicted_throughput in predicted_throughputs[predicted_throughputs >= required_throughput]]
        distances = [abs(d) for d in distances]

        power_weight = 1.
        efficiency_weight = max(1. - self.idx * self.efficiency_desc_rate, 0.3)
        scores = [
            efficiency_weight * eff + (1 - efficiency_weight) * ((1-power_weight) * (1 - dist) + power_weight * (1 - norm_power))
            for eff, dist, norm_power in zip(efficiencies, distances, normalized_powers)
        ]
        
        # Get the indices of the top 5 scores
        top_indices = np.argsort(scores)[-self.suggest_randomness:]

        # Select a random index among the top 5
        best_combination_index = random.choice(top_indices)

        best_combination = valid_combinations[best_combination_index]
        return best_combination[0], best_combination[1]
    
    def print_throughput_dict(self):
        # Print the elements in throughput_min_power with ascending throughput values
        for throughput in sorted(self.throughput_min_power):
            print(f"Throughput: {throughput} -> {self.throughput_min_power[throughput]}")