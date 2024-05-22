import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

class DVFSController:
    def __init__(self, w, GPU_FREQ, CPU_FREQ, efficiency_array):
        self.w = w
        self.GPU_FREQ = GPU_FREQ
        self.CPU_FREQ = CPU_FREQ
        self.efficiency_array = np.array(efficiency_array)
        self.observed_data = {'X': [], 'y': []}
        self.suggestion_latency = []
        self.update_latency = []

        self.max_throughput = 0
        self.min_throughput = 0
        
        # Initialize Gaussian Process Regressor
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
        
        # Standard Scaler for input normalization
        self.scaler = StandardScaler()
    
    def get_update_latency(self):
        return self.update_latency
    
    def update_model(self, cpu_freq, gpu_freq, real_throughput):
        start_time = time.time()
        self.update_model_helper(cpu_freq, gpu_freq, real_throughput)
        end_time = time.time()

        self.update_latency.append(end_time - start_time)

    def update_model_helper(self, cpu_freq, gpu_freq, real_throughput):
        if len(self.observed_data['X']) == 0:
            self.max_throughput = real_throughput
        if len(self.observed_data['X']) == 1:
            self.min_throughput = real_throughput

        # Update observed data
        self.observed_data['X'].append([cpu_freq, gpu_freq])
        self.observed_data['y'].append(real_throughput)
        
        # Fit model if we have at least one observation
        if len(self.observed_data['X']) > 1:
            X = np.array(self.observed_data['X'])
            y = np.array(self.observed_data['y'])
            X_scaled = self.scaler.fit_transform(X)  # Standardization
            self.gpr.fit(X_scaled, y)
    
    def get_suggestion_latency(self):
        return self.suggestion_latency
    
    def suggest_combination(self, required_throughput):
        start_time = time.time()
        suggestion_frequency = self.suggest_combination_helper(required_throughput)
        end_time = time.time()

        self.suggestion_latency.append(end_time - start_time)
        return suggestion_frequency

    def suggest_combination_helper(self, required_throughput):
        # Define the search space
        search_space = np.array([[cf, gf] for cf in self.CPU_FREQ for gf in self.GPU_FREQ])

        # Predict throughputs using the model
        if len(self.observed_data['X']) == 0:
            return self.CPU_FREQ[-1], self.GPU_FREQ[-1]
        elif len(self.observed_data['X']) == 1:
            return self.CPU_FREQ[0], self.GPU_FREQ[0]
        else:
            if required_throughput >= self.max_throughput:
                return self.CPU_FREQ[-1], self.GPU_FREQ[-1]
            elif  required_throughput <= self.min_throughput:
                return self.CPU_FREQ[0], self.GPU_FREQ[0]
            else:
                search_space_scaled = self.scaler.transform(search_space)  # Standardization
                predicted_throughputs, sigma = self.gpr.predict(search_space_scaled, return_std=True)
        
        # Filter combinations that meet the required throughput
        valid_combinations = search_space[predicted_throughputs >= required_throughput]
        
        # If no valid combination, return the most efficient combination
        if len(valid_combinations) == 0:
            max_efficiency_index = np.unravel_index(np.argmax(self.efficiency_array, axis=None), self.efficiency_array.shape)
            return self.CPU_FREQ[max_efficiency_index[1]], self.GPU_FREQ[max_efficiency_index[0]]

        # Calculate efficiency and distance for each valid combination
        efficiencies = [self.efficiency_array[self.GPU_FREQ.index(gf), self.CPU_FREQ.index(cf)] for cf, gf in valid_combinations]
        distances = [min((predicted_throughput - required_throughput) / required_throughput, 1) for predicted_throughput in predicted_throughputs[predicted_throughputs >= required_throughput]]

        # Normalize distances to be between 0 and 1
        distances = [abs(d) for d in distances]

        # Calculate the combined score
        scores = [self.w * eff + (1 - self.w) * (1 - dist) for eff, dist in zip(efficiencies, distances)]
        best_combination_index = np.argmax(scores)
        
        best_combination = valid_combinations[best_combination_index]
        return best_combination[0], best_combination[1]