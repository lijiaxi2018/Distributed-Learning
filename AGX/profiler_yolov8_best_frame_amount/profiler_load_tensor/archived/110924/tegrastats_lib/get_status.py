import subprocess
import re

def get_tegrastats_metrics():
    # Start tegrastats without the --count option
    cmd = ['tegrastats', '--interval', '10']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Read the first line of output
        line = process.stdout.readline().strip()

        # Initialize the metrics dictionary
        metrics = {}

        # Parse CPU frequencies
        cpu_pattern = r'CPU \[([^\]]+)\]'
        cpu_match = re.search(cpu_pattern, line)
        if cpu_match:
            cpu_info = cpu_match.group(1)
            # Extract frequencies
            cpu_freqs = re.findall(r'\d+%@(\d+)', cpu_info)
            cpu_freqs = [int(freq) for freq in cpu_freqs]
            metrics['cpu_freqs'] = cpu_freqs
            metrics['cpu_freq'] = max(cpu_freqs) if cpu_freqs else None
        else:
            metrics['cpu_freqs'] = []
            metrics['cpu_freq'] = None

        # Parse GPU frequency
        gpu_pattern = r'GR3D_FREQ (\d+)%@?(\d+)?'
        gpu_match = re.search(gpu_pattern, line)
        if gpu_match:
            gpu_usage = gpu_match.group(1)
            gpu_freq = gpu_match.group(2)
            if gpu_freq:
                metrics['gpu_freq'] = int(gpu_freq)
            else:
                metrics['gpu_freq'] = None  # Frequency not provided
        else:
            metrics['gpu_freq'] = None

        # Parse Memory frequency (EMC)
        emc_pattern = r'EMC_FREQ (\d+)%@(\d+)'
        emc_match = re.search(emc_pattern, line)
        if emc_match:
            emc_usage = emc_match.group(1)
            emc_freq = emc_match.group(2)
            metrics['mem_freq'] = int(emc_freq)
        else:
            metrics['mem_freq'] = None  # EMC frequency not available

        # Parse temperatures
        temp_pattern = r'(\w+)@([0-9.]+)C'
        temp_matches = re.findall(temp_pattern, line)
        temps = {}
        for sensor, temp in temp_matches:
            temps[sensor] = float(temp)
        metrics['cpu_temp'] = temps.get('cpu', None)
        metrics['gpu_temp'] = temps.get('gpu', None)
        # Adjust 'tdiode' if necessary based on your system's output
        metrics['mem_temp'] = temps.get('tdiode', None)

        return metrics

    finally:
        # Terminate the tegrastats process
        process.terminate()
        process.wait()

# Example usage:
if __name__ == '__main__':
    metrics = get_tegrastats_metrics()
    print(metrics)