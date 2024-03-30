def getcurStatus():
	"""Get current system knob status, including cpu/gpu/memory freqs
	as well as the hotplug status of the Denver cores"""
	
	cpuFreq_fname = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
	gpuFreq_fname = "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq"
	emcFreq_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"

	cpuFreq, gpuFreq, emcFreq = None, None, None
	with open(cpuFreq_fname, 'r') as f:
		cpuFreq = int(f.read().strip('\n'))
	with open(gpuFreq_fname, 'r') as f:
		gpuFreq = int(f.read().strip('\n'))
	with open(emcFreq_fname, 'r') as f:
		emcFreq = int(f.read().strip('\n'))

	return cpuFreq, gpuFreq, emcFreq

if __name__ == "__main__":
	cpuFreq_cur, gpuFreq_cur, emcFreq_cur = getcurStatus()
	print("Current Frequency", cpuFreq_cur, gpuFreq_cur, emcFreq_cur)