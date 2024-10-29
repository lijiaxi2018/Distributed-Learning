
import sys

if __name__ == "__main__":
	while(True):
		with open("/sys/kernel/debug/bpmp/debug/clk/emc/rate", 'w') as f:
			f.write(str(int(sys.argv[1])))