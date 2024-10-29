import sys
from lib import getCpuStatus

if __name__ == "__main__":
	print("Current CPU Frequency", getCpuStatus())