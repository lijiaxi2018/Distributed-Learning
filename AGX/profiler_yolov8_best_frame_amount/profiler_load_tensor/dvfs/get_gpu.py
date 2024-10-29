import sys
from lib import getGpuStatus

if __name__ == "__main__":
	print("Current GPU Frequency", getGpuStatus())