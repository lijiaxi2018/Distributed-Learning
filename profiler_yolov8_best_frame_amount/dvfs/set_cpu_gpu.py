import sys
from lib import setCpu, setGpu

if __name__ == "__main__":
	setCpu(int(sys.argv[1]))
	setGpu(int(sys.argv[2]))