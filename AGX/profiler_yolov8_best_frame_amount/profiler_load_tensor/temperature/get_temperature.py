def get_cpu_temperature():
    temp = -1.
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000.0
    except FileNotFoundError:
        print("Thermal files not found.")
    return temp

def get_gpu_temperature():
    temp = -1.
    try:
        with open("/sys/class/thermal/thermal_zone1/temp") as f:
            temp = int(f.read().strip()) / 1000.0
    except FileNotFoundError:
        print("Thermal files not found.")
    return temp
