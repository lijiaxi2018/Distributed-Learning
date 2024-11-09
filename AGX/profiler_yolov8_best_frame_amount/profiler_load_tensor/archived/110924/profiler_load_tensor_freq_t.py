import json
import time
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus, getEmcStatus
from PIL import Image
from tegrastats_lib.get_status import get_tegrastats_metrics
from temperature.get_temperature import get_cpu_temperature, get_gpu_temperature
from torchvision import transforms
from ultralytics import YOLO

OUTPUT_FILENAME = "Latency"
IMAGE_SIZE = (320, 640)
ITERATION = 6000

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

img = Image.open("image.jpg")
img_tensor = transform(img).unsqueeze(0)
img_tensor = img_tensor.to("cuda")

model_path="/home/jiaxi/cs525/Assets/models/yolov8n.pt"
model = YOLO(model_path)
model.to("cuda")

# Inference
img_tensor_list = []
for i in range(ITERATION):
    img_tensor_list.append(img_tensor)

latency_list = []
cpu_freq_list = []
gpu_freq_list = []
emc_freq_list = []

cpu_temp_list = []
gpu_temp_list = []
for img in img_tensor_list:
    metrics = get_tegrastats_metrics()

    cpu_freq_list.append(metrics['cpu_freq'])
    # gpu_freq_list.append(getGpuStatus())
    # emc_freq_list.append(getEmcStatus())

    cpu_temp_list.append(metrics['cpu_temp'])
    gpu_temp_list.append(metrics['gpu_temp'])

    t0 = time.perf_counter()
    results = model(img)
    t1 = time.perf_counter()
    latency = t1 - t0

    latency_list.append(latency)
    print("Latency: ", latency)

result = {}
result['latency_list'] = latency_list
result['cpu_freq_list'] = cpu_freq_list
# result['gpu_freq_list'] = gpu_freq_list
# result['emc_freq_list'] = emc_freq_list

result['cpu_temp_list'] = cpu_temp_list
result['gpu_temp_list'] = gpu_temp_list

with open(f'{OUTPUT_FILENAME}.json', 'w') as file:
    json.dump(result, file, indent=4)

