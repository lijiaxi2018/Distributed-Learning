import torch
from PIL import Image
from torchvision import transforms
import time

# Define the transformation (resizing and normalization as per YOLOv8 input requirement)
transform = transforms.Compose([
    transforms.Resize((640, 640)),   # YOLOv8 typically uses 640x640 images
    transforms.ToTensor()            # Convert image to PyTorch tensor
])

# Load the image
img = Image.open("0.jpg")

# Apply the transformations
img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

# Move tensor to GPU
img_tensor = img_tensor.to("cuda")

from ultralytics import YOLO

model_path="\\Users\\ljx\\Documents\\Study\\cs525\\Assets\\models\\yolov8n.pt"
# Load the YOLOv8 model (assuming it's a model file or pretrained one)
model = YOLO(model_path)

model.conf = 0.25  # Set confidence threshold (try increasing this value)
model.iou = 0.45   # Set IoU threshold for NMS

# Ensure model is on the GPU
model.to("cuda")

# Perform inference on the preloaded tensor


lists = []
for i in range(100):
    lists.append(img_tensor)

for img in lists:
    t0 = time.perf_counter()
    results = model(img)
    # results = model("0.jpg")
    t1 = time.perf_counter()
    latency = t1 - t0
    print("Latency: ", latency)
    
# t1 = time.perf_counter()
# Process the results
# print(results)

latency = t1 - t0
print("Latency: ", latency)

