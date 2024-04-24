from ultralytics import YOLO

def detect_yolov8(model_path="/home/jiaxi/cs525/Assets/models/yolov8n.pt", source_path="/home/jiaxi/cs525/Assets/1800_1K", save_result=False):
    model = YOLO(model_path)
    results = model(source_path, save=save_result)