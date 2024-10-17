from ultralytics import YOLO

def detect_yolov8(source_path, image_width, model_path="/home/jiaxi/cs525/Assets/models/yolov8n.pt", save_result=False):
    model = YOLO(model_path)
    results = model(source_path, imgsz=image_width, save=save_result)