from ultralytics import YOLO

def detect_yolov8(model_path="/home/jiaxi/cs525/Assets/models/yolov8n.pt", source_path="/home/jiaxi/cs525/Assets/1800_1K", image_width=960, save_result=False):
    model = YOLO(model_path)
    results = model(source_path, imgsz=image_width, save=save_result)

if __name__ == "__main__":
	detect_yolov8("/home/jiaxi/cs525/Assets/models/yolov8n.pt", "/home/jiaxi/cs525/Assets/frame_0.jpg")