from ultralytics import YOLO

def detect_yolov8(source_path="\\Users\\ljx\\Documents\\Study\\cs525\\Assets\\dataset\\120_1K", image_width=960, save_bbox=False, save_confidence=False, model_path="\\Users\\ljx\\Documents\\Study\\cs525\\Assets\\models\\yolov8n.pt"):
    model = YOLO(model_path)
    results = model(source_path, imgsz=image_width, save_txt=save_bbox, save_conf=save_confidence)

if __name__ == "__main__":
    detect_yolov8("\\Users\\ljx\\Documents\\Study\\cs525\\Assets\\dataset\\120_1K", 320, True, True)
