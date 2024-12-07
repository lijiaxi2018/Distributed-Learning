import os
import cv2
import torch
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from PIL import Image

def detect_faster_rcnn(input_dir, image_width, confidence_threshold=0.5):
    """
    Perform object detection using Faster R-CNN on all frames in a directory.
    
    Parameters:
        input_dir (str): Path to the directory containing input frames (images).
        output_dir (str): Path to save the output images with detections.
        confidence_threshold (float): Confidence threshold for filtering predictions.
    """
    # # Create output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained Faster R-CNN model
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()

    # Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loop through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f'Processing: {filename}')

            # Read image
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert("RGB")

            width, height = image.size
            aspect_ratio = height / width
            new_width = image_width
            new_height = int(new_width * aspect_ratio)
            image = image.resize((new_width, new_height))

            # Transform image for model input
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Perform detection
            with torch.no_grad():
                predictions = model(input_tensor)

            # # Get boxes, labels, and scores
            # boxes = predictions[0]['boxes']
            # labels = predictions[0]['labels']
            # scores = predictions[0]['scores']

            # # Filter predictions based on confidence threshold
            # filtered_boxes = []
            # for box, score in zip(boxes, scores):
            #     if score > confidence_threshold:
            #         filtered_boxes.append(box)

            # # Convert image to OpenCV format for drawing
            # image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # # Draw bounding boxes on the image
            # for box in filtered_boxes:
            #     x1, y1, x2, y2 = box.int().tolist()
            #     cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

            # # Save the resulting image
            # output_path = os.path.join(output_dir, filename)
            # cv2.imwrite(output_path, cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))

    # print(f"Object detection completed. Results saved to {output_dir}.")

if __name__=="__main__":
    detect_faster_rcnn("/home/jiaxi/cs525/Assets/60_1K", 640)