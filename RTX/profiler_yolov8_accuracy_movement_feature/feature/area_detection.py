import cv2
import numpy as np

def compute_area_difference(image_path1, image_path2, target_size, area_thresh_low_bound=25):
    """
    Calculates the normalized maximum area of motion or changes between two images.
    
    Parameters:
        image_path1 (str): Path to the first image (current frame).
        image_path2 (str): Path to the second image (previous frame).
        area_thresh_low_bound (int): Threshold value for detecting significant pixel differences.
    
    Returns:
        float: The normalized maximum area of motion or changes.
    """
    # Read the images
    frame = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if frame is None or prev_frame is None:
        raise ValueError("One or both image paths are invalid or images could not be read.")
    
    # Ensure the images are the same size
    if frame.shape != prev_frame.shape:
        raise ValueError("The input images must have the same dimensions.")
    
    frame = cv2.resize(frame, target_size)
    prev_frame = cv2.resize(prev_frame, target_size)
    
    # Calculate the total number of pixels
    total_pixels = frame.shape[0] * frame.shape[1]
    
    # Calculate the absolute difference between the two frames
    frame_delta = cv2.absdiff(frame, prev_frame)
    
    # Apply a binary threshold
    _, thresh = cv2.threshold(frame_delta, area_thresh_low_bound, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill gaps
    thresh = cv2.dilate(thresh, None)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found, return 0.0
    if not contours:
        return 0.0
    
    # Calculate the normalized maximum area of the contours
    max_normalized_area = max(cv2.contourArea(c) / total_pixels for c in contours)
    
    return max_normalized_area