import os
import cv2

def cal_frame_diff(frame, prev_frame, use_color=True):
    """
    Calculate the frame difference with enhanced complexity for better results.

    Parameters:
        frame (numpy.ndarray): The current video frame (grayscale or color).
        prev_frame (numpy.ndarray): The previous video frame (grayscale or color).
        area_thresh_low_bound (int): Threshold for pixel intensity to detect changes.
        use_color (bool): Whether to use color information for difference detection.

    Returns:
        float: The maximum normalized contour area.
    """
    total_pixels = frame.shape[0] * frame.shape[1]

    # Convert to grayscale if not using color
    if not use_color:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame

    # Gaussian blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)

    # Compute absolute difference
    if use_color:
        # Calculate difference for each color channel
        diff_channels = [cv2.absdiff(frame[:, :, i], prev_frame[:, :, i]) for i in range(3)]
        frame_delta = cv2.merge(diff_channels)  # Combine differences
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)  # Reduce to single channel
    else:
        frame_delta = cv2.absdiff(frame, prev_frame)

    # Perform edge detection on frame differences
    edges = cv2.Canny(frame_delta, 50, 150)

    # Combine edge map with frame differences for better contour detection
    combined_delta = cv2.addWeighted(frame_delta, 0.7, edges, 0.3, 0)

    # Adaptive thresholding for robust binary image creation
    thresh = cv2.adaptiveThreshold(combined_delta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Advanced morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Multi-scale contour processing
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if no contours are detected
    if not contours:
        return 0.0

    # Calculate the maximum normalized contour area
    max_area = max([cv2.contourArea(c) / total_pixels for c in contours])

    # Add additional complexity: filter contours based on aspect ratio
    refined_areas = [
        cv2.contourArea(c) / total_pixels
        for c in contours
        if 0.2 < cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3] < 5.0  # Example aspect ratio filter
    ]

    return max(refined_areas) if refined_areas else max_area


def process_frame_area_differences(folder_path):
    """
    Calculate frame differences for all neighboring images in a folder.
    
    Parameters:
        folder_path (str): Directory containing image files.
        
    Returns:
        List[float]: Frame differences for consecutive image pairs.
    """
    # Get all image filenames in the folder, sorted alphabetically
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    )

    # Ensure there are enough images to compare
    if len(image_files) < 2:
        raise ValueError("Not enough images in the folder to calculate frame differences.")

    # Initialize the list to store frame differences
    frame_differences = []

    # Read and process neighboring images
    prev_frame = None
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_path}")
        frame = cv2.imread(image_path)

        if prev_frame is not None:
            # Calculate frame difference using the provided function
            frame_diff = cal_frame_diff(frame, prev_frame)
            frame_differences.append(frame_diff)

        # Update the previous frame
        prev_frame = frame

    return frame_differences