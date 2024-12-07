import cv2
import os
import numpy as np

def compute_corner_difference(image1, image2):
    """
    Computes the corner difference between two images using Harris Corner Detection.

    Parameters:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.

    Returns:
        float: Normalized corner difference (a measure of how different the corners are).
    """
    def detect_corners(image):
        """
        Detect corners in an image using Harris Corner Detection.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: A binary mask indicating corner locations.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Convert to float32 for Harris Corner Detection
        gray = np.float32(gray)
        
        # Apply Harris Corner Detection
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        
        # Dilate the result for better visibility of corners
        dst = cv2.dilate(dst, None)
        
        # Threshold to create a binary corner mask
        corner_mask = dst > 0.01 * dst.max()
        return corner_mask

    # Ensure both images have the same size
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Input images must have the same dimensions.")
    
    # Detect corners in both images
    corners1 = detect_corners(image1)
    corners2 = detect_corners(image2)
    
    # Compute the absolute difference between the two corner masks
    corner_diff = np.abs(corners1.astype(np.float32) - corners2.astype(np.float32))
    
    # Compute the normalized difference
    total_pixels = image1.shape[0] * image1.shape[1]
    normalized_difference = np.sum(corner_diff) / total_pixels
    
    return normalized_difference

def process_corner_differences(folder_path):
    """
    Calculate the corner differences for all neighboring images in a folder.

    Parameters:
        folder_path (str): Directory containing image files.

    Returns:
        List[float]: Corner differences for consecutive image pairs.
    """
    # Get all image filenames in the folder, sorted alphabetically
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    )

    # Ensure there are enough images to compare
    if len(image_files) < 2:
        raise ValueError("Not enough images in the folder to calculate corner differences.")

    # Initialize the list to store corner differences
    corner_differences = []

    # Iterate over neighboring images
    prev_image = None
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}")

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Unable to read image: {image_file}")
            continue

        # If there's a previous image, compute the corner difference
        if prev_image is not None:
            try:
                difference = compute_corner_difference(prev_image, image)
                corner_differences.append(difference)
            except ValueError as e:
                print(f"Error comparing {image_files[i-1]} and {image_file}: {e}")

        # Update the previous image
        prev_image = image

    return corner_differences