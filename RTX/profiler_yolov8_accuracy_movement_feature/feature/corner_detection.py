import cv2

def compute_corner_difference(image_path1, image_path2, target_size):
    """
    Calculate the corner feature difference between two images using ORB feature detection.
    
    Parameters:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
    
    Returns:
        float: The feature difference score (lower is more similar).
    """
    # Read the images in grayscale
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")
    
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)
    
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Use BFMatcher to find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (lower distance indicates better matches)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate the average distance of matches as the feature difference score
    if len(matches) == 0:
        return float('inf')  # No matches found, return a large value
    else:
        average_distance = sum([match.distance for match in matches]) / len(matches)
        return average_distance