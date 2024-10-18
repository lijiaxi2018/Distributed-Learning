import cv2
import numpy as np

def compute_edge_difference(image_path1, image_path2, bins=256):
    # Load the images in grayscale
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    # Check if the images were loaded properly
    if img1 is None:
        print(f"Error loading image: {image_path1}")
        return
    if img2 is None:
        print(f"Error loading image: {image_path2}")
        return
    
    # Apply Canny edge detection to both images
    edges1 = cv2.Canny(img1, 100, 200)
    edges2 = cv2.Canny(img2, 100, 200)
    
    # Compute the histogram of edge pixels (0 for no edge, 255 for edge)
    hist1, _ = np.histogram(edges1.ravel(), bins=bins, range=(0, 256))
    hist2, _ = np.histogram(edges2.ravel(), bins=bins, range=(0, 256))
    
    # Normalize the histograms (so that their total counts are the same)
    hist1 = hist1.astype(np.float32) / np.sum(hist1)
    hist2 = hist2.astype(np.float32) / np.sum(hist2)
    
    # Compute the difference between the two histograms
    hist_diff = np.sum(np.abs(hist1 - hist2))
    
    # Output the histogram difference
    # print(f"Edge distribution difference between {image_path1} and {image_path2}: {hist_diff}")

    return hist_diff