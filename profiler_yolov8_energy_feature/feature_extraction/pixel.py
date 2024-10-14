import os
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_ssim_for_neighbors(directory_path, resize_dim):
    # List to store SSIM values
    ssim_values = []
    
    # Get a sorted list of image file names from the directory
    image_files = sorted([f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # Check if there are enough images for comparison
    if len(image_files) < 2:
        print("Not enough images to compare.")
        return None

    # Loop over the images and calculate SSIM between neighboring pairs
    for i in range(len(image_files) - 1):
        img1_path = os.path.join(directory_path, image_files[i])
        img2_path = os.path.join(directory_path, image_files[i+1])

        # Read the images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Could not read image: {image_files[i]} or {image_files[i+1]}")
            continue

        # Resize images to the same dimensions
        img1_resized = cv2.resize(img1, resize_dim)
        img2_resized = cv2.resize(img2, resize_dim)

        # Calculate SSIM
        score, _ = ssim(img1_resized, img2_resized, full=True)
        ssim_values.append((image_files[i], image_files[i+1], score))

    # Return SSIM values for neighboring pairs
    return ssim_values