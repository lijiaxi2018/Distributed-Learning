import cv2
from skimage.metrics import structural_similarity as ssim

def compute_ssim_difference(image_path1, image_path2, target_size):
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
    
    # Ensure the two images have the same dimensions
    if img1.shape != img2.shape:
        print(f"Error: Images have different dimensions - {image_path1}: {img1.shape}, {image_path2}: {img2.shape}")
        return
    
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)

    # Compute SSIM between the two images
    ssim_index, _ = ssim(img1, img2, full=True)
    
    # Output the SSIM value
    # print(f"SSIM between {image_path1} and {image_path2}: {ssim_index}")

    return ssim_index