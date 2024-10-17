import cv2
import os
import numpy as np

def detect_corners_in_directory(directory_path, output_dir):
    # # Create the output directory if it doesn't exist
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # Iterate over all the files in the directory
    for filename in sorted(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {filename}")
        
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(file_path)
            
            if image is None:
                print(f"Unable to read image: {filename}")
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Convert to float32 for the cornerHarris function
            gray = np.float32(gray)
            
            # Detect corners using Harris Corner Detection
            dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
            
            # Dilate the corner points to enhance the features
            dst = cv2.dilate(dst, None)

            # Threshold to mark the corners in the original image
            image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red

            # # Create the output path
            # output_file_path = os.path.join(output_dir, f"corners_{filename}")
            
            # # Save the result to the output directory
            # cv2.imwrite(output_file_path, image)
            # print(f"Processed {filename} and saved corner detection result to {output_file_path}")
