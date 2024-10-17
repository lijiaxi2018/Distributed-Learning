import cv2
import os
import numpy as np

def detect_edges_in_directory_v1(directory_path, output_dir):
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
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Unable to read image: {filename}")
                continue

            blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
            edges = cv2.Canny(blurred_image, 30, 100)
            
            # # Create the output path
            # output_file_path = os.path.join(output_dir, f"edges_{filename}")
            
            # # Save the result to the output directory
            # cv2.imwrite(output_file_path, edges)
            # print(f"Processed {filename} and saved edges to {output_file_path}")

def detect_edges_in_directory_v2(directory):
    # Get a sorted list of all image file paths in the directory
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Ensure there are at least two images
    if len(image_files) < 2:
        print("Not enough images in the directory.")
        return
    
    # Initialize list to store edge differences between neighboring images
    edge_differences = []

    # Loop through each pair of neighboring images
    for i in range(len(image_files) - 1):
        print(f"Processing {image_files[i]}")
        
        # Read the neighboring images
        img1 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_files[i + 1], cv2.IMREAD_GRAYSCALE)
        
        # Check if the images were loaded properly
        if img1 is None or img2 is None:
            print(f"Error loading images: {image_files[i]}, {image_files[i+1]}")
            continue
        
        img1 = cv2.GaussianBlur(img1, (3, 3), 0)
        img2 = cv2.GaussianBlur(img2, (3, 3), 0)

        # Apply Canny edge detection to both images
        edges1 = cv2.Canny(img1, 30, 150)
        edges2 = cv2.Canny(img2, 30, 150)
        
        # Compute the absolute difference between the edges of the two images
        edge_diff = np.sum(np.abs(edges1.astype(np.int32) - edges2.astype(np.int32)))
        
        # Store the edge difference along with the image names
        edge_differences.append({
            'image1': os.path.basename(image_files[i]),
            'image2': os.path.basename(image_files[i + 1]),
            'edge_difference': edge_diff
        })
    
    # # Output the results
    # for diff in edge_differences:
    #     print(f"Edge difference between {diff['image1']} and {diff['image2']}: {diff['edge_difference']}")