import cv2
import os
import sys

def extract_frames_from_videos(input_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a sorted list of files in the input folder
    filenames = sorted(os.listdir(input_folder))
    
    # Process each file
    frames_per_clip = []
    for filename in filenames:
        video_path = os.path.join(input_folder, filename)
        if os.path.isfile(video_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Get the base name of the video file without the extension
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Open the video file
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Error: Could not open video {filename}")
                continue
            
            frame_number = 1
            while True:
                ret, frame = video.read()
                if not ret:
                    break  # End of video if no frame is returned

                # Define the output frame file path
                frame_path = os.path.join(output_folder, f"{base_name}_frame{frame_number:05d}.jpg")
                
                # Save the frame
                cv2.imwrite(frame_path, frame)
                frame_number += 1

            # Release the video capture object
            video.release()
            print(f"Extracted {frame_number - 1} frames from video {filename}")
            frames_per_clip.append(frame_number - 1)
    
    return frames_per_clip

if __name__ == "__main__":
    extract_frames_from_videos(sys.argv[1], sys.argv[2])