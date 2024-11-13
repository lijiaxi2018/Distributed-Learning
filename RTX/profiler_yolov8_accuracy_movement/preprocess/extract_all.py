import cv2
import os
import sys

def extract_all_frames(working_folder, interval):
    clips_folder = os.path.join(working_folder, f"Clip_I{interval}")
    frames_folder = os.path.join(working_folder, f"Frame_All_I{interval}")

    # Create the output folder if it does not exist
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    # Get a sorted list of files in the input folder
    filenames = sorted(os.listdir(clips_folder))
    
    # Process each file
    for filename in filenames:
        clip_path = os.path.join(clips_folder, filename)
        if os.path.isfile(clip_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Get the base name of the video file without the extension
            clip_base_name = os.path.splitext(os.path.basename(clip_path))[0]
            
            # Open the video file
            video = cv2.VideoCapture(clip_path)
            if not video.isOpened():
                print(f"Error: Could not open video {filename}")
                continue
            
            frame_number = 1
            while True:
                ret, frame = video.read()
                if not ret:
                    break  # End of video if no frame is returned

                # Define the output frame file path
                frame_path = os.path.join(frames_folder, f"{clip_base_name}_frame{frame_number:05d}.jpg")
                
                # Save the frame
                cv2.imwrite(frame_path, frame)
                frame_number += 1

            # Release the video capture object
            video.release()
            print(f"Extracted {frame_number - 1} frames from video {filename}")
    
    return

if __name__ == "__main__":
    extract_all_frames('Video1.mp4', 'temp', 15)