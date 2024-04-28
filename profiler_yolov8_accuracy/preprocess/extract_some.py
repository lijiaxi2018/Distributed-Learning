import cv2
import os
import sys

def extract_frames_by_fps(input_folder, output_folder, fps_target):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a sorted list of files in the input folder
    filenames = sorted(os.listdir(input_folder))

    # Process each video file
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

            # Get the original FPS of the video
            original_fps = video.get(cv2.CAP_PROP_FPS)

            # Calculate the frame interval to match the target fps
            frame_interval = round(original_fps / fps_target)

            frame_number = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break  # End of video if no frame is returned

                # Only save frames according to the specified fps
                if frame_number % frame_interval == 0:
                    # Define the output frame file path
                    frame_path = os.path.join(output_folder, f"{base_name}_frame{frame_number + 1}.jpg")
                    # Save the frame
                    cv2.imwrite(frame_path, frame)

                frame_number += 1

            # Release the video capture object
            video.release()
            print(f"Extracted frames from video {filename} at {fps_target} fps")

if __name__ == "__main__":
    extract_frames_by_fps(sys.argv[1], sys.argv[2], int(sys.argv[3]))
