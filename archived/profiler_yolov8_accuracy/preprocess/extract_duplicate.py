import cv2
import json
import os
import sys

def extract_and_duplicate_frames(working_folder, interval, fps_target):
    clips_folder = os.path.join(working_folder, f"Clip_I{interval}")
    frames_folder = os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}")
    duplicate_result_path = os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}.json")
    
    # Create the output folder if it does not exist
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    # Get a sorted list of files in the input folder
    filenames = sorted(os.listdir(clips_folder))

    # Process each video file
    duplicate_result = {}
    for filename in filenames:
        video_path = os.path.join(clips_folder, filename)
        if os.path.isfile(video_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Get the base name of the video file without the extension
            clip_base_name = os.path.splitext(os.path.basename(video_path))[0]
            sampled_frame_number = []

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
            last_sampled_frame = None
            while True:
                ret, frame = video.read()
                if not ret:
                    break  # End of video if no frame is returned

                # Save the frame if it is the sampled frame or duplicate the last sampled frame
                if frame_number % frame_interval == 0 or last_sampled_frame is not None:
                    # Define the output frame file path
                    frame_path = os.path.join(frames_folder, f"{clip_base_name}_frame{frame_number + 1:05d}.jpg")
                    # If current frame is sampled, update last_sampled_frame
                    if frame_number % frame_interval == 0:
                        last_sampled_frame = frame
                        sampled_frame_number.append(f"{clip_base_name}_frame{frame_number + 1:05d}.jpg")
                    # Save the last sampled frame (either current or previous sampled frame)
                    cv2.imwrite(frame_path, last_sampled_frame)

                frame_number += 1

            # Release the video capture object
            video.release()
            print(f"Processed frames from video {filename} with target frame rate of {fps_target} fps")
            duplicate_result[clip_base_name] = sampled_frame_number
    
    with open(duplicate_result_path, 'w') as file:
        json.dump(duplicate_result, file, indent=4)

if __name__ == "__main__":
    extract_and_duplicate_frames(sys.argv[1], sys.argv[2], int(sys.argv[3]))
