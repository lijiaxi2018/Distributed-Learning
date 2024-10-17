import cv2
import json
import os
import sys

def split_video_into_clips(video_path, working_folder, interval):
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    clips_folder = os.path.join(working_folder, f"Clip_I{interval}")
    split_result_path = os.path.join(working_folder, f"Split_Result.json")

    # Create the output folder if it does not exist
    if not os.path.exists(clips_folder):
        os.makedirs(clips_folder)

    split_result = {}
    if os.path.isfile(video_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Open the video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {filename}")
            return

        # Get frames per second (FPS) of the video
        fps = video.get(cv2.CAP_PROP_FPS)
        split_result['fps'] = fps
        # Calculate the number of frames per clip
        frames_per_clip = int(fps * interval)
        split_result['frames_per_clip'] = frames_per_clip

        # Initialize frame count and clip number
        frame_count = 0
        clip_number = 1
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        needed_frames = total_frames - (total_frames % frames_per_clip)
        split_result['needed_frames'] = needed_frames

        # Define the output video writer; placeholder to be initialized per clip
        out = None

        # Loop through the video
        while frame_count < needed_frames:
            ret, frame = video.read()
            if not ret:
                break  # End of video if no frame is returned

            # Start a new clip file if needed
            if frame_count % frames_per_clip == 0:
                if out is not None:
                    out.release()  # Close the previous video writer object

                # Define the output video file path
                output_path = os.path.join(clips_folder, f'{base_name}_clip{clip_number:05d}.mp4')
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video.get(3)), int(video.get(4))))
                clip_number += 1

            # Write the frame to the currently open video file
            out.write(frame)
            frame_count += 1

        # Release the video capture and last video writer objects
        video.release()
        if out:
            out.release()

        print(f"Video {filename} split into {clip_number - 1} clips without saving the remainder.")
        split_result['clip_number'] = clip_number - 1
    
    with open(split_result_path, 'w') as file:
        json.dump(split_result, file, indent=4)
    return

if __name__ == "__main__":
    split_video_into_clips('Video1.mp4', 'temp', 15)
