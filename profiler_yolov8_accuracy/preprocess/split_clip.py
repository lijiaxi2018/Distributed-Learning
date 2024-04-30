import cv2
import os
import sys

def split_all_videos_into_clips(input_folder, output_folder, interval):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = sorted(os.listdir(input_folder))
    clips_per_video = []
    # Iterate through each file in the input folder
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

            # Get frames per second (FPS) of the video
            fps = video.get(cv2.CAP_PROP_FPS)
            # Calculate the number of frames per clip
            frames_per_clip = int(fps * interval)

            # Initialize frame count and clip number
            frame_count = 0
            clip_number = 1
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            needed_frames = total_frames - (total_frames % frames_per_clip)

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
                    output_path = os.path.join(output_folder, f'{base_name}_clip{clip_number:05d}.mp4')
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
            clips_per_video.append(clip_number - 1)
    
    return filenames, clips_per_video

if __name__ == "__main__":
    split_all_videos_into_clips(sys.argv[1], sys.argv[2], int(sys.argv[3]))
