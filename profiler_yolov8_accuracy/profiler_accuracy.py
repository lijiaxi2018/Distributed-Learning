import os
import sys
from preprocess.split_clip import split_all_videos_into_clips
from preprocess.extract_all import extract_frames_from_videos
from preprocess.extract_some import extract_frames_by_fps
from preprocess.extract_duplicate import extract_and_duplicate_frames

if __name__ == "__main__":
    test_videos_path = sys.argv[1]
    working_path = sys.argv[2]
    
    if not os.path.exists(working_path):
        os.makedirs(working_path)
    
    interval = int(sys.argv[3])
    fps = int(sys.argv[4])

    clips_path = os.path.join(working_path, f"Clip_{interval}")
    frames_all_path = os.path.join(working_path, f"Frame_All")
    frames_some_path = os.path.join(working_path, f"Frame_{fps}")
    frames_duplicate_path = os.path.join(working_path, f"Frame_D{fps}")

    split_all_videos_into_clips(test_videos_path, clips_path, interval)
    extract_frames_from_videos(clips_path, frames_all_path)
    extract_frames_by_fps(clips_path, frames_some_path, fps)
    extract_and_duplicate_frames(clips_path, frames_duplicate_path, fps)