import os
import sys
from preprocess.split_clip import split_all_videos_into_clips
from preprocess.extract_all import extract_frames_from_videos
from preprocess.extract_duplicate import extract_and_duplicate_frames
from detect.detect import detect_yolov8
from detect.lib import move_folder, delete_folder
from metric.metric_yyy import metric
from metric.metric_clip import metric_clip
from metric.metric_video import metric_video

INTERVALS = [5]
FPSS = [1, 2, 3, 5, 10, 15, 30]
IMAGE_WIDTH = 1280
DETECT_WIDTH = 960
CLASSES = [2]

if __name__ == "__main__":
    test_videos_path = sys.argv[1]

    interval = INTERVALS[0]
    working_path = f"{test_videos_path}_I{interval}"
    if not os.path.exists(working_path):
        os.makedirs(working_path)
    
    # Global data preprocessing
    clips_path = os.path.join(working_path, f"Clip_I{interval}")
    frames_all_path = os.path.join(working_path, f"Frame_All_I{interval}")
    video_filenames, clips_per_video = split_all_videos_into_clips(test_videos_path, clips_path, interval)
    frames_per_clip = extract_frames_from_videos(clips_path, frames_all_path)

    # Global data detection
    detect_yolov8(frames_all_path, IMAGE_WIDTH, True, True)
    move_folder(".\\runs\\detect\\predict\\labels", working_path, f"Label_GT_I{interval}")
    delete_folder(".\\runs")

    for fps in FPSS:
        frames_duplicate_path = os.path.join(working_path, f"Frame_Dup_I{interval}_F{fps}")
        extract_and_duplicate_frames(clips_path, frames_duplicate_path, fps)

        detect_yolov8(frames_duplicate_path, DETECT_WIDTH, True, True)
        move_folder(".\\runs\\detect\\predict\\labels", working_path, f"Label_Dup_I{interval}_F{fps}")
        metric(frames_all_path, os.path.join(working_path, f"Label_GT_I{interval}"), os.path.join(working_path, f"Label_Dup_I{interval}_F{fps}"), os.path.join(working_path, f"Label_Dup_I{interval}_F{fps}.json"), 0.5, CLASSES)
        delete_folder(".\\runs")
    
    metric_clip(frames_per_clip, FPSS, os.path.join(working_path, f"Label_Dup_I{interval}_F"), os.path.join(working_path, f"Clip_Result_I{interval}.json"))
    metric_video(working_path, os.path.join(working_path, f"Clip_Result_I{interval}.json"), video_filenames, clips_per_video)