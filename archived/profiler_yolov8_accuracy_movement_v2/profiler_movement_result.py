import os
import sys
from preprocess.split_clip import split_video_into_clips
from preprocess.extract_all import extract_all_frames
from preprocess.extract_duplicate import extract_and_duplicate_frames
from detect.detect import detect_yolov8
from detect.lib import move_folder, delete_folder
from metric.metric_yyy import metric
from metric.summarize_result import summarize_result
from movement.frame_movement import frame_movement
from movement.clip_movement import clip_movement

INTERVALS = [1]
FPSS = [1, 2, 3, 5, 6, 10, 15, 30]
IMAGE_WIDTH = 1280
DETECT_WIDTH = 960
CLASSES = [2]

if __name__ == "__main__":
    video_folder_path = sys.argv[1]
    interval = INTERVALS[0]
    base_working_path = f"{video_folder_path}_I{interval}"
    if not os.path.exists(base_working_path):
        os.makedirs(base_working_path)

    filenames = sorted(os.listdir(video_folder_path))
    for filename in filenames:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        video_path = os.path.join(video_folder_path, filename)
        
        working_path = os.path.join(base_working_path, f"{base_name}_I{interval}")
        if not os.path.exists(working_path):
            os.makedirs(working_path)
        
        # split_video_into_clips(video_path, working_path, interval)
        # extract_all_frames(working_path, interval)

        # frames_all_path = os.path.join(working_path, f"Frame_All_I{interval}")
        # detect_yolov8(frames_all_path, IMAGE_WIDTH, True, True)
        # move_folder(".\\runs\\detect\\predict\\labels", working_path, f"Label_GT_I{interval}")
        # delete_folder(".\\runs")

        # # Accuracy Across Different FPSs
        # for fps in FPSS:
        #     frames_duplicate_path = os.path.join(working_path, f"Frame_Dup_I{interval}_F{fps}")
        #     extract_and_duplicate_frames(working_path, interval, fps)

        #     detect_yolov8(frames_duplicate_path, DETECT_WIDTH, True, True)
        #     move_folder(".\\runs\\detect\\predict\\labels", working_path, f"Label_Dup_I{interval}_F{fps}")
        #     metric(frames_all_path, working_path, interval, fps, 0.5, CLASSES)
        #     delete_folder(".\\runs")
        
        # accuracy_results_filename = f"{base_name}_Accuracy_Result.json"
        # summarize_result(working_path, accuracy_results_filename, interval, FPSS, CLASSES)

        # Movement Across Different FPSs
        for fps in FPSS:
            frame_movement(working_path, interval, fps, CLASSES)
        movement_results_filename = f"{base_name}_Movement_Result.json"
        clip_movement(working_path, movement_results_filename, interval, FPSS, CLASSES)