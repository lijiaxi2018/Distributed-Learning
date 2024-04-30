import json
import os

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return None

def metric_video(working_path, metric_clip_path, video_filenames, clips_per_video):
    metric_clip_result = load_json_file(metric_clip_path)
    class_indexes = list(metric_clip_result.keys())

    curr_sep = 0
    for i in range(len(video_filenames)):
        video_result = {}

        filename = video_filenames[i]
        num_clips = clips_per_video[i]
        for class_idx in class_indexes:
            video_result[class_idx] = metric_clip_result[class_idx][curr_sep:curr_sep+num_clips]
        
        base_name = os.path.splitext(os.path.basename(filename))[0]
        with open(os.path.join(working_path, f"{base_name}_Result.json"), 'w') as file:
            json.dump(video_result, file, indent=4)
        
        curr_sep += num_clips

if __name__ == "__main__":
    metric_video("./", "./Clip_Result_I5.json", ['Video1_clip1.mp4', 'Video1_clip2.mp4', 'Video2_clip1.mp4', 'Video2_clip2.mp4'], [3, 3, 3, 3])
    