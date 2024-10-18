import json
import os
from .edge_detection import compute_edge_difference
from .pixel_detection import compute_ssim_difference

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return None

def feature_number_to_type(feature_number):
    if feature_number == 0:
        return "Edge"
    if feature_number == 1:
        return "Pixel"
    
    return "None"

def frame_feature(working_folder, interval, fps_target, categories, feature_number, target_size):
    sampled_frames_info = load_json_file(os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}.json"))

    feature_data_path = os.path.join(working_folder, f"Feature_{feature_number_to_type(feature_number)}_Dup_I{interval}_F{fps_target}.json")
    clip_names = sorted(list(sampled_frames_info.keys()))

    feature_data = {}
    for j in range(len(clip_names)):
        clip_name = clip_names[j]

        feature_data[clip_name] = []
        for i in range(len(sampled_frames_info[clip_name]) - 1):
            curr_path = os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_name][i])
            next_path = os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_name][i+1])

            if feature_number == 0:
                feature_data[clip_name].append(compute_edge_difference(curr_path, next_path, target_size))
            if feature_number == 1:
                feature_data[clip_name].append(compute_ssim_difference(curr_path, next_path, target_size))
            
        if j > 0:
            curr_clip_first_path = os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_names[j]][0])
            prev_clip_last_path = os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_names[j-1]][-1])

            if feature_number == 0:
                feature_data[clip_name].append(compute_edge_difference(curr_clip_first_path, prev_clip_last_path, target_size))
            if feature_number == 1:
                feature_data[clip_name].append(compute_ssim_difference(curr_clip_first_path, prev_clip_last_path, target_size))

    # Handle first frame when FPS = 1
    if fps_target == 1:
        feature_data[clip_names[0]] = feature_data[clip_names[1]].copy()

    with open(feature_data_path, 'w') as file:
        json.dump(feature_data, file, indent=4)
    