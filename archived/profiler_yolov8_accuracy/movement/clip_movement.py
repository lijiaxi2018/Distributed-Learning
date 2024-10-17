import json
import os
import numpy as np

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return None

def calculate_movement_result(one_clip_result, fps):
	pair_single_results = []
	for pair_results in one_clip_result:
		if len(pair_results) == 0:
			pair_single_results.append(0)
		else:
			pair_single_results.append(np.mean(np.array(pair_results)))

	clip_single_result = np.mean(np.array(pair_single_results))
	return clip_single_result

def clip_movement(working_folder, results_filename, interval, fpss, class_indexes=[2]):
    movement_frame_results = {}
    for fps in fpss:
        movement_frame_results[fps] = load_json_file(os.path.join(working_folder, f"Movement_Dup_I{interval}_F{fps}.json"))

    split_result = load_json_file(os.path.join(working_folder, f"Split_Result.json"))
    clip_number = split_result['clip_number']
    frames_per_clip = split_result['frames_per_clip']

    clips_result = {}
    for class_idx in class_indexes:
        class_result = []
        
        for i in range(clip_number):
            clip_result = {}
            clip_result["seq"] = i
            clip_result["frame_count"] = frames_per_clip
            clip_result["movement"] = {}

            for fps in fpss:
                fps_movement_frame_results = movement_frame_results[fps]
                fps_movement_frame_results_keys = sorted(list(fps_movement_frame_results.keys()))

                fps_result = calculate_movement_result(fps_movement_frame_results[fps_movement_frame_results_keys[i]], fps)
                clip_result["movement"][str(fps)] = fps_result

            class_result.append(clip_result)
            
        clips_result[class_idx] = class_result

    with open(os.path.join(working_folder, results_filename), 'w') as file:
        json.dump(clips_result, file, indent=4)