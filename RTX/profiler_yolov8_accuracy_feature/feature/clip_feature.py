import json
import os
import numpy as np
from .frame_feature import feature_number_to_type

def load_json_file(file_path):
	try:
		with open(file_path, 'r') as file:
			data = json.load(file)
		return data
	
	except Exception as e:
		print(f"An error occurred while loading the JSON file: {e}")
		return None

def combine_frame_result_low_level_feature(one_clip_result, fps):
	return np.average(np.array(one_clip_result))

def clip_feature(working_folder, results_filename, interval, fpss, feature_number, class_indexes=[2]):
	feature_frame_results = {}
	for fps in fpss:
		feature_frame_results[fps] = load_json_file(os.path.join(working_folder, f"Feature_{feature_number_to_type(feature_number)}_Dup_I{interval}_F{fps}.json"))

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
			clip_result["feature"] = {}

			for fps in fpss:
				fps_feature_frame_results = feature_frame_results[fps]
				fps_feature_frame_results_keys = sorted(list(fps_feature_frame_results.keys()))

				fps_result = combine_frame_result_low_level_feature(fps_feature_frame_results[fps_feature_frame_results_keys[i]], fps)

				clip_result["feature"][str(fps)] = fps_result

			class_result.append(clip_result)
			
		clips_result[class_idx] = class_result

	with open(os.path.join(working_folder, results_filename), 'w') as file:
		json.dump(clips_result, file, indent=4)