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

def calculate_movement_result_v1(one_clip_result, fps):
	# Initialize lists to store the average results
	avg_lists = {
		'left_top': [],
		'right_top': [],
		'left_bottom': [],
		'right_bottom': [],
		'confidence': [],
		'amount': [],
		'iou': [],
	}

	for pair_results in one_clip_result:
		pair_amount = pair_results[1]
		if pair_amount == 0:
			continue

		# Calculate the sums using list comprehension
		distance_sums = [sum(pair_results[0][i][j] for i in range(pair_amount)) for j in range(4)]
		confidence_sum = sum(pair_results[2][i] for i in range(pair_amount))
		iou_sum = sum(pair_results[3][i] for i in range(pair_amount))

		# Convert sums to averages and append to the corresponding lists
		avg_lists['left_top'].append(distance_sums[0] / pair_amount / (1/fps))
		avg_lists['right_top'].append(distance_sums[1] / pair_amount / (1/fps))
		avg_lists['left_bottom'].append(distance_sums[2] / pair_amount / (1/fps))
		avg_lists['right_bottom'].append(distance_sums[3] / pair_amount / (1/fps))
		avg_lists['confidence'].append(confidence_sum / pair_amount)
		avg_lists['iou'].append(iou_sum / pair_amount)
		avg_lists['amount'].append(pair_amount)

	# Return default values if no valid data
	if not avg_lists['left_top']:
		return [-1., -1., -1., -1., -1., -1., -1.]

	# Calculate the average of each list and return the result
	return [np.mean(avg_lists[key]) for key in ['left_top', 'right_top', 'left_bottom', 'right_bottom', 'amount', 'confidence', 'iou']]


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

				fps_result = calculate_movement_result_v1(fps_movement_frame_results[fps_movement_frame_results_keys[i]], fps)
				if fps_result[4] == 0 and i != 0:
					fps_result = class_result[i - 1]["movement"][str(fps)]

				clip_result["movement"][str(fps)] = fps_result

			class_result.append(clip_result)
			
		clips_result[class_idx] = class_result

	with open(os.path.join(working_folder, results_filename), 'w') as file:
		json.dump(clips_result, file, indent=4)