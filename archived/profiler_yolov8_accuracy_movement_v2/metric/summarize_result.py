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

def summarize_result(working_folder, results_filename, interval, fpss, class_indexes=[2]):
    metric_results = {}
    for fps in fpss:
        metric_results[fps] = load_json_file(os.path.join(working_folder, f"Label_Dup_I{interval}_F{fps}.json"))
    
    split_result = load_json_file(os.path.join(working_folder, f"Split_Result.json"))
    clip_number = split_result['clip_number']
    frames_per_clip = split_result['frames_per_clip']

    clips_result = {}
    for class_idx in class_indexes:
        class_result = []
        
        idx = 0
        for i in range(clip_number):
            clip_result = {}
            clip_result["seq"] = i
            clip_result["frame_count"] = frames_per_clip
            clip_result["metric"] = {}

            for fps in fpss:
                fps_result = {}

                single_metric_result = metric_results[fps]
                tp_clip = int(np.sum(np.array(single_metric_result[str(class_idx)]["TP"][idx:idx+frames_per_clip])))
                rp_clip = int(np.sum(np.array(single_metric_result[str(class_idx)]["RP"][idx:idx+frames_per_clip])))
                pp_clip = int(np.sum(np.array(single_metric_result[str(class_idx)]["PP"][idx:idx+frames_per_clip])))
                fps_result["TP"] = tp_clip
                fps_result["RP"] = rp_clip
                fps_result["PP"] = pp_clip

                # Handle division by zero
                prec = 1.0
                if pp_clip != 0:
                    prec = tp_clip / pp_clip
                fps_result["Precision"] = prec
                
                # Handle division by zero
                reca = 1.0
                if rp_clip != 0:
                    reca = tp_clip / rp_clip
                fps_result["Recall"] = reca

                # Handle division by zero
                f1 = 1.0
                if prec != 0 and reca != 0:
                    f1 = 2 / ( (1 / prec) + (1 / reca) )
                fps_result["F1"] = f1

                clip_result["metric"][str(fps)] = fps_result

            class_result.append(clip_result)

            idx += frames_per_clip
            
        clips_result[class_idx] = class_result
    
    with open(os.path.join(working_folder, results_filename), 'w') as file:
        json.dump(clips_result, file, indent=4)