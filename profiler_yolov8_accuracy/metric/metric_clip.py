import json
import numpy as np

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return None

def metric_clip(frames_per_clip, fpss, prefix, results_path):
    class_indexes = list(load_json_file(f"{prefix}{fpss[0]}.json").keys())
    
    metric_results = {}
    for fps in fpss:
        metric_results[fps] = load_json_file(f"{prefix}{fps}.json")

    clips_result = {}
    for class_idx in class_indexes:
        class_result = []
        
        idx = 0
        count = 0
        for frame_count in frames_per_clip:
            clip_result = {}
            clip_result["seq"] = count
            clip_result["frame_count"] = frame_count
            clip_result["metric"] = {}

            for fps in fpss:
                fps_result = {}

                single_metric_result = metric_results[fps]
                tp_clip = int(np.sum(np.array(single_metric_result[class_idx]["TP"][idx:idx+frame_count])))
                rp_clip = int(np.sum(np.array(single_metric_result[class_idx]["RP"][idx:idx+frame_count])))
                pp_clip = int(np.sum(np.array(single_metric_result[class_idx]["PP"][idx:idx+frame_count]))
)
                fps_result["TP"] = tp_clip
                fps_result["RP"] = rp_clip
                fps_result["PP"] = pp_clip

                fps_result["Precision"] = tp_clip / pp_clip
                fps_result["Recall"] = tp_clip / rp_clip
                fps_result["F1"] = 2 / ( (1 / (tp_clip / pp_clip)) + (1 / (tp_clip / rp_clip)) )

                clip_result["metric"][str(fps)] = fps_result

            class_result.append(clip_result)

            idx += frame_count
            count += 1
            
        clips_result[class_idx] = class_result
    
    with open(results_path, 'w') as file:
        json.dump(clips_result, file, indent=4)
   
if __name__ == "__main__":
    metric_clip([149, 149, 149, 149, 150, 150, 150, 150], [5, 10], "./Label_Dup_I5_F", "./results.json")