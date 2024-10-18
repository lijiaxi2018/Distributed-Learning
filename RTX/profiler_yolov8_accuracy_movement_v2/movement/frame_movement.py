import json
import os
import math

THRESHOLD = 0.4

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return None

def parse_detection_file(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as file:
        detections = [line.strip().split() for line in file]
    return [(int(d[0]), [float(x) for x in d[1:]]) for d in detections]

def calculate_point_distance(pa, pb):
    return math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_matched_iou(file_path1, file_path2, categories, iou_threshold=THRESHOLD):
    # Parse the detection files
    detections1 = parse_detection_file(file_path1)
    detections2 = parse_detection_file(file_path2)

    # Parse and filter detections
    filtered_detections1 = [d for d in detections1 if d[0] in categories]
    filtered_detections2 = [d for d in detections2 if d[0] in categories]

    matched = [False] * len(filtered_detections2)

    ious = []
    for det1 in filtered_detections1:
        category1, box1 = det1

        curr_matched_idx = -1
        curr_iou = -1.
        for i, det2 in enumerate(filtered_detections2):
            category2, box2 = det2

            if category1 == category2 and not matched[i]:
                box1_converted = [box1[0], box1[1], box1[2], box1[3]]
                box2_converted = [box2[0], box2[1], box2[2], box2[3]]
                
                iou = calculate_iou(box1_converted, box2_converted)
                if iou > iou_threshold and iou > curr_iou:
                    curr_matched_idx = i
                    curr_iou = iou
                
        if curr_matched_idx != -1:
            matched[curr_matched_idx] = True
            ious.append(curr_iou)

    return ious

def calculate_matched_key_point_distance(file_path1, file_path2, categories, iou_threshold=THRESHOLD):
    # Parse the detection files
    detections1 = parse_detection_file(file_path1)
    detections2 = parse_detection_file(file_path2)

    # Parse and filter detections
    filtered_detections1 = [d for d in detections1 if d[0] in categories]
    filtered_detections2 = [d for d in detections2 if d[0] in categories]

    matched = [False] * len(filtered_detections2)

    key_point_movements = []
    for det1 in filtered_detections1:
        category1, box1 = det1

        curr_matched_idx = -1
        curr_iou = -1.
        for i, det2 in enumerate(filtered_detections2):
            category2, box2 = det2

            if category1 == category2 and not matched[i]:
                box1_converted = [box1[0], box1[1], box1[2], box1[3]]
                box2_converted = [box2[0], box2[1], box2[2], box2[3]]
                
                iou = calculate_iou(box1_converted, box2_converted)
                if iou > iou_threshold and iou > curr_iou:
                    curr_matched_idx = i
                    curr_iou = iou
                
        if curr_matched_idx != -1:
            matched[curr_matched_idx] = True

            box1_left_top = (box1[0] - 0.5 * box1[2], box1[1] - 0.5 * box1[3])
            box1_right_top = (box1[0] + 0.5 * box1[2], box1[1] - 0.5 * box1[3])
            box1_left_bottom = (box1[0] - 0.5 * box1[2], box1[1] + 0.5 * box1[3])
            box1_right_bottom = (box1[0] + 0.5 * box1[2], box1[1] + 0.5 * box1[3])

            box2_left_top = (box2[0] - 0.5 * box2[2], box2[1] - 0.5 * box2[3])
            box2_right_top = (box2[0] + 0.5 * box2[2], box2[1] - 0.5 * box2[3])
            box2_left_bottom = (box2[0] - 0.5 * box2[2], box2[1] + 0.5 * box2[3])
            box2_right_bottom = (box2[0] + 0.5 * box2[2], box2[1] + 0.5 * box2[3])

            left_top_distance = calculate_point_distance(box1_left_top, box2_left_top)
            right_top_distance = calculate_point_distance(box1_right_top, box2_right_top)
            left_bottom_distance = calculate_point_distance(box1_left_bottom, box2_left_bottom)
            right_bottom_distance = calculate_point_distance(box1_right_bottom, box2_right_bottom)

            key_point_movements.append((left_top_distance, right_top_distance, left_bottom_distance, right_bottom_distance))

    return key_point_movements

def calculate_matched_amount(file_path1, file_path2, categories, iou_threshold=THRESHOLD):
    # Parse the detection files
    detections1 = parse_detection_file(file_path1)
    detections2 = parse_detection_file(file_path2)

    # Parse and filter detections
    filtered_detections1 = [d for d in detections1 if d[0] in categories]
    filtered_detections2 = [d for d in detections2 if d[0] in categories]

    matched = [False] * len(filtered_detections2)

    ious = []
    for det1 in filtered_detections1:
        category1, box1 = det1

        curr_matched_idx = -1
        curr_iou = -1.
        for i, det2 in enumerate(filtered_detections2):
            category2, box2 = det2

            if category1 == category2 and not matched[i]:
                box1_converted = [box1[0], box1[1], box1[2], box1[3]]
                box2_converted = [box2[0], box2[1], box2[2], box2[3]]
                
                iou = calculate_iou(box1_converted, box2_converted)
                if iou > iou_threshold and iou > curr_iou:
                    curr_matched_idx = i
                    curr_iou = iou
                
        if curr_matched_idx != -1:
            matched[curr_matched_idx] = True
            ious.append(curr_iou)

    return len(ious)

def calculate_matched_confidence(file_path1, file_path2, categories, iou_threshold=THRESHOLD):
    # Parse the detection files
    detections1 = parse_detection_file(file_path1)
    detections2 = parse_detection_file(file_path2)

    # Parse and filter detections
    filtered_detections1 = [d for d in detections1 if d[0] in categories]
    filtered_detections2 = [d for d in detections2 if d[0] in categories]

    matched = [False] * len(filtered_detections2)

    confidences = []
    for det1 in filtered_detections1:
        category1, box1 = det1

        curr_matched_idx = -1
        curr_iou = -1.
        for i, det2 in enumerate(filtered_detections2):
            category2, box2 = det2

            if category1 == category2 and not matched[i]:
                box1_converted = [box1[0], box1[1], box1[2], box1[3]]
                box2_converted = [box2[0], box2[1], box2[2], box2[3]]
                
                iou = calculate_iou(box1_converted, box2_converted)
                if iou > iou_threshold and iou > curr_iou:
                    curr_matched_idx = i
                    curr_iou = iou
                
        if curr_matched_idx != -1:
            matched[curr_matched_idx] = True
            average_confidence = 0.5 * (box1[4] + box2[4])
            confidences.append(average_confidence)

    return confidences

def calculate_movement_data(file_path1, file_path2, categories):
    key_point_distance = calculate_matched_key_point_distance(file_path1, file_path2, categories)
    amount = calculate_matched_amount(file_path1, file_path2, categories)
    confidence = calculate_matched_confidence(file_path1, file_path2, categories)
    iou = calculate_matched_iou(file_path1, file_path2, categories)

    return [key_point_distance, amount, confidence, iou]


def frame_movement(working_folder, interval, fps_target, categories):
    sampled_frames_info = load_json_file(os.path.join(working_folder, f"Frame_Dup_I{interval}_F{fps_target}.json"))

    movement_data_path = os.path.join(working_folder, f"Movement_Dup_I{interval}_F{fps_target}.json")
    clip_names = sorted(list(sampled_frames_info.keys()))

    movement_data = {}
    for j in range(len(clip_names)):
        clip_name = clip_names[j]

        movement_data[clip_name] = []
        for i in range(len(sampled_frames_info[clip_name]) - 1):
            curr_path = os.path.join(working_folder, f"Label_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_name][i])
            next_path = os.path.join(working_folder, f"Label_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_name][i+1])
            curr_path = curr_path[0 : curr_path.index('.')] + '.txt'
            next_path = next_path[0 : next_path.index('.')] + '.txt'

            movement_data[clip_name].append(calculate_movement_data(curr_path, next_path, categories))
        
        if j > 0:
            curr_clip_first_path = os.path.join(working_folder, f"Label_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_names[j]][0])
            prev_clip_last_path = os.path.join(working_folder, f"Label_Dup_I{interval}_F{fps_target}", sampled_frames_info[clip_names[j-1]][-1])
            curr_clip_first_path = curr_clip_first_path[0 : curr_clip_first_path.index('.')] + '.txt'
            prev_clip_last_path = prev_clip_last_path[0 : prev_clip_last_path.index('.')] + '.txt'

            movement_data[clip_name].append(calculate_movement_data(curr_clip_first_path, prev_clip_last_path, categories))
        
    # Handle first frame when FPS = 1
    if fps_target == 1:
        movement_data[clip_names[0]] = movement_data[clip_names[1]].copy()

    with open(movement_data_path, 'w') as file:
        json.dump(movement_data, file, indent=4)
    