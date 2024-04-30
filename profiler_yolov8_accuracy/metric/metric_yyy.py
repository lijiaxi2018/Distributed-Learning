
'''
Given all the test samples (.png, .jpg, .jpeg) inside the image directory, all the corresponding real labels (.txt) inside 
the real directory, and all the corresponding predicted labels (.txt) inside pred directory (can be specified by the user),
return the precision, recall, and F1 score for each class.

It assumes the object categories are in YOLO format, real labels to be in YOLO format, and predicted labels in YOLO format. 
The default threshold is set to be 0.5 and can be specified by the user.

Example usage: python metric_yyy.py pred 0.5
File requirements: image, real, and pred directories are in the current directory

Reference 
https://github.com/phananh1010/360-object-detection-annotation/blob/master/annotator.py

'''

import cv2
import json
import os
import sys
import numpy as np
from shapely.geometry import Polygon

CLASSES = ['People']

IMAGE_ENDINGS = (".png", ".jpg", ".jpeg")
def getAllImages(path):
    images_filenames = []
    for f in os.listdir(path):
        if any(f.endswith(ending) for ending in IMAGE_ENDINGS):
            images_filenames.append(f)
    return images_filenames

def read_bb_yolo(bb_filename):
    bbs = []

    if os.path.exists(bb_filename) == False:
        return bbs

    f = open(bb_filename, "r")
    lines = f.readlines()
    for line in lines:
        line = line[0:len(line)-1]
        splitted = line.split(' ')

        bb = [int(splitted[0]), float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])]
        if len(splitted) >= 6:
            bb.append(float(splitted[5]))
        bbs.append(bb)
    return bbs

def bb_xywh2xyxyxyxy(bb_xywh):
    bbs_xyxyxyxy = []

    for bb in bb_xywh:
        x1, y1 = bb[1] - 0.5 * bb[3], bb[2] - 0.5 * bb[4]
        x2, y2 = bb[1] + 0.5 * bb[3], bb[2] - 0.5 * bb[4]
        x3, y3 = bb[1] - 0.5 * bb[3], bb[2] + 0.5 * bb[4]
        x4, y4 = bb[1] + 0.5 * bb[3], bb[2] + 0.5 * bb[4]

        bb_xyxyxyxy = [bb[0], x1, y1, x2, y2, x3, y3, x4, y4]
        if len(bb) >= 6:
            bb_xyxyxyxy.append(bb[5])
        
        bbs_xyxyxyxy.append(bb_xyxyxyxy)
    return bbs_xyxyxyxy

def bb_scale(bbs, width, height):
    bbs_scaled = []

    for bb in bbs:
        x1, x2, x3, x4 = bb[1] * width, bb[3] * width, bb[5] * width, bb[7] * width
        y1, y2, y3, y4 = bb[2] * height, bb[4] * height, bb[6] * height, bb[8] * height

        bb_scaled = [bb[0], x1, y1, x2, y2, x3, y3, x4, y4]
        if len(bb) >= 10:
            bb_scaled.append(bb[9])
        
        bbs_scaled.append(bb_scaled)
    return bbs_scaled

def single_image_confusion_matrix(bb_real_8, bb_pred_8, class_idx, iou_threshold=0.5):
    tp = 0
    real_p = 0
    pred_p = 0

    bb_real_class = [bb for bb in bb_real_8 if bb[0] == class_idx]
    bb_pred_class = [bb for bb in bb_pred_8 if bb[0] == class_idx]

    correct_real = []
    correct_pred = []
    for j in range(len(bb_real_class)):
        for i in range(len(bb_pred_class)):
            if i in correct_pred:
                continue
            
            bbr = bb_real_class[j]
            bbp = bb_pred_class[i]
            
            pol_r = [[bbr[1], bbr[2]], [bbr[3], bbr[4]], [bbr[7], bbr[8]], [bbr[5], bbr[6]]]
            pol_p = [[bbp[1], bbp[2]], [bbp[3], bbp[4]], [bbp[7], bbp[8]], [bbp[5], bbp[6]]]

            polygon_r_shape = Polygon(pol_r)
            polygon_p_shape = Polygon(pol_p)

            intersection = polygon_r_shape.intersection(polygon_p_shape).area
            union = polygon_r_shape.union(polygon_p_shape).area
            iou = intersection / union 

            if iou > iou_threshold:
                correct_real.append(j)
                correct_pred.append(i)
                break
        
    tp += len(correct_pred)
    real_p += len(bb_real_class)
    pred_p += len(bb_pred_class)
    assert (len(correct_real) == len(correct_pred))

    return tp, real_p, pred_p
        
def metric(image_directory, real_directory, pred_directory, results_path, threshold=0.5, class_indexes=[2]):
    results = {}
    image_filenames = getAllImages(image_directory)
    image_filenames = sorted(image_filenames)

    for class_idx in class_indexes:
        class_result = {}
        tps, real_ps, pred_ps = [], [], []
    
        for image_filename in image_filenames:
            print('Processing: ' + image_filename)

            image_path = image_directory + '\\' + image_filename
            bb_real_path = real_directory + '\\' + image_filename[0 : image_filename.index('.')] + '.txt'
            bb_pred_path = pred_directory + '\\' + image_filename[0 : image_filename.index('.')] + '.txt'

            src = cv2.imread(image_path)
            sphereH, sphereW, _ = map(int, src.shape)

            bb_real_raw = read_bb_yolo(bb_real_path)
            bb_real_8 = bb_xywh2xyxyxyxy(bb_real_raw)
            bb_real_8 = bb_scale(bb_real_8, sphereW, sphereH)

            bb_pred_raw = read_bb_yolo(bb_pred_path)
            bb_pred_8 = bb_xywh2xyxyxyxy(bb_pred_raw)
            bb_pred_8 = bb_scale(bb_pred_8, sphereW, sphereH)

            tp, real_p, pred_p = single_image_confusion_matrix(bb_real_8, bb_pred_8, class_idx, threshold)
            tps.append(tp)
            real_ps.append(real_p)
            pred_ps.append(pred_p)
        
        class_result["TP"] = tps
        class_result["RP"] = real_ps
        class_result["PP"] = pred_ps

        print(f"CLASS_IDX: {class_idx}")
        tp_sum = np.sum(np.array(tps))
        real_p_sum = np.sum(np.array(real_ps))
        pred_p_sum = np.sum(np.array(pred_ps))

        if pred_p_sum == 0:
            print('Precision: Divided by 0')
            class_result["Precision"] = -1
        else:
            print('Precision: ' + str(tp_sum) + '/' + str(pred_p_sum) + '=' + str(tp_sum / pred_p_sum))
            class_result["Precision"] = tp_sum / pred_p_sum
        
        if real_p_sum == 0:
            print('Recall: Divided by 0')
            class_result["Recall"] = -1
        else:
            print('Recall: ' + str(tp_sum) + '/' + str(real_p_sum) + '=' + str(tp_sum / real_p_sum) )
            class_result["Recall"] = tp_sum / real_p_sum
        
        if pred_p_sum != 0 and real_p_sum != 0:
            pr = tp_sum / pred_p_sum
            re = tp_sum / real_p_sum
            print('F1 Score: ' + str( 2 / ( (1 / pr) + (1 / re) ) ) )
            class_result["F1"] = 2 / ( (1 / pr) + (1 / re) )
        else:
            print('Either Precision or Recall: Divided by 0')
            class_result["F1"] = -1
        
        results[class_idx] = class_result
        
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    metric('Frame_All_I5_F3', 'Label_GT_I5_F3', 'Label_Dup_I5_F3', 'results.json')