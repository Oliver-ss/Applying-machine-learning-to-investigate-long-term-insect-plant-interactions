'''*************************************************************************
	> File Name: get_recall_precision.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Fri 20 Sep 2019 12:48:10 PM EDT
 ************************************************************************'''
import json
import numpy as np
import csv
import os

INDEX = {
        'margin': 0, 'interior': 1, 'skel': 2, 'stipp': 3,
        'blotch': 4, 'serp': 5, 'scrap': 6, 'normmar': 7, 'normint': 8, 'undef': 9,
        }

LABEL = ['margin', 'interior', 'skel', 'stipp', 'blotch', 'serp', 'scrap', 'normmar', 'normint', 'undef']

# make key of data dict be labels
def parse_rect(filename, name, threshold=None):
    with open(filename) as f:
        data = json.load(f)
    if name == 'gt':
        objects = {0: {}, 1: {}, 2: {}}
    elif name == 'pred':
        objects = {0: [], 1: [], 2: []}
    for k in data.keys():
        for box in data[k]:
            if name == 'gt':
                label = INDEX[box['label']]
                if label > 6: # omit last 3 classes
                    continue
                obj = {}
                obj['name'] = k
                x, y, w, h = box['rect']
                obj['bbox'] = [x, y, x+w, y+h]
                if w < 20 or h < 20:
                    obj['difficult'] = True
                else:
                    obj['difficult'] = False
                obj['label'] = min(2, label) # conclue other classes as 1 class
                if obj['name'] not in objects[obj['label']].keys():
                    objects[obj['label']][obj['name']] = []
                objects[obj['label']][obj['name']].append(obj)
            elif name == 'pred':
                obj = {}
                obj['label'] = int(box[5])
                obj['name'] = k
                obj['bbox'] = box[:4]
                obj['score'] = box[4]
                if obj['label'] > 1:
                    continue
                if obj['score'] > threshold[obj['label']]:
                    objects[obj['label']].append(obj)
            else:
                print('wrong mode to parse rect')
                return
    return objects


def get_class_recall_precision(gt, pred, class_index, ovthresh=0.5):
    gt_rects = gt[class_index]  # a dict, keys are img names
    pred_rects = pred[class_index]  # a list of all predicted bboxes
    if not pred_rects:
        return 0, 0
    class_recs = {}
    npos = 0
    for imagename in gt_rects.keys():
        R = gt_rects[imagename]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    confidence = np.array([x['score'] for x in pred_rects])
    image_ids = [x['name'] for x in pred_rects]
    BB = np.array([x['bbox'] for x in pred_rects])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    #print(sorted_ind)
    #print(BB)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        if image_ids[d] not in class_recs.keys():
            fp[d] = 1
            continue
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) *
                   (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec[-1], prec[-1]


def get_recall_precision(gt_filepath, pred_filepath, class_threshold, save_filepath):
    gt = parse_rect(gt_filepath, 'gt')
    pred = parse_rect(pred_filepath, 'pred', class_threshold)
    recall = ['recall']
    precision = ['precision']
    for i in range(len(class_threshold)):
        rec, prec = get_class_recall_precision(gt, pred, i)
        recall.append(np.round(rec, 2))
        precision.append(np.round(prec, 2))

    with open(save_filepath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'margin', 'interior'])
        writer.writerow(['threshold'] + class_threshold)
        writer.writerow(recall)
        writer.writerow(precision)


if __name__ == '__main__':
    gt_filepath = '../../Data/Labels/label_val_new.json'
    pred_filepath = 'train_log/test/validation_400.json'
    save_filepath = 'train_log/threshold_R_P.csv'
    for i in np.arange(0.95, 0, -0.05):
        class_threshold = [np.round(i, 2), 0]
        get_recall_precision(gt_filepath, pred_filepath, class_threshold, save_filepath)
    for i in np.arange(0.95, 0, -0.05):
        class_threshold = [0, np.round(i, 2)]
        get_recall_precision(gt_filepath, pred_filepath, class_threshold, save_filepath)
