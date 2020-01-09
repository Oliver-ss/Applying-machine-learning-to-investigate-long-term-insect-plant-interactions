'''*************************************************************************
	> File Name: get_mAP.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 16 Sep 2019 11:29:36 PM EDT
 ************************************************************************'''
import numpy as np
import os
import json

INDEX = {
        'margin': 0, 'interior': 1, 'skel': 2, 'stipp': 3,
        'blotch': 4, 'serp': 5, 'scrap': 6, 'normmar': 7, 'normint': 8, 'undef': 9,
        }

LABEL = ['margin', 'interior', 'skel', 'stipp', 'blotch', 'serp', 'scrap', 'normmar', 'normint', 'undef']

# make key of data dict be labels
def parse_rect(filename, name):
    with open(filename) as f:
        data = json.load(f)
    if name == 'gt':
        objects = {0: {}, 1: {}, 2: {}, 3: {}}
    elif name == 'pred':
        objects = {0: [], 1: [], 2: [], 3: []}
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
                objects[obj['label']].append(obj)
            else:
                print('wrong mode to parse rect')
                return
    return objects

class Eval_mAP:
    def __init__(self, num_classes, gt_file, pred_file):
        self.num_classes = num_classes
        self.gt = parse_rect(gt_file, 'gt')
        self.pred = parse_rect(pred_file, 'pred')

    def get_result(self, class_index, ovthresh=0.5):
        gt_rects = self.gt[class_index] # a dict, keys are img names
        pred_rects = self.pred[class_index] # a list of all predicted bboxes
        #npos = 0
        #for r in gt_rects.values():
        #    npos += len(r)
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
                uni_new = np.minimum(uni, (bb[2]-bb[0]) * (bb[3] - bb[1]),
                          (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]))
                overlaps = inters / uni_new
                #overlaps = inters / uni

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    #else:
                    #    fp[d] = 1.
                    else:
                        tp[d] = 1.
                        npos += 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.get_ap(rec, prec)
        return ap

    def get_ap(self, rec, prec, use_07_metric=True):
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pred", type=str)
    parser.add_argument("gt", type=str)
    args = parser.parse_args()
    #print(parse_rect("../../Data/Labels/label_test_new.json", 'gt'))
    #print(parse_rect("../../config/SSD-sgd/train_log/test/predictions.json", 'pred'))
    eval_Damage = Eval_mAP(3, args.gt, args.pred)
    print(eval_Damage.get_result(0))
    print(eval_Damage.get_result(1))
    #print(eval_Damage.get_result(2))

