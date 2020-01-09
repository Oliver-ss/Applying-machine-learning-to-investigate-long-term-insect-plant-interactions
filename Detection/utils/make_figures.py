'''*************************************************************************
	> File Name: visualize.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Wed 18 Sep 2019 12:46:13 PM EDT
 ************************************************************************'''
import cv2
import json
import numpy as np
import os
from tqdm import tqdm
import rpack

COLORS = ([255,0,0], [0,255,0], [0,0,255])

FONT = cv2.FONT_HERSHEY_SIMPLEX

INDEX = {
        'margin': 0, 'interior': 1, 'skel': 2, 'stipp': 3,
        'blotch': 4, 'serp': 5, 'scrap': 6, 'normmar': 7, 'normint': 8, 'undef': 9,
        }

LABEL = ['margin', 'interior', 'skel', 'stipp', 'blotch', 'serp', 'scrap', 'normmar', 'normint', 'undef']


def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            img = cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                img = cv2.line(img,s,e,color,thickness)
            i+=1
    return img

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        img = drawline(img,s,e,color,thickness,style)
    return img

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    img = drawpoly(img,pts,color,thickness,style)
    return img

def get_tp_fp_fn(pred, gt, class_index, img_name, ovthresh=0.5):
    gt = np.array(gt[class_index])
    pred = np.array(pred[class_index])
    tp, fp, fn = [], [], []
    if_det = [False] * gt.shape[0]
    for box in pred:
        ixmin = np.maximum(gt[:, 0], box[0])
        iymin = np.maximum(gt[:, 1], box[1])
        ixmax = np.minimum(gt[:, 2], box[2])
        iymax = np.minimum(gt[:, 3], box[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = np.minimum((box[2] - box[0]) * (box[3] - box[1]), (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            #from IPython import embed
            #embed()
            tp.append((min(box[0], gt[jmax][0]), min(box[1], gt[jmax][1]), max(box[2], gt[jmax][2]), max(box[3], gt[jmax][3]), img_name))
            if_det[jmax] = True
        else:
            fp.append((box[0], box[1], box[2], box[3], img_name))

    for i in range(len(if_det)):
        if not if_det[i]:
            fn.append((gt[i][0], gt[i][1], gt[i][2], gt[i][3], img_name))

    #from IPython import embed
    #embed()
    # delete repeating bboxes
    tp = list(set(tp))
    fp = list(set(fp))
    fn = list(set(fn))
    return tp, fp, fn

def get_packed(boxes):
    sizes = [(int(box[2] - box[0]), int(box[3] - box[1])) for box in boxes]
    heights = [int(box[3] - box[1]) for box in boxes]
    #arr = np.array(sizes, dtype=[('w', int), ('h', int)])
    arr = np.array(heights)

    sorted_ind = np.argsort(arr)
    sorted_ind = list(sorted_ind[::-1])
    sorted_sizes = sorted(sizes, key = lambda x: x[1], reverse=True)
    #sorted_sizes = np.sort(arr, order='h')
    #sorted_sizes = sorted_sizes[::-1]
    #sorted_sizes = list(sorted_sizes)
    #from IPython import embed
    #embed()
    positions = rpack.pack(sorted_sizes)
    shape = rpack.enclosing_size(sorted_sizes, positions)
    new_img = np.zeros((shape[1], shape[0], 3))
    for i in range(len(positions)):
        x, y = positions[i]
        w, h = sorted_sizes[i]
        box = boxes[sorted_ind[i]]
        img = cv2.imread(box[4])
        img = np.array(img)
        new_img[y:y+h, x:x+w ,:] = img[int(box[1]):int(box[1]) + h, int(box[0]):int(box[0] + w)]
    return new_img

def save_img(pred, gt, save_folder='train_log/test_figures/', thres=(0.4, 0.6)):
    img_names = list(gt.keys())
    tp0, fp0, fn0, tp1, fp1, fn1 = [], [], [], [], [], []

    for i in tqdm(range(len(img_names))):
        pred_new = {0: [], 1: []}
        gt_new = {0: [], 1: []}
        img_file = img_names[i]
        img = cv2.imread(img_file)
        img = img[:,:,::-1]
        for box in pred[img_file]:
            xs, ys, xe, ye = box[:4]
            confidence = box[4]
            label = int(box[5])
            if label > 1:
                continue
            if confidence > thres[label]:
                pred_new[label].append(box)
                img = cv2.rectangle(img, (int(xs), int(ys)), (int(xe), int(ye)), COLORS[label], 4)
                #img = cv2.putText(img, str(np.round(confidence*10,2)), (int(xs), int(ys)),
                #        FONT, 2, COLORS[label], 2, cv2.LINE_AA)

        for box in gt[img_file]:
            xs, ys, w, h = box['rect']
            xe, ye = xs + w, ys + h
            label = INDEX[box['label']]
            if label > 1:
                continue
            #else:
            #    label = min(label, 2)
            gt_new[label].append([xs, ys, xe, ye, label])
            img = drawrect(img, (int(xs), int(ys)), (int(xe), int(ye)), COLORS[label], 4, style='dash')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = save_folder + img_file.split('/')[-1]
        cv2.imwrite(name, img)

        tp0_, fp0_, fn0_ = get_tp_fp_fn(pred_new, gt_new, 0, name)
        tp1_, fp1_, fn1_ = get_tp_fp_fn(pred_new, gt_new, 1, name)

        tp0 += tp0_
        fp0 += fp0_
        fn0 += fn0_
        tp1 += tp1_
        fp1 += fp1_
        fn1 += fn1_

    tp0_img = get_packed(tp0)
    cv2.imwrite(save_folder + 'tp0.jpg', tp0_img)
    fp0_img = get_packed(fp0)
    cv2.imwrite(save_folder + 'fp0.jpg', fp0_img)
    fn0_img = get_packed(fn0)
    cv2.imwrite(save_folder + 'fn0.jpg', fn0_img)
    tp1_img = get_packed(tp1)
    cv2.imwrite(save_folder + 'tp1.jpg', tp1_img)
    fp1_img = get_packed(fp1)
    cv2.imwrite(save_folder + 'fp1.jpg', fp1_img)
    fn1_img = get_packed(fn1)
    cv2.imwrite(save_folder + 'fn1.jpg', fn1_img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("thres1", type=float)
    parser.add_argument("thres2", type=float)
    #parser.add_argument("thres3", type=float)
    parser.add_argument("pred", type=str)
    args = parser.parse_args()
    thres = (args.thres1, args.thres2)

    with open(args.pred) as f:
        pred = json.load(f)

    with open('../../Data/Labels/test_full.json') as f:
        gt = json.load(f)

    save_folder = 'train_log/figures/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_img(pred, gt, save_folder, thres)

