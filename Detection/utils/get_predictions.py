'''*************************************************************************
	> File Name: get_predictions.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 16 Sep 2019 09:30:31 PM EDT
 ************************************************************************'''
import os
import torch
from torch.autograd import Variable
from model import build_ssd
from common import config
import cv2
import numpy as np
from data import BaseTransform
import json
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore')

# seven colors of rainbow, expressed in RGB
COLORS = [(255, 0, 0), (255, 165, 0), (255, 255, 0),
        (0, 255, 0), (0, 127, 255), (0, 0, 255), (139, 0, 255)]

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABELS = ['margin', 'interior', 'skel', 'stipp', 'blotch', 'serp', 'background']

def predict(frames, transform, net, tile_number):
    # tile_number: how many tiles has been computed
    height, width = frames.shape[1:3]
    x = []
    for i in range(frames.shape[0]):
        x.append(transform(frames[i])[0])
    x = np.stack(x, axis=0)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    with torch.no_grad():
        y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    bbox = []
    for k in range(detections.size(0)):
        # skip background class
        for i in range(1, detections.size(1)):
            j = 0
            while detections[k, i, j, 0] >= 0.1:
                pt = (detections[k, i, j, 1:] * scale).cpu().numpy()
                bbox.append({
                    'score': float(detections[k, i, j, 0].cpu().numpy()),
                    'tile': k + tile_number,  # store the tile index to compute offset of bbox
                    'index': i-1,  # class index
                    'bbox': pt.tolist()  # [xs, ys, se, ye]
                })
                j += 1
    return bbox


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  #bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(dets[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def test_img(net_filepath, img_folder, tile, batch_size, skip=300):
    # load net
    num_classes = config.num_classes
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(net_filepath, map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    # load data
    transform = BaseTransform()
    with open(img_folder) as f:
        labels = json.load(f)
    img_names = list(labels.keys())
    data = {}
    for k in tqdm(range(len(img_names))):
        dets = []
        overlap = int(0.15 * tile)
        img_file = img_names[k]
        img = cv2.imread(img_file)
        img = img[:,:,::-1]

        # skip the image boundary
        h, w, c = img.shape
        img = img[skip:h-skip, :, :]

        h, w, c = img.shape
        imgs = []
        stride = tile - overlap
        h_num = (h - tile) // stride + 1
        w_num = (w - tile) // stride + 1
        for i in range(h_num):
            for j in range(w_num):
                # split the image into tiles
                x = img[i * stride:(i * stride + tile), j * stride:(j * stride + tile), :]
                imgs.append(x)
        # stack tiles
        input = np.stack(imgs, axis=0)
        bbox = []
        for i in range((input.shape[0] - 1) // batch_size + 1):
            bbox += predict(input[batch_size * i:batch_size * (i + 1)], transform, net, i * batch_size)

        # TODO：
        for i in range(len(bbox)):
            xs, ys, xe, ye = bbox[i]['bbox'][:]
            tile_ind = bbox[i]['tile']
            class_index = bbox[i]['index']
            xdiff = xe - xs
            ydiff = ye - ys
            row_num = tile_ind // w_num
            col_num = tile_ind % w_num
            # compute offset
            ys += row_num * stride + skip
            xs += col_num * stride
            xe = xs + xdiff
            ye = ys + ydiff
            score = bbox[i]['score']
            dets.append([xs, ys, xe, ye, score, class_index])
        #print(len(dets))
        keep = nms(dets, 0.2)
        #print(len(keep))
        data[img_file] = keep
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default="train_log/model/10000.pth")
    parser.add_argument('--label', default="../../Data/Labels/test_full.json")
    parser.add_argument('--save', default="preditions.json")
    parser.add_argument('--tile', type=int, default=300)
    args = parser.parse_args()

    net_filepath = args.model
    test_img_file = args.label
    tile = args.tile
    batch_size = 8
    skip = 500
    data = test_img(net_filepath, test_img_file, tile, batch_size, skip)

    if not os.path.exists('train_log/test'):
        os.makedirs('train_log/test')
    with open('train_log/test/' + args.save, 'w') as f:
        json.dump(data, f)
