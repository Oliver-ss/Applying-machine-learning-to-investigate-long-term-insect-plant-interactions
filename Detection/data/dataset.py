'''*************************************************************************
	> File Name: dataset.py
	> Author: Your Name (edit this in the init.vim file)
	> Mail: Your Mail@megvii.com
	> Created Time: Tue Sep  3 15:54:29 2019
 ************************************************************************'''
#!/usr/bin/env python3
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from torchvision import transforms
#from config import *
import json

INDEX = {
        'margin': 0, 'interior': 1, 'skel': 2, 'scrap':3, 'stipp': 4,
        'blotch': 5, 'serp': 6, 'undef': 7, 'normmar': 8, 'normint': 9,
        }

LABEL = ['margin', 'interior', 'skel', 'scrap', 'stipp', 'blotch', 'serp', 'undef', 'normmar', 'normint']
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

class Damage_Dataset(data.Dataset):
    '''
    input is image, target is annotations for every image
    '''
    def __init__(self, name, label_root, transform=None):
        self.name = name
        self.label_root = label_root
        self.transform = transform
        self.labels = load_json(label_root)
        self.ids = list(self.labels.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return torch.from_numpy(img).permute(2, 0, 1), target

    def pull_item(self, index):
        try:
            img = cv2.imread(self.ids[index])
        except img is None:
            print(self.ids[index]+' does not exist')

        img = img[:,:,::-1].copy() # translate the image from BGR to RGB
        height, width, channels = img.shape
        gt = self.labels[self.ids[index]]
        boxes = []
        labels = []
        for g in gt:
            label = INDEX[g['label']]
            if label > 6:
                continue # omit the normal margin, normal interior and undefined damage type
            if label > 1:
                label = 2
            rect = g['rect']
            # to macth the transform format
            rect = [float(rect[0])/width, float(rect[1])/height, float(rect[0]+rect[2])/width, float(rect[1]+rect[3])/height]
            boxes.append(rect)
            labels.append(label)

        boxes, labels = np.array(boxes), np.array(labels)
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        #print(img, target)

        if self.transform is not None:
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            height, width, channels = img.shape

        return img, target, height, width

    def draw_boxes(self, index):
        img, target, h, w = self.pull_item(index)
        print(target)
        for box in target:
            x, y, xr, yr = box[:4]
            label = LABEL[int(box[4])]
            img = cv2.rectangle(img, (int(x*w), int(y*h)), (int(xr*w), int(yr*h)), (0,0,255), 3)
            img = cv2.putText(img, label, (int(x*w), int(y*h)),
                    FONT, 1, (0, 0, 0), 1, cv2.LINE_AA)
        return img


def show_samples(ds, num):
    for i in range(num):
        img = ds.draw_boxes(i)
    #    name = 'example' + str(i) + '.jpg'
    #    cv2.imwrite(name, img)
        cv2.imshow('img', img[:,:,::-1])
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()


def save_images(ds, num):
    n = len(ds.ids)
    file = {}
    for i in range(num):
        img, target, _, _ = ds.pull_item(i%n)
        name = '../../Data/Images/Validation-3class/'+str(i)+'.jpg'
        img = img[:,:,::-1]
        cv2.imwrite(name, img)
        target = target.tolist()
        file[name] = []
        for t in target:
            xs, ys, xe, ye = t[:4]
            label = LABEL[int(t[4])]
            file[name].append({'rect': [xs*300, ys*300, (xe-xs)*300,(ye-ys)*300], 'label': label})
        #print(file)
    with open('../../Data/Labels/Validation-3class.json', 'w') as f:
        json.dump(file, f)

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from utils import SSDAugmentation
    import torch.utils.data as data
    ds = Damage_Dataset('train', '../../../Data/Labels/label_train_new.json', transform=SSDAugmentation())
    #ds = Damage_Dataset('train', '../../Data/Labels/Validation-3class.json')
    show_samples(ds, 100)
    #save_images(ds, 100)
