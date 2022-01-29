import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from utils import imutils
from utils import myutils
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision.utils import save_image
import glob

train = True
imageFolder = "/exper/gaze/attention-target-detection/data/videoatttarget"
Path = "/exper/gaze/attention-target-detection/data/videoatttarget/annotations/train" if train else "/exper/gaze/attention-target-detection/data/videoatttarget/annotations/test"
dir = {
'load_dir': os.path.join(imageFolder, "images"),
'save_dir': os.path.join(imageFolder, "images_output")
}
pre_color = [[68, 56, 0], [0, 177, 255], [211, 231, 227], [32, 19, 44], [180, 148, 241], [198, 235, 255], [103, 108, 0]]
gt_color = [[68, 56, 0], [0, 177, 255], [211, 231, 227], [32, 19, 44], [180, 148, 241], [198, 235, 255], [103, 108, 0]]

for key in dir:
    if key != 'load_dir' and not os.path.isdir(dir[key]):
        os.mkdir(dir[key])

for foldername in os.listdir(Path):
    load_foldername = os.path.join(dir['load_dir'], foldername)
    for key in dir:
        if key != 'load_dir':
            save_foldername = os.path.join(dir[key], foldername)
            if not os.path.isdir(save_foldername):
                os.mkdir(save_foldername)
            for clipname in os.listdir(load_foldername):
                save_foldername_second = os.path.join(save_foldername, clipname)
                if not os.path.isdir(save_foldername_second):
                    os.mkdir(save_foldername_second)


def draw_result(im, eye, head, gt, i):
    image_height, image_width = im.shape[:2]
    x1, y1 = eye
    gt_x, gt_y = gt
    x1, y1, gt_x, gt_y = map(int, [x1, y1, gt_x, gt_y])

    if gt_x != -1:
        cv2.circle(im, (gt_x, gt_y), 20, gt_color[i], -1)
        cv2.line(im, (x1, y1), (gt_x, gt_y), gt_color[i], 10)
    cv2.rectangle(im, (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), pre_color[i], 10)

    return im

def main():

    shows = glob.glob(os.path.join(Path, '*'))
    all_sequence_paths = []  # all of annotation txt path
    for s in shows:
        clip = glob.glob(os.path.join(s, '*'))
        # all_sequence_paths.extend(sequence_annotations)
        for c in clip:
            txtFile = glob.glob(os.path.join(c, '*.txt'))
            dir_path = {
                    'save_path': os.path.join(dir['save_dir'], s.split('/')[-1], c.split('/')[-1])
                    }
            image_dir = {}
            idx = 0
            for t in txtFile:
                df = pd.read_csv(t, header=None, index_col=False,
                        names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey'])
                for i, row in df.iterrows():
                    image_path = os.path.join(dir['load_dir'], s.split('/')[-1], c.split('/')[-1], row['path'])
                    print(image_path)
                    im = cv2.imread(image_path) if row['path'] not in image_dir else image_dir[row['path']]
                    # show_name = t.split('/')[-3]  # annotation_dir/show_name/clip/seq.txt
                    # clip = t.split('/')[-2]
                    # seq_len = len(df.index)
                    x_min = row['xmin']
                    y_min = row['ymin']
                    x_max = row['xmax']
                    y_max = row['ymax']
                    eye_x = (x_min + x_max) // 2
                    eye_y = (y_min + y_max) // 2
                    gaze_x = row['gazex']
                    gaze_y = row['gazey']

                    headposition = (x_min, y_min, x_max, y_max)
                    im = draw_result(im, (eye_x, eye_y), headposition, (gaze_x, gaze_y), idx % 7)
                    image_dir[row['path']] = im
                idx += 1
            for k in image_dir:
                cv2.imwrite(os.path.join(dir_path['save_path'], k), image_dir[k])

if __name__ == '__main__':
    main()

