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

imageFolder = "/exper/gaze/attention-target-detection/data/gazefollow"
dir = {
'load_dir': os.path.join(imageFolder, "train"),
'save_dir': os.path.join(imageFolder, "train_output")
}
csv_path = os.path.join(imageFolder, "train_annotations_release.txt")
pre_color = [[68, 56, 0], [0, 177, 255], [211, 231, 227], [32, 19, 44], [180, 148, 241], [198, 235, 255], [103, 108, 0]]
gt_color = [[68, 56, 0], [0, 177, 255], [211, 231, 227], [32, 19, 44], [180, 148, 241], [198, 235, 255], [103, 108, 0]]

for key in dir:
    if key != 'load_dir' and not os.path.isdir(dir[key]):
        os.mkdir(dir[key])

for foldername in os.listdir(dir['load_dir']):
    load_foldername = os.path.join(dir['load_dir'], foldername)
    for key in dir:
        if key != 'load_dir':
            save_foldername = os.path.join(dir[key], foldername)
            if not os.path.isdir(save_foldername):
                os.mkdir(save_foldername)


def draw_result(im, eye, head, gt, inout, i):
    image_height, image_width = im.shape[:2]
    x1, y1 = eye
    x1, y1 = image_width * x1, y1 * image_height
    gt_x, gt_y = gt
    gt_x, gt_y = image_width * gt_x, gt_y * image_height
    x1, y1, gt_x, gt_y = map(int, [x1, y1, gt_x, gt_y])

    if inout:
        cv2.circle(im, (gt_x, gt_y), 20, gt_color[i], -1)
        cv2.line(im, (x1, y1), (gt_x, gt_y), gt_color[i], 10)
    cv2.rectangle(im, (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), pre_color[i], 10)

    return im

def main():

    column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                    'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta']
    df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
    # df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
    # df.reset_index(inplace=True)

    df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'inout', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
            'bbox_y_max']].groupby(['path'])

    keys = list(df.groups.keys()) # ['path'] pair key
    length = len(keys)

    for index in range(length):
        g = df.get_group(keys[index]) # label of ['path'] pair
        dir_path = {}
        image_path = ""
        idx = 0
        for i, row in g.iterrows():
            if image_path == "":
                path = row['path']
                dir_path = {
                'save_path': os.path.join(imageFolder, "train_output", path.split('/')[-2], path.split('/')[-1])
                }
                image_path = os.path.join(imageFolder, path)
                im = cv2.imread(image_path)
            x_min = row['bbox_x_min']
            y_min = row['bbox_y_min']
            x_max = row['bbox_x_max']
            y_max = row['bbox_y_max']
            eye_x = row['eye_x']
            eye_y = row['eye_y']
            gaze_x = row['gaze_x']
            gaze_y = row['gaze_y']
            inout = row['inout']

            headposition = (x_min, y_min, x_max, y_max)
            im = draw_result(im, (eye_x, eye_y), headposition, (gaze_x, gaze_y), inout, idx % 7)
            idx += 1
            if idx > 1:
                print(path, eye_x, eye_y)

        cv2.imwrite(dir_path['save_path'], im)


if __name__ == '__main__':
    main()

