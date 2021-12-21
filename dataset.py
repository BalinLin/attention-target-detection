import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pandas as pd

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.misc import imresize

import os
import glob
import csv

from utils import imutils
from utils import myutils
from config import *

import warnings
warnings.simplefilter(action='ignore')

def generate_data_field(eye_point, input_size):
    """eye_point is (x, y) and between 0 and 1"""
    height = width = input_size
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid

class GazeFollow(Dataset):
    def __init__(self, data_dir, depth_dir, csv_path, transform, input_size=input_resolution, output_size=output_resolution,
                 test=False, imshow=False):
        if test:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            # make ['path', 'eye_x'] as pair key, one ['path', 'eye_x'] pair have a lot of label with different "gaze_x, gaze_y"
            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max']].groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys()) # ['path', 'eye_x'] pair key
            self.X_test = df
            self.length = len(self.keys)
        else:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                               'gaze_y', 'inout']]
            self.X_train = df['path']
            self.length = len(df)

        self.data_dir = data_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.test = test

        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow

    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index]) # label of ['path', 'eye_x'] pair
            cont_gaze = []
            for i, row in g.iterrows():
                path = row['path']
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout = self.y_train.iloc[index]
            gaze_inside = bool(inout)

        # expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        depth = Image.open(os.path.join(self.depth_dir, path.split("/")[1], path.split("/")[2]))
        depth = depth.convert('L')
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max]) # map type to float
        woflip = {}

        # get gaze depth for rebasing depth
        depth_gaze_x, depth_gaze_y = int(gaze_x * width), int(gaze_y * height)
        relative_depth = depth.getpixel((depth_gaze_x, depth_gaze_y)) / 256

        if self.imshow:
            img.save("origin_img.jpg")
            depth.save("origin_depth.jpg")

        if self.test:
            imsize = torch.IntTensor([width, height])
        else:
            ## data augmentation

            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, eye_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, eye_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, eye_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, eye_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min

                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it (https://pytorch.org/vision/master/_modules/torchvision/transforms/functional.html)
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                depth = TF.crop(depth, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                eye_x, eye_y =   (eye_x * width - offset_x) / float(crop_width), \
                                 (eye_y * height - offset_y) / float(crop_height)

                # else:
                #     gaze_x = -1; gaze_y = -1

                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                woflip['face'] = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                woflip['gaze'] = torch.FloatTensor([gaze_x, gaze_y])
                woflip['eye'] = torch.FloatTensor([eye_x, eye_y])
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                eye_x = 1 - eye_x
            else:
                woflip['face'] = img.transpose(Image.FLIP_LEFT_RIGHT).crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                woflip['gaze'] = torch.FloatTensor([1 - gaze_x, gaze_y])
                woflip['eye'] = torch.FloatTensor([1 - eye_x, eye_y])

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        # get head position
        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.copy().crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face_depth = depth.copy().crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Rebasing offset
        head_depth_arr = np.asarray(face_depth)
        pixel_num = int(x_max - x_min) * int(y_max - y_min)
        head_depth = head_depth_arr.sum()
        head_depth = head_depth / pixel_num / 256 if isinstance(head_depth, np.integer) else 0

        relative_depth -= head_depth

        if self.imshow:
            img.save("img_aug.jpg")
            depth.save("depth_aug.jpg")
            face.save('face_aug.jpg')
            face_depth.save('face_depth_aug.jpg')

        if self.transform is not None:
            transform_list = []
            transform_list.append(transforms.Resize((input_resolution, input_resolution)))
            transform_list.append(transforms.ToTensor())
            transform_depth = transforms.Compose(transform_list)

            img = self.transform(img)
            face = self.transform(face)
            if not self.test:
                woflip['face'] = self.transform(woflip['face'])

            depth = transform_depth(depth)
            depth = depth - head_depth # rebased

            face_depth = transform_depth(face_depth)
            face_depth = face_depth - head_depth # rebased

        # generate the heat map used for deconv prediction
        gaze = [gaze_x, gaze_y]
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if self.test:  # aggregated heatmap
            # NOTE: torch.max(gaze_heatmap) ~= 0.1
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         3,
                                                         type='Gaussian')
            gaze_heatmap /= num_valid
        else:
            # NOTE: torch.max(gaze_heatmap) = 1
            # if gaze_inside:
            gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')

        # generate gaze field for fov
        eye = [eye_x, eye_y]
        gaze_field = generate_data_field(eye_point = eye, input_size = self.input_size)

        if self.imshow:
            fig = plt.figure(111)
            img = 255 - imutils.unnorm(img.numpy()) * 255
            img = np.clip(img, 0, 255)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.imshow(imresize(gaze_heatmap, (self.input_size, self.input_size)), cmap='jet', alpha=0.3)
            plt.imshow(imresize(1 - head_channel.squeeze(0), (self.input_size, self.input_size)), alpha=0.2)
            plt.savefig('viz_aug.png')

        if self.test:
            return img, depth, face, face_depth, head_channel, gaze_heatmap, torch.from_numpy(gaze_field), torch.FloatTensor(eye), cont_gaze, imsize, path
        else:
            return img, depth, face, woflip, face_depth, head_channel, gaze_heatmap, torch.from_numpy(gaze_field), torch.FloatTensor(eye), torch.FloatTensor(gaze), path, gaze_inside, relative_depth

    def __len__(self):
        return self.length


class VideoAttTarget_video(Dataset):
    def __init__(self, data_dir, depth_dir, annotation_dir, transform, input_size=input_resolution, output_size=output_resolution,
                 test=False, imshow=False, seq_len_limit=400):
        shows = glob.glob(os.path.join(annotation_dir, '*'))
        self.all_sequence_paths = []  # all of annotation txt path
        for s in shows:
            sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))
            self.all_sequence_paths.extend(sequence_annotations)
        self.data_dir = data_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.input_size = input_size
        self.output_size = output_size
        self.test = test
        self.imshow = imshow
        self.length = len(self.all_sequence_paths)
        self.seq_len_limit = seq_len_limit

    def __getitem__(self, index):
        sequence_path = self.all_sequence_paths[index]
        df = pd.read_csv(sequence_path, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey'])
        show_name = sequence_path.split('/')[-3]  # annotation_dir/show_name/clip/seq.txt
        clip = sequence_path.split('/')[-2]
        seq_len = len(df.index)

        # moving-avg smoothing
        window_size = 11 # should be odd number
        df['xmin'] = myutils.smooth_by_conv(window_size, df, 'xmin')
        df['ymin'] = myutils.smooth_by_conv(window_size, df, 'ymin')
        df['xmax'] = myutils.smooth_by_conv(window_size, df, 'xmax')
        df['ymax'] = myutils.smooth_by_conv(window_size, df, 'ymax')

        if not self.test:
            # cond for data augmentation
            cond_jitter = np.random.random_sample()
            cond_flip = np.random.random_sample()
            cond_color = np.random.random_sample()
            if cond_color < 0.5:
                n1 = np.random.uniform(0.5, 1.5)
                n2 = np.random.uniform(0.5, 1.5)
                n3 = np.random.uniform(0.5, 1.5)
            cond_crop = np.random.random_sample()

            # if longer than seq_len_limit, cut it down to the limit with the init index randomly sampled
            if seq_len > self.seq_len_limit:
                sampled_ind = np.random.randint(0, seq_len - self.seq_len_limit)
                seq_len = self.seq_len_limit
            else:
                sampled_ind = 0

            if cond_crop < 0.5:
                sliced_x_min = df['xmin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_x_max = df['xmax'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_min = df['ymin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_max = df['ymax'].iloc[sampled_ind:sampled_ind+seq_len]

                sliced_gaze_x = df['gazex'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_gaze_y = df['gazey'].iloc[sampled_ind:sampled_ind+seq_len]

                check_sum = sliced_gaze_x.sum() + sliced_gaze_y.sum()
                all_outside = check_sum == -2*seq_len

                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if all_outside:
                    crop_x_min = np.min([sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_y_min.max(), sliced_y_max.max()])
                else:
                    crop_x_min = np.min([sliced_gaze_x.min(), sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_gaze_y.min(), sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_gaze_x.max(), sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_gaze_y.max(), sliced_y_min.max(), sliced_y_max.max()])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Get image size
                path = os.path.join(self.data_dir, show_name, clip, df['path'].iloc[0])
                img = Image.open(path)
                img = img.convert('RGB')
                width, height = img.size

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)
        else:
            sampled_ind = 0


        images, depths, faces, face_depths, head_channels, heatmaps, gaze_fields, paths, gazes, eyes, imsizes, gaze_inouts = [], [], [], [], [], [], [], [], [], [], [], []
        index_tracker = -1
        for i, row in df.iterrows():
            index_tracker = index_tracker+1
            if not self.test:
                if index_tracker < sampled_ind or index_tracker >= (sampled_ind + self.seq_len_limit):
                    continue

            face_x1 = row['xmin']  # note: Already in image coordinates
            face_y1 = row['ymin']  # note: Already in image coordinates
            face_x2 = row['xmax']  # note: Already in image coordinates
            face_y2 = row['ymax']  # note: Already in image coordinates
            gaze_x = row['gazex']  # note: Already in image coordinates
            gaze_y = row['gazey']  # note: Already in image coordinates

            impath = os.path.join(self.data_dir, show_name, clip, row['path'])
            img = Image.open(impath)
            img = img.convert('RGB')

            dppath = os.path.join(self.depth_dir, show_name, clip, row['path'])
            depth = Image.open(dppath)
            depth = depth.convert('L')

            width, height = img.size
            imsize = torch.FloatTensor([width, height])
            # imsizes.append(imsize)

            face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
            gaze_x, gaze_y = map(float, [gaze_x, gaze_y])
            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
            else:
                if gaze_x < 0: # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = True

            if not self.test:
                ## data augmentation
                # Jitter (expansion-only) bounding box size.
                if cond_jitter < 0.5:
                    k = cond_jitter * 0.1
                    face_x1 -= k * abs(face_x2 - face_x1)
                    face_y1 -= k * abs(face_y2 - face_y1)
                    face_x2 += k * abs(face_x2 - face_x1)
                    face_y2 += k * abs(face_y2 - face_y1)
                    face_x1 = np.clip(face_x1, 0, width)
                    face_x2 = np.clip(face_x2, 0, width)
                    face_y1 = np.clip(face_y1, 0, height)
                    face_y2 = np.clip(face_y2, 0, height)

                # Random Crop
                if cond_crop < 0.5:
                    # Crop it
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                    depth = TF.crop(depth, crop_y_min, crop_x_min, crop_height, crop_width)

                    # Record the crop's (x, y) offset
                    offset_x, offset_y = crop_x_min, crop_y_min

                    # convert coordinates into the cropped frame
                    face_x1, face_y1, face_x2, face_y2 = face_x1 - offset_x, face_y1 - offset_y, face_x2 - offset_x, face_y2 - offset_y
                    if gaze_inside:
                        gaze_x, gaze_y = (gaze_x- offset_x), \
                                         (gaze_y - offset_y)
                    else:
                        gaze_x = -1; gaze_y = -1

                    width, height = crop_width, crop_height

                # Flip?
                if cond_flip < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                    x_max_2 = width - face_x1
                    x_min_2 = width - face_x2
                    face_x2 = x_max_2
                    face_x1 = x_min_2
                    if gaze_x != -1 and gaze_y != -1:
                        gaze_x = width - gaze_x

                # Random color change
                if cond_color < 0.5:
                    img = TF.adjust_brightness(img, brightness_factor=n1)
                    img = TF.adjust_contrast(img, contrast_factor=n2)
                    img = TF.adjust_saturation(img, saturation_factor=n3)

            # Head channel image
            head_channel = imutils.get_head_box_channel(face_x1, face_y1, face_x2, face_y2, width, height,
                                                        resolution=self.input_size, coordconv=False).unsqueeze(0)

            # Face crop
            face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))
            face_depth = depth.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))

            # Rebasing offset
            head_depth_arr = np.asarray(face_depth)
            pixel_num = int(face_x2 - face_x1) * int(face_y2 - face_y1)
            head_depth = head_depth_arr.sum()
            head_depth = head_depth / pixel_num / 256 if isinstance(head_depth, np.integer) else 0

            # generate gaze field for fov
            eye = [(face_x2 + face_x1) / 2 / width, (face_y2 + face_y1) / 2 / height]
            gaze_field = generate_data_field(eye_point = eye, input_size = self.input_size)

            if self.transform is not None:
                transform_list = []
                transform_list.append(transforms.Resize((input_resolution, input_resolution)))
                transform_list.append(transforms.ToTensor())
                transform_depth = transforms.Compose(transform_list)

                img = self.transform(img)
                face = self.transform(face)

                depth = transform_depth(depth)
                depth = depth - head_depth # rebased

                face_depth = transform_depth(face_depth)
                face_depth = face_depth - head_depth # rebased

            # Deconv output
            if gaze_inside:
                gaze_x /= float(width) # fractional gaze
                gaze_y /= float(height)
                gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                gaze_map = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
                gazes.append(torch.FloatTensor([gaze_x, gaze_y]))
            else:
                gaze_map = torch.zeros(self.output_size, self.output_size)
                gazes.append(torch.FloatTensor([-1, -1]))

            images.append(img)
            depths.append(depth)
            faces.append(face)
            face_depths.append(face_depth)
            head_channels.append(head_channel)
            heatmaps.append(gaze_map)
            gaze_fields.append(torch.from_numpy(gaze_field))
            eyes.append(torch.FloatTensor(eye))
            gaze_inouts.append(torch.FloatTensor([int(gaze_inside)]))

        if self.imshow:
            for i in range(len(faces)):
                fig = plt.figure(111)
                img = 255 - imutils.unnorm(images[i].numpy()) * 255
                img = np.clip(img, 0, 255)
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.imshow(imresize(heatmaps[i], (self.input_size, self.input_size)), cmap='jet', alpha=0.3)
                plt.imshow(imresize(1 - head_channels[i].squeeze(0), (self.input_size, self.input_size)), alpha=0.2)
                plt.savefig(os.path.join('debug', 'viz_%d_inout=%d.png' % (i, gaze_inouts[i])))
                plt.close('all')

        images = torch.stack(images)
        depths = torch.stack(depths)
        faces = torch.stack(faces)
        face_depths = torch.stack(face_depths)
        head_channels = torch.stack(head_channels)
        heatmaps = torch.stack(heatmaps)
        gaze_fields = torch.stack(gaze_fields)
        eyes = torch.stack(eyes)
        gazes = torch.stack(gazes)
        gaze_inouts = torch.stack(gaze_inouts)
        # imsizes = torch.stack(imsizes)
        # torch.Size([seq_len_limit_config, 3, 224, 224]) torch.Size([seq_len_limit_config, 3, 224, 224]) torch.Size([seq_len_limit_config, 1, 224, 224]) torch.Size([seq_len_limit_config, 64, 64])
        # print(faces.shape, images.shape, head_channels.shape, heatmaps.shape)

        if self.test:
            return images, depths, faces, face_depths, head_channels, heatmaps, gaze_fields, eyes, gazes, gaze_inouts
        else: # train
            return images, depths, faces, face_depths, head_channels, heatmaps, gaze_fields, eyes, gaze_inouts

    def __len__(self):
        return self.length
