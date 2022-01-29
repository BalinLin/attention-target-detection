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
from model_visal import ModelSpatial
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision.utils import save_image

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((224, 224)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def _get_transform_depth():
    transform_list = []
    transform_list.append(transforms.Resize((224, 224)))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

transform = _get_transform()
transform_depth = _get_transform_depth()
device = torch.device('cuda', 0)

imageFolder = "/exper/gaze/attention-target-detection/images"
dir = {
'load_dir': os.path.join(imageFolder, "test2"),
'save_dir': os.path.join(imageFolder, "test2_output"),
'save_dir_heatmap': os.path.join(imageFolder, "test2_heatmap"),
'save_dir_image': os.path.join(imageFolder, "test2_image"),
'save_dir_depth': os.path.join(imageFolder, "test2_depth"),
'save_dir_depth_rebase': os.path.join(imageFolder, "test2_depth_rebase"),
'save_dir_depth_offset': os.path.join(imageFolder, "test2_depth_offset"),
'save_dir_face': os.path.join(imageFolder, "test2_face"),
'save_dir_face_image': os.path.join(imageFolder, "test2_face_image"),
'save_dir_face_depth': os.path.join(imageFolder, "test2_face_depth"),
'save_dir_face_depth_rebase': os.path.join(imageFolder, "test2_face_depth_rebase"),
'save_dir_face_depth_offset': os.path.join(imageFolder, "test2_face_depth_offset"),
'save_dir_gaze_field_map_1': os.path.join(imageFolder, "test2_gaze_field_map_1"),
'save_dir_gaze_field_map_2': os.path.join(imageFolder, "test2_gaze_field_map_2"),
'save_dir_gaze_field_map_3': os.path.join(imageFolder, "test2_gaze_field_map_3")
}
csv_path = os.path.join(imageFolder, "test_annotations_release.txt")
cm = True

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

def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
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

def preprocess_image(image_path, head, eye):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    depth_path = os.path.join(imageFolder, "test2_depth_1_with_norm", image_path.split('/')[-2], image_path.split('/')[-1])
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]

    face = image[int(head[1]):int(head[3]), int(head[0]):int(head[2]), :]
    face_image_depth = depth[int(head[1]):int(head[3]), int(head[0]):int(head[2])]
    # process face_image for face net
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)
    face_image = transform(face)
    face_image_depth = Image.fromarray(face_image_depth)
    face = transform_depth(face)
    # head_depth for rebasing
    head_depth_arr = np.asarray(face_image_depth)
    pixel_num = int(head[2] - head[0]) * int(head[3] - head[1])
    head_depth = head_depth_arr.sum()
    head_depth = head_depth / pixel_num / 256 if isinstance(head_depth, np.integer) else 0

    face_image_depth = transform_depth(face_image_depth)
    # process image for saliency net
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    depth = Image.fromarray(depth)
    depth = transform_depth(depth)

    # Rebase
    depth_rebase = depth - head_depth
    face_image_depth_rebase = face_image_depth - head_depth


    # generate head image
    head_channel = imutils.get_head_box_channel(head[0], head[1], head[2], head[3], w, h,
                                                    resolution=224, coordconv=False).unsqueeze(0)

    # generate gaze field

    gaze_field = generate_data_field(eye_point=eye)
    sample = {'image' : image,
              'depth': depth,
              'depth_rebase': depth_rebase,
              'face': face,
              'face_image': face_image,
              'face_image_depth': face_image_depth,
              'face_image_depth_rebase': face_image_depth_rebase,
              'head': head_channel,
              'gaze_field': torch.from_numpy(gaze_field),
              'eye_position': torch.FloatTensor(eye)}

    return sample


def test(net, test_image_path, head, eye):

    net.eval()
    heatmaps = []
    all_images = {}

    data = preprocess_image(test_image_path, head, eye)

    image, depth, depth_rebase, face, face_image, face_image_depth, face_image_depth_rebase, head, gaze_field, eye_position = \
         data['image'], data['depth'], data['depth_rebase'], data['face'], data['face_image'], data['face_image_depth'], data['face_image_depth_rebase'], data['head'], data['gaze_field'], data['eye_position']
    all_images['image'] = image
    all_images['depth'] = depth
    all_images['depth_rebase'] = depth_rebase
    all_images['face'] = face
    all_images['face_image'] = face_image
    all_images['face_depth'] = face_image_depth
    all_images['face_depth_rebase'] = face_image_depth_rebase
    all_images['head'] = head
    all_images['gaze_field'] = gaze_field
    all_images['eye_position'] = eye_position
    image, depth, depth_rebase, face_image, face_image_depth, face_image_depth_rebase, head, gaze_field, eye_position = \
         map(lambda x: Variable(x.unsqueeze(0).cuda(), volatile=True), [image, depth, depth_rebase, face_image, face_image_depth, face_image_depth_rebase, head, gaze_field, eye_position])
    predict_heatmap, attmap, inout_pred, direction, gaze_field_map, model_all_images = net(image, depth_rebase, head, face_image, face_image_depth_rebase, gaze_field, eye_position, device)
    all_images['gaze_field_map_1'] = model_all_images['gaze_field_map_1']
    all_images['gaze_field_map_2'] = model_all_images['gaze_field_map_2']
    all_images['gaze_field_map_3'] = model_all_images['gaze_field_map_3']
    all_images['depth_offset'] = model_all_images['depth_offset']
    all_images['face_depth_offset'] = model_all_images['face_depth_offset']


    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([64, 64])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 64., h_index / 64.])

    return heatmap, f_point[0], f_point[1], all_images

def draw_result(image_path, eye, heatmap, gaze_point, head, gt, all_images):
    image_path = os.path.join(imageFolder, image_path)
    pre_color = [179, 252, 17]
    gt_color = [0, 215, 255]
    x1, y1 = eye
    x2, y2 = gaze_point
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    gt_x = 0
    gt_y = 0
    c = 0

    for x, y in gt:
        if x != -1:
            gt_x += x
            gt_y += y
            c += 1
            # x, y = int(image_width * x), int(y * image_height)
            # cv2.circle(im, (x, y), 5, gt_color, -1)
            # cv2.line(im, (x1, y1), (x, y), gt_color, 3)

    gt_x /= c
    gt_y /= c
    gt_x = int(gt_x * image_width)
    gt_y = int(gt_y * image_height)

    cv2.circle(im, (gt_x, gt_y), 5, gt_color, -1)
    cv2.line(im, (x1, y1), (gt_x, gt_y), gt_color, 3)

    # cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, pre_color, -1)
    cv2.line(im, (x1, y1), (x2, y2), pre_color, 3)
    cv2.rectangle(im, (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), pre_color, 3)

    # heatmap visualization
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = cv2.resize(heatmap, (image_width, image_height))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = (0.3 * heatmap.astype(np.float32) + 0.7 * im.astype(np.float32)).astype(np.uint8)
    img = np.concatenate((im, heatmap), axis=1)

    dir_path = {
    'save_path': os.path.join(imageFolder, "test2_output", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_heatmap': os.path.join(imageFolder, "test2_heatmap", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_image': os.path.join(imageFolder, "test2_image", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_depth': os.path.join(imageFolder, "test2_depth", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_depth_rebase': os.path.join(imageFolder, "test2_depth_rebase", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_depth_offset': os.path.join(imageFolder, "test2_depth_offset", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_face': os.path.join(imageFolder, "test2_face", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_face_image': os.path.join(imageFolder, "test2_face_image", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_face_depth': os.path.join(imageFolder, "test2_face_depth", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_face_depth_rebase': os.path.join(imageFolder, "test2_face_depth_rebase", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_face_depth_offset': os.path.join(imageFolder, "test2_face_depth_offset", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_gaze_field_map_1': os.path.join(imageFolder, "test2_gaze_field_map_1", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_gaze_field_map_2': os.path.join(imageFolder, "test2_gaze_field_map_2", image_path.split('/')[-2], image_path.split('/')[-1]),
    'save_path_gaze_field_map_3': os.path.join(imageFolder, "test2_gaze_field_map_3", image_path.split('/')[-2], image_path.split('/')[-1])
    }

    cv2.imwrite(dir_path['save_path'], im)
    cv2.imwrite(dir_path['save_path_heatmap'], heatmap)
    save_image(all_images['image'], dir_path['save_path_image'])
    save_image(all_images['depth'], dir_path['save_path_depth'])
    save_image(all_images['depth_rebase'], dir_path['save_path_depth_rebase'])
    save_image(all_images['depth_offset'], dir_path['save_path_depth_offset'])
    save_image(all_images['face'], dir_path['save_path_face'])
    save_image(all_images['face_image'], dir_path['save_path_face_image'])
    save_image(all_images['face_depth'], dir_path['save_path_face_depth'])
    save_image(all_images['face_depth_rebase'], dir_path['save_path_face_depth_rebase'])
    save_image(all_images['face_depth_offset'], dir_path['save_path_face_depth_offset'])
    save_image(all_images['gaze_field_map_1'], dir_path['save_path_gaze_field_map_1'])
    save_image(all_images['gaze_field_map_2'], dir_path['save_path_gaze_field_map_2'])
    save_image(all_images['gaze_field_map_3'], dir_path['save_path_gaze_field_map_3'])

    if cm:
        cmImage = cv2.imread(dir_path['save_path_depth'])
        cmImage = cv2.applyColorMap(cmImage, cv2.COLORMAP_JET)
        cv2.imwrite(dir_path['save_path_depth'], cmImage)
        cmImage = cv2.imread(dir_path['save_path_depth_rebase'])
        cmImage = cv2.applyColorMap(cmImage, cv2.COLORMAP_JET)
        cv2.imwrite(dir_path['save_path_depth_rebase'], cmImage)
        cmImage = cv2.imread(dir_path['save_path_depth_offset'])
        cmImage = cv2.applyColorMap(cmImage, cv2.COLORMAP_JET)
        cv2.imwrite(dir_path['save_path_depth_offset'], cmImage)
        cmImage = cv2.imread(dir_path['save_path_face_depth'])
        cmImage = cv2.applyColorMap(cmImage, cv2.COLORMAP_JET)
        cv2.imwrite(dir_path['save_path_face_depth'], cmImage)
        cmImage = cv2.imread(dir_path['save_path_face_depth_rebase'])
        cmImage = cv2.applyColorMap(cmImage, cv2.COLORMAP_JET)
        cv2.imwrite(dir_path['save_path_face_depth_rebase'], cmImage)
        cmImage = cv2.imread(dir_path['save_path_face_depth_offset'])
        cmImage = cv2.applyColorMap(cmImage, cv2.COLORMAP_JET)
        cv2.imwrite(dir_path['save_path_face_depth_offset'], cmImage)

    return img

def main():

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load('minus_depth_GazeTR_pos.pt')
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()

    column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                    'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta']
    df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")

    df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
            'bbox_y_max']].groupby(['path', 'eye_x'])

    keys = list(df.groups.keys()) # ['path', 'eye_x'] pair key
    X_test = df
    length = len(keys)

    for index in range(length):
        g = X_test.get_group(keys[index]) # label of ['path', 'eye_x'] pair
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

        test_image_path = path
        x = eye_x
        y = eye_y
        headposition = (x_min, y_min, x_max, y_max)
        print(test_image_path, x, y)
        # gaze_heatmap_pred, attmap, inout_pred, direction, gaze_field_map = model(images, depth, head, faces, face_depth, gaze_field, eye, device)

        heatmap, p_x, p_y, all_images = test(model, os.path.join("./images", test_image_path), headposition, (x, y))
        draw_result(test_image_path, (x, y), heatmap, (p_x, p_y), headposition, cont_gaze, all_images)

if __name__ == '__main__':
    main()

