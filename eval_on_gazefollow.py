import torch
from torchvision import transforms
import torch.nn as nn

from model import ModelSpatial
from dataset import GazeFollow
from config import *
from utils import imutils, evaluation

import argparse
import os
import numpy as np
from scipy.misc import imresize
import warnings
import natsort

warnings.simplefilter(action='ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--model_weights", type=str, default="model_gazefollow.pt", help="model weights")
parser.add_argument("--batch_size", type=int, default=48, help="batch size")
parser.add_argument("--num_worker", type=int, default=12, help="num worker")
parser.add_argument("--log_dir", type=str, default="2021-08-26_06-55-45", help="directory to eval log files")
args = parser.parse_args()
home = os.path.expanduser("~")
logfolder = args.log_dir
logdir = os.path.join("logs", logfolder)


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def test():
    transform = _get_transform()

    # Prepare data
    print("Loading Data")
    val_dataset = GazeFollow(gazefollow_val_data, gazefollow_val_depth, gazefollow_val_label,
                      transform, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_worker)

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    print("Constructing model")
    model = ModelSpatial()
    model.cuda().to(device)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('Evaluation in progress ...')
    model.train(False)
    AUC = []; min_dist = []; avg_dist = []; min_angle_dir = []; min_angle_norm = []; avg_angle_dir = [] ; avg_angle_norm = []
    with torch.no_grad():
        for val_batch, (val_img, val_dep, val_face, val_face_dep, val_head_channel, val_gaze_heatmap, val_gaze_field, val_eye, cont_gaze, imsize, _) in enumerate(val_loader):
            val_images = val_img.to(device)
            val_depth = val_dep.to(device)
            val_faces = val_face.to(device)
            val_face_depth = val_face_dep.to(device)
            val_head = val_head_channel.to(device)
            val_gaze_heatmap = val_gaze_heatmap.to(device)
            val_gaze_field = val_gaze_field.to(device)
            val_eye = val_eye.to(device)

            val_gaze_heatmap_pred, val_attmap, val_inout_pred, val_direction, val_gaze_field_map = model(val_images, val_depth, val_head, val_faces, val_face_depth, val_gaze_field, val_eye, device)
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1) # (N, 1, 64, 64) -> (N, 64, 64)
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.cpu()

            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(cont_gaze)):
                # remove padding and recover valid ground truth points
                valid_gaze = cont_gaze[b_i]
                valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
                # AUC: area under curve of ROC
                multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                scaled_heatmap = imresize(val_gaze_heatmap_pred[b_i], (imsize[b_i][1], imsize[b_i][0]), interp = 'bilinear')
                auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                AUC.append(auc_score)
                # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                pred_x, pred_y = evaluation.argmax_pts(val_gaze_heatmap_pred[b_i])
                norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
                all_distances = []; all_angle_dir = []; all_angle_norm = []
                val_direction_dir = val_direction[b_i, :2].cpu()
                val_direction_norm = torch.FloatTensor(norm_p) - val_eye[b_i].cpu()
                for gt_gaze in valid_gaze:
                    all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                    val_gt_direction = gt_gaze - val_eye[b_i].cpu()
                    all_angle_dir.append(evaluation.angle_degree(val_gt_direction, val_direction_dir))
                    all_angle_norm.append(evaluation.angle_degree(val_gt_direction, val_direction_norm))
                min_dist.append(min(all_distances))
                min_angle_dir.append(min(all_angle_dir))
                min_angle_norm.append(min(all_angle_norm))
                # average distance: distance between the predicted point and human average point
                mean_gt_gaze = torch.mean(valid_gaze, 0)
                mean_gt_direction = mean_gt_gaze - val_eye[b_i].cpu()
                avg_dist.append(evaluation.L2_dist(mean_gt_gaze, norm_p))
                avg_angle_dir.append(evaluation.angle_degree(mean_gt_direction, val_direction_dir))
                avg_angle_norm.append(evaluation.angle_degree(mean_gt_direction, val_direction_norm))

    print("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}".format(
          torch.mean(torch.tensor(AUC)),
          torch.mean(torch.tensor(min_dist)),
          torch.mean(torch.tensor(avg_dist))))
    print("\tmin angle dir:{:.4f}\tmin angle norm:{:.4f}\tavg angle dir:{:.4f}\tavg angle norm:{:.4f}".format(
          torch.mean(torch.tensor(min_angle_dir)),
          torch.mean(torch.tensor(min_angle_norm)),
          torch.mean(torch.tensor(avg_angle_dir)),
          torch.mean(torch.tensor(avg_angle_norm))))

if __name__ == "__main__":
    test()