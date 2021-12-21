from numpy.lib.type_check import imag
import torch
from torchvision import transforms
import torch.nn as nn
from torch.optim import lr_scheduler
import wandb

from model import ModelSpatial
from dataset import GazeFollow
from config import *
from utils import imutils, evaluation

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
from scipy.misc import imresize
import warnings

warnings.simplefilter(action='ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
# parser.add_argument("--init_weights", type=str, default="initial_weights_for_spatial_training.pt", help="initial weights")
parser.add_argument("--init_weights", type=str, default="", help="initial weights")
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--batch_size", type=int, default=48, help="batch size")
parser.add_argument("--num_worker", type=int, default=12, help="num worker")
parser.add_argument("--epochs", type=int, default=70, help="number of epochs")
parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=50000, help="evaluate every ___ iterations")
parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="logs", help="directory to save log files")
parser.add_argument("--random_seed", type=int, default=12345, help="random seed")
args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def train():

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    transform = _get_transform()

    # Prepare data
    print("Loading Data")
    train_dataset = GazeFollow(gazefollow_train_data, gazefollow_train_depth, gazefollow_train_label,
                      transform, input_size=input_resolution, output_size=output_resolution)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_worker)

    val_dataset = GazeFollow(gazefollow_val_data, gazefollow_val_depth, gazefollow_val_label,
                      transform, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_worker)

    # Set up log dir
    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    print("Constructing model")
    model = ModelSpatial()
    model.to(device)
    if args.init_weights:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Loss functions
    # MSE(https://blog.csdn.net/hao5335156/article/details/81029791)
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    L1_loss = nn.L1Loss(reduce=False, reduction='mean') # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()
    cosine_similarity = nn.CosineSimilarity()

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    # scheduler
    # scheduler_multistep increse lr once it meet milestones.
    # scheduler_plateau reduce lr when plateau (loss not reduce).
    scheduler_multistep = lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,10], gamma=5)
    scheduler_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7, verbose=False)
    # scheduler_multistep = lr_scheduler.MultiStepLR(optimizer, milestones=[1,3,5,7,9,10], gamma=2)
    # scheduler_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-7, verbose=False)

    step = 0
    loss_amp_factor_mse = 10000 # multiplied to the loss to prevent underflow
    loss_amp_factor_inout = 100 # multiplied to the loss to prevent underflow
    loss_amp_factor_angle = 100 # multiplied to the loss to prevent underflow
    loss_amp_factor_depth = 100 # multiplied to the loss to prevent underflow
    lambda_heatmap = 1 # weight for heatmap loss
    lambda_angle = 1 # weight for angle loss
    lambda_depth = 1 # weight for depth loss

    max_steps = len(train_loader)
    optimizer.zero_grad()

    wandb.init(project="gazefollow", config=args)
    # wandb.watch(model, mse_loss, log='all', log_freq=100)

    print("Training in progress ...")
    for ep in range(args.epochs):
        # idx, (img, face, head_channel, gaze_heatmap, path, gaze_inside)
        # img -> whole image(Scene Image), face -> head image(Cropped Head), head_channel -> position image(Head Position)
            # img.shape -> (N, 3, 224, 224)
            # dep.shape -> (N, 1, 224, 224)
            # face.shape -> (N, 3, 224, 224)
            # face_dep.shape -> (N, 1, 224, 224)
            # head_channel.shape -> (N, 1, 224, 224)
            # gaze_heatmap.shape -> (N, 64, 64)
            # gaze_field.shape -> (N, 2, 224, 224)
            # eye.shape -> (N, 2)
            # gaze.shape -> (N, 2)
            # relative_depth.shape -> (N, 1)
        for batch, (img, dep, face, woflip, face_dep, head_channel, gaze_heatmap, gaze_field, eye, gaze, name, gaze_inside, relative_depth) in enumerate(train_loader):
            model.train(True) # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
            images = img.to(device)
            depth = dep.to(device)
            head = head_channel.to(device)
            faces = face.to(device)
            woflip_face = woflip['face'].to(device)
            woflip_gaze = woflip['gaze'].to(device)
            woflip_eye = woflip['eye'].to(device)
            face_depth = face_dep.to(device)
            gaze_heatmap = gaze_heatmap.to(device)
            gaze_field = gaze_field.to(device)
            eye = eye.to(device)
            gaze = gaze.to(device)
            relative_depth = relative_depth.to(device)

            # predict heatmap(N, 1, 64, 64), mean of attention, in/out
            gaze_heatmap_pred, attmap, inout_pred, direction, gaze_field_map = model(images, depth, head, faces, face_depth, gaze_field, eye, device)
            direction_2 = model.gaze(woflip_face, eye, device)
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # Loss
                # l2 loss computed only for inside case
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap) * loss_amp_factor_mse # (N, 64, 64)
            l2_loss = torch.mean(l2_loss, dim=1) # (N, 64)
            l2_loss = torch.mean(l2_loss, dim=1) # (N)
            gaze_inside = gaze_inside.to(device).to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside) # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss)/torch.sum(gaze_inside)
                # cross entropy loss for in vs out
            Xent_loss = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze()) * loss_amp_factor_inout
                # Angle loss 1
            gt_direction = gaze - eye
            angle_loss_1 = (1 - cosine_similarity(direction[:, :2], gt_direction)) * loss_amp_factor_angle
            angle_loss_1 = torch.mul(angle_loss_1, gaze_inside) # zero out loss when it's out-of-frame gaze case
            angle_loss_1 = torch.sum(angle_loss_1)/torch.sum(gaze_inside)
                # Angle loss 2
            # gt_direction_2 = woflip_gaze - woflip_eye
            # angle_loss_2 = (1 - cosine_similarity(direction_2[:, :2], gt_direction_2)) * loss_amp_factor_angle
            # angle_loss_2 = torch.mul(angle_loss_2, gaze_inside) # zero out loss when it's out-of-frame gaze case
            # angle_loss_2 = torch.sum(angle_loss_2)/torch.sum(gaze_inside)
                # Angle equivalent
            direction_eq = torch.cat((-direction_2[:, 0:1], direction_2[:, 1:2]), dim=1)
            angle_loss_eq = (1 - cosine_similarity(direction[:, :2], direction_eq)) * loss_amp_factor_angle
            angle_loss_eq = torch.mul(angle_loss_eq, gaze_inside) # zero out loss when it's out-of-frame gaze case
            angle_loss_eq = torch.sum(angle_loss_eq)/torch.sum(gaze_inside)
                # Angle all
            angle_loss = angle_loss_1 + angle_loss_eq * 0.1
                # depth loss
            depth_loss = L1_loss(direction[:, 2], relative_depth) * loss_amp_factor_depth
            depth_loss = torch.mul(depth_loss, gaze_inside) # zero out loss when it's out-of-frame gaze case
            depth_loss = torch.sum(depth_loss)/torch.sum(gaze_inside)

            total_loss = lambda_heatmap * l2_loss + lambda_angle * angle_loss + lambda_depth * depth_loss #+ Xent_loss

            # NOTE: summed loss is used to train the main model.
            #       l2_loss is used to get SOTA on GazeFollow benchmark.
            total_loss.backward() # loss accumulation

            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if batch % args.print_every == 0:
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (angle){:.4f} (depth){:.4f} (Xent){:.4f}".format(ep, batch+1, max_steps, l2_loss, angle_loss, depth_loss,  Xent_loss))

            if (batch != 0 and batch % args.eval_every == 0) or batch+1 == max_steps:
                print('Validation in progress ...')
                model.train(False)
                AUC = []; min_dist = []; avg_dist = []; min_angle_dir = []; min_angle_norm = []; avg_angle_dir = [] ; avg_angle_norm = []
                with torch.no_grad():
                    # idx, (img, face, head_channel, gaze_heatmap, cont_gaze, imsize, path)
                        # img.shape -> (N, 3, 224, 224),
                        # dep.shape -> (N, 1, 224, 224)
                        # face.shape -> (N, 3, 224, 224)
                        # face_dep.shape -> (N, 1, 224, 224)
                        # head_channel.shape -> (N, 1, 224, 224)
                        # gaze_heatmap.shape -> (N, 64, 64)
                        # gaze_field.shape -> (N, 2, 224, 224)
                        # eye.shape -> (N, 2)
                        # cont_gaze -> (N, 20, 2)
                    for val_batch, (val_img, val_dep, val_face, val_face_dep, val_head_channel, val_gaze_heatmap, val_gaze_field, val_eye, cont_gaze, imsize, _) in enumerate(val_loader):
                        val_images = val_img.to(device)
                        val_depth = val_dep.to(device)
                        val_faces = val_face.to(device)
                        val_face_depth = val_face_dep.to(device)
                        val_head = val_head_channel.to(device)
                        val_gaze_heatmap = val_gaze_heatmap.to(device)
                        val_gaze_field = val_gaze_field.to(device)
                        val_eye = val_eye.to(device)

                        # predict heatmap(N, 1, 64, 64), mean of attention, in/out
                        val_gaze_heatmap_pred, val_attmap, val_inout_pred, val_direction, val_gaze_field_map = model(val_images, val_depth, val_head, val_faces, val_face_depth, val_gaze_field, val_eye, device)
                        val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1) # (N, 1, 64, 64) -> (N, 64, 64)
                        # Loss
                            # l2 loss computed only for inside case, test set only have inside case.
                        val_l2_loss = mse_loss(val_gaze_heatmap_pred, val_gaze_heatmap) * loss_amp_factor_mse # (N, 64, 64)
                        val_l2_loss = torch.mean(val_l2_loss, dim=1) # (N, 64)
                        val_l2_loss = torch.mean(val_l2_loss, dim=1) # (N)
                        val_l2_loss = torch.mean(val_l2_loss, dim=0) # (1)
                            # Angle loss
                        val_angle_loss = torch.tensor(float('inf')).to(device)
                        val_depth_loss = torch.tensor(float('inf')).to(device)

                        val_gaze_heatmap_pred = val_gaze_heatmap_pred.cpu()

                        # go through each data point and record AUC, min dist, avg dist
                        for b_i in range(len(cont_gaze)):
                            # remove padding and recover valid ground truth points
                            valid_gaze = cont_gaze[b_i]
                            valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2) # (20, 2) -> ('<20', 2) get rid of dummy gaze
                            # AUC: area under curve of ROC
                            multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i]) # get white image with black gaze dot
                            scaled_heatmap = imresize(val_gaze_heatmap_pred[b_i], (imsize[b_i][1], imsize[b_i][0]), interp = 'bilinear')
                            auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                            AUC.append(auc_score)
                            # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                            pred_x, pred_y = evaluation.argmax_pts(val_gaze_heatmap_pred[b_i]) # return index of max heatmap value
                            norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
                            all_distances = []; all_angle_dir = []; all_angle_norm = []
                            val_direction_dir = val_direction[b_i, :2].cpu()
                            val_direction_norm = torch.FloatTensor(norm_p) - val_eye[b_i].cpu()
                            for gt_gaze in valid_gaze:
                                # L2 dist
                                all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                                gt_gaze = gt_gaze.to(device)
                                # angle loss
                                val_gt_direction = gt_gaze - val_eye[b_i]
                                val_angle_loss_temp = (1 - cosine_similarity(val_direction[b_i, :2].unsqueeze(0), val_gt_direction.unsqueeze(0))) * loss_amp_factor_angle
                                val_angle_loss = val_angle_loss_temp if val_angle_loss > val_angle_loss_temp else val_angle_loss
                                # depth loss
                                x, y = int(gt_gaze[0] * input_resolution), int(gt_gaze[1] * input_resolution)
                                val_relative_depth = val_depth[b_i, 0, x, y]
                                val_depth_loss_temp = L1_loss(val_direction[b_i, 2], val_relative_depth) * loss_amp_factor_depth
                                val_depth_loss = val_depth_loss_temp if val_depth_loss > val_depth_loss_temp else val_depth_loss

                                # angle
                                val_gt_direction = val_gt_direction.cpu()
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

                        val_total_loss = lambda_heatmap * val_l2_loss + lambda_angle * val_angle_loss + lambda_depth * val_depth_loss #+ Xent_loss

                print("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}".format(
                    torch.mean(torch.tensor(AUC)),
                    torch.mean(torch.tensor(min_dist)),
                    torch.mean(torch.tensor(avg_dist))))
                print("\tmin angle dir:{:.4f}\tmin angle norm:{:.4f}\tavg angle dir:{:.4f}\tavg angle norm:{:.4f}".format(
                    torch.mean(torch.tensor(min_angle_dir)),
                    torch.mean(torch.tensor(min_angle_norm)),
                    torch.mean(torch.tensor(avg_angle_dir)),
                    torch.mean(torch.tensor(avg_angle_norm))))

                if batch+1 == max_steps:
                    # wandb loss
                    wandb.log({"Train Loss": total_loss}, step=(ep+1))

                    # wandb img
                    t = transforms.Resize(input_resolution)
                    wandb.log({"img": [wandb.Image(images, caption="images"),
                                        wandb.Image(depth, caption="depth"),
                                        wandb.Image(faces, caption="faces"),
                                        wandb.Image(face_depth, caption="faces depth"),
                                        wandb.Image(head, caption="head"),
                                        wandb.Image(t(gaze_heatmap.unsqueeze(1)), caption="gaze_heatmap"),
                                        wandb.Image(t(gaze_heatmap_pred.unsqueeze(1)), caption="gaze_heatmap_pred"),
                                        wandb.Image(gaze_field_map, caption="gaze_field_map")]},
                                        step=(ep+1))
                    # wandb val
                    wandb.log({"Validation Loss": val_total_loss,
                            "Validation AUC": torch.mean(torch.tensor(AUC)),
                            "Validation min dist": torch.mean(torch.tensor(min_dist)),
                            "Validation avg dist": torch.mean(torch.tensor(avg_dist)),
                            "Validation min angle direction": torch.mean(torch.tensor(min_angle_dir)),
                            "Validation min angle norm": torch.mean(torch.tensor(min_angle_norm)),
                            "Validation avg angle direction": torch.mean(torch.tensor(avg_angle_dir)),
                            "Validation avg angle norm": torch.mean(torch.tensor(avg_angle_norm))},
                            step=(ep+1))

                    # wandb learning rate
                    wandb.log({"Learning Rate": optimizer.param_groups[0]['lr']}, step=(ep+1))

                    # scheduler
                    scheduler_multistep.step()
                    if ep >= 10:
                        scheduler_plateau.step(val_total_loss)


        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))


if __name__ == "__main__":
    train()
