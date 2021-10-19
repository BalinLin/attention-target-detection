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
from tensorboardX import SummaryWriter
import warnings

warnings.simplefilter(action='ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
# parser.add_argument("--init_weights", type=str, default="initial_weights_for_spatial_training.pt", help="initial weights")
parser.add_argument("--init_weights", type=str, default="", help="initial weights")
parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
parser.add_argument("--batch_size", type=int, default=24, help="batch size")
parser.add_argument("--epochs", type=int, default=70, help="number of epochs")
parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=500, help="evaluate every ___ iterations")
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
                                               num_workers=0)

    val_dataset = GazeFollow(gazefollow_val_data, gazefollow_val_depth, gazefollow_val_label,
                      transform, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    # Set up log dir
    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

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
    L1_loss = nn.L1Loss(reduction='mean')
    bcelogit_loss = nn.BCEWithLogitsLoss()
    cosine_similarity = nn.CosineSimilarity()

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    # scheduler
    # scheduler_multistep increse lr once it meet milestones.
    # scheduler_plateau reduce lr when plateau (loss not reduce).
    scheduler_multistep = lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,10], gamma=5)
    scheduler_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    step = 0
    loss_amp_factor_mse = 10000 # multiplied to the loss to prevent underflow
    loss_amp_factor_inout = 100 # multiplied to the loss to prevent underflow
    loss_amp_factor_angle = 100 # multiplied to the loss to prevent underflow
    w1 = 0.5 # weight for heatmap loss
    w2 = 0.5 # weight for angle loss
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
        for batch, (img, dep, face, face_dep, head_channel, gaze_heatmap, gaze_field, eye, gaze, name, gaze_inside) in enumerate(train_loader):
            model.train(True) # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
            images = img.to(device)
            depth = dep.to(device)
            head = head_channel.to(device)
            faces = face.to(device)
            face_depth = face_dep.to(device)
            gaze_heatmap = gaze_heatmap.to(device)
            gaze_field = gaze_field.to(device)
            eye = eye.to(device)
            gaze = gaze.to(device)

            # predict heatmap(N, 1, 64, 64), mean of attention, in/out
            gaze_heatmap_pred, attmap, inout_pred, direction, gaze_field_map = model(images, depth, head, faces, face_depth, gaze_field, device)
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # Loss
                # l2 loss computed only for inside case
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap)*loss_amp_factor_mse # (N, 64, 64)
            l2_loss = torch.mean(l2_loss, dim=1) # (N, 64)
            l2_loss = torch.mean(l2_loss, dim=1) # (N)
            gaze_inside = gaze_inside.to(device).to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside) # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss)/torch.sum(gaze_inside)
                # cross entropy loss for in vs out
            Xent_loss = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze()) * loss_amp_factor_inout
                # Angle loss
            gt_direction = gaze - eye
            angle_loss = (torch.mean(1 - cosine_similarity(direction, gt_direction)) +
                          L1_loss(direction, gt_direction) ) / 2 * loss_amp_factor_angle
            if ep == 0:
                total_loss = angle_loss
            elif ep >= 7 and ep <= 14:
                total_loss = l2_loss #+ Xent_loss
            else:
                total_loss = w1 * l2_loss + w2 * angle_loss #+ Xent_loss

            # NOTE: summed loss is used to train the main model.
            #       l2_loss is used to get SOTA on GazeFollow benchmark.
            total_loss.backward() # loss accumulation

            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if batch % args.print_every == 0:
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Xent){:.4f}".format(ep, batch+1, max_steps, l2_loss, Xent_loss))
                # Tensorboard
                ind = np.random.choice(len(images), replace=False)
                writer.add_scalar("Train Loss", total_loss, global_step=step)

            if (batch != 0 and batch % args.eval_every == 0) or batch+1 == max_steps:
                print('Validation in progress ...')
                model.train(False)
                AUC = []; min_dist = []; avg_dist = []
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
                        val_gaze_heatmap_pred, val_attmap, val_inout_pred, val_direction, val_gaze_field_map = model(val_images, val_depth, val_head, val_faces, val_face_depth, val_gaze_field, device)
                        val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1) # (N, 1, 64, 64) -> (N, 64, 64)
                        # Loss
                            # l2 loss computed only for inside case, test set only have inside case.
                        val_l2_loss = mse_loss(val_gaze_heatmap_pred, val_gaze_heatmap)*loss_amp_factor_mse # (N, 64, 64)
                        val_l2_loss = torch.mean(val_l2_loss, dim=1) # (N, 64)
                        val_l2_loss = torch.mean(val_l2_loss, dim=1) # (N)
                        val_l2_loss = torch.mean(val_l2_loss, dim=0) # (1)
                            # Angle loss
                        val_angle_loss = torch.tensor(float('inf')).to(device)

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
                            all_distances = []
                            for gt_gaze in valid_gaze:
                                all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                                gt_gaze = gt_gaze.to(device)
                                val_gt_direction_temp = gt_gaze - val_eye
                                val_angle_loss_temp = (torch.mean(1 - cosine_similarity(val_direction, val_gt_direction_temp)) +
                                                       L1_loss(val_direction, val_gt_direction_temp) ) / 2 * loss_amp_factor_angle
                                val_angle_loss = val_angle_loss_temp if val_angle_loss > val_angle_loss_temp else val_angle_loss
                            min_dist.append(min(all_distances))
                            # average distance: distance between the predicted point and human average point
                            mean_gt_gaze = torch.mean(valid_gaze, 0)
                            avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
                            avg_dist.append(avg_distance)

                        if ep == 0:
                            val_total_loss = val_angle_loss
                        elif ep >= 7 and ep <= 14:
                            val_total_loss = val_l2_loss #+ Xent_loss
                        else:
                            val_total_loss = w1 * val_l2_loss + w2 * val_angle_loss #+ Xent_loss

                print("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}".format(
                      torch.mean(torch.tensor(AUC)),
                      torch.mean(torch.tensor(min_dist)),
                      torch.mean(torch.tensor(avg_dist))))

                # Tensorboard
                val_ind = np.random.choice(len(val_images), replace=False)
                writer.add_scalar('Validation AUC', torch.mean(torch.tensor(AUC)), global_step=step)
                writer.add_scalar('Validation min dist', torch.mean(torch.tensor(min_dist)), global_step=step)
                writer.add_scalar('Validation avg dist', torch.mean(torch.tensor(avg_dist)), global_step=step)

                if batch+1 == max_steps:
                    # wandb loss
                    wandb.log({"Train Loss": total_loss}, step=(ep+1))

                    # wandb img
                    t = transforms.Resize(input_resolution)
                    wandb.log({"img": [wandb.Image(images, caption="images"),
                                        wandb.Image(depth, caption="depth"),
                                        wandb.Image(faces, caption="faces"),
                                        wandb.Image(head, caption="head"),
                                        wandb.Image(t(gaze_heatmap.unsqueeze(1)), caption="gaze_heatmap"),
                                        wandb.Image(t(gaze_heatmap_pred.unsqueeze(1)), caption="gaze_heatmap_pred"),
                                        wandb.Image(gaze_field_map, caption="gaze_heatmap_pred")]},
                                        step=(ep+1))
                    # wandb val
                    wandb.log({"Validation Loss": val_total_loss,
                            "Validation AUC": torch.mean(torch.tensor(AUC)),
                            "Validation min dist": torch.mean(torch.tensor(min_dist)),
                            "Validation avg dist": torch.mean(torch.tensor(avg_dist))},
                            step=(ep+1))

                    # wandb learning rate
                    wandb.log({"Learning Rate": optimizer.param_groups[0]['lr']}, step=(ep+1))

                    # scheduler
                    scheduler_multistep.step()
                    scheduler_plateau.step(val_total_loss)


        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))


if __name__ == "__main__":
    train()
