from matplotlib.pyplot import magma
import torch
from torchvision import transforms
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import wandb

from model import ModelSpatial
from dataset import VideoAttTarget_video
from config import *
from utils import imutils, evaluation, misc
from lib.pytorch_convolutional_rnn import convolutional_rnn

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
parser.add_argument("--init_weights", type=str, default='test.pt', help="initial weights")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_worker", type=int, default=4, help="batch size")
parser.add_argument("--chunk_size", type=int, default=3, help="update every ___ frames")
parser.add_argument("--epochs", type=int, default=3, help="max number of epochs")
parser.add_argument("--print_every", type=int, default=6, help="print every ___ iterations")
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
    train_dataset = VideoAttTarget_video(videoattentiontarget_train_data, videoattentiontarget_train_depth, videoattentiontarget_train_label,
                                          transform=transform, test=False, seq_len_limit=seq_len_limit_config)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_worker,
                                               collate_fn=video_pack_sequences)

    val_dataset = VideoAttTarget_video(videoattentiontarget_val_data, videoattentiontarget_val_depth, videoattentiontarget_val_label,
                                        transform=transform, test=True, seq_len_limit=seq_len_limit_config)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_worker,
                                             collate_fn=video_pack_sequences)

    # Set up log dir
    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    num_lstm_layers = 2
    print("Constructing model")
    model = ModelSpatial()
    model.cuda(device)
    if args.init_weights:
        print("Loading weights")
        model_dict = model.state_dict()
        snapshot = torch.load(args.init_weights)
        snapshot = snapshot['model']
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)

    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam([
                        {'params': model.deconv1.parameters(), 'lr': args.lr},
                        {'params': model.deconv2.parameters(), 'lr': args.lr},
                        {'params': model.deconv3.parameters(), 'lr': args.lr},
                        {'params': model.conv4.parameters(), 'lr': args.lr},
                        {'params': model.fc_inout.parameters(), 'lr': args.lr*5},
                        ], lr = 0)

    step = 0
    loss_amp_factor_mse = 10000 # multiplied to the loss to prevent underflow
    loss_amp_factor_inout = 100 # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()
    # wandb.init(project="videoatttarget", config=args)
    # wandb.watch(model, mse_loss, log='all', log_freq=100)

    print("Training in progress ...")
    for ep in range(args.epochs):
        # img -> whole image(Scene Image), face -> head image(Cropped Head), head_channel -> position image(Head Position)
            # img.shape -> (N, max_length, 3, 224, 224)
            # depth.shape -> (N, max_length, 1, 224, 224)
            # face.shape -> (N, max_length, 3, 224, 224)
            # face_depth.shape -> (N, max_length, 1, 224, 224)
            # head_channel.shape -> (N, max_length, 1, 224, 224)
            # gaze_heatmap.shape -> (N, max_length, 64, 64)
            # gaze_field.shape -> (N, max_length, 2, 224, 224)
            # inout_label.shape -> (N, max_length, 1)
            # lengths.shape -> (N)
        for batch, (img, depth, face, face_depth, head_channel, gaze_heatmap, gaze_field, inout_label, lengths) in enumerate(train_loader):
            model.train(True)
            # freeze batchnorm layers
            for module in model.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()

            # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
            # https://zhuanlan.zhihu.com/p/342685890
            pad_data = pack_padded_sequence(img, lengths, batch_first=True)
            X_pad_data_img, X_pad_sizes = pad_data.data, pad_data.batch_sizes
            X_pad_data_depth = pack_padded_sequence(depth, lengths, batch_first=True).data
            X_pad_data_face = pack_padded_sequence(face, lengths, batch_first=True).data
            X_pad_data_face_depth = pack_padded_sequence(face_depth, lengths, batch_first=True).data
            X_pad_data_head = pack_padded_sequence(head_channel, lengths, batch_first=True).data
            Y_pad_data_heatmap = pack_padded_sequence(gaze_heatmap, lengths, batch_first=True).data
            Y_pad_data_gaze_field = pack_padded_sequence(gaze_field, lengths, batch_first=True).data
            Y_pad_data_inout = pack_padded_sequence(inout_label, lengths, batch_first=True).data

            # (num_layers, batch_size, feature dims)
            # hx = (torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device),
            #       torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device))
            last_index = 0
            # previous_hx_size = args.batch_size
            avg_l2_loss = []
            avg_Xent_loss = []

            for i in range(0, lengths[0], args.chunk_size): # (args.chunk_size, C, H, W)
                # In this for loop, we read batched images across the time dimension
                    # we step forward N = chunk_size frames
                X_pad_sizes_slice = X_pad_sizes[i:i + args.chunk_size].cuda(device)
                curr_length = np.sum(X_pad_sizes_slice.cpu().detach().numpy())

                # slice padded data
                X_pad_data_slice_img = X_pad_data_img[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_depth = X_pad_data_depth[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_face = X_pad_data_face[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_face_depth = X_pad_data_face_depth[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_head = X_pad_data_head[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_heatmap = Y_pad_data_heatmap[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_gaze_field = Y_pad_data_gaze_field[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_inout = Y_pad_data_inout[last_index:last_index + curr_length].cuda(device)
                last_index += curr_length

                # detach previous hidden states to stop gradient flow
                # prev_hx = (hx[0][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach(),
                #            hx[1][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach())

                # forward pass
                gaze_heatmap_pred, attmap, inout_pred, direction, gaze_field_map = model(X_pad_data_slice_img, X_pad_data_slice_depth, \
                                                                                         X_pad_data_slice_head, X_pad_data_slice_face, \
                                                                                         X_pad_data_slice_face_depth, Y_pad_data_slice_gaze_field, \
                                                                                         device)

                # gaze_heatmap_pred, inout_pred, hx = model(X_pad_data_slice_img, X_pad_data_slice_head, X_pad_data_slice_face, \
                #                                          hidden_scene=prev_hx, batch_sizes=X_pad_sizes_slice)

                # compute loss
                    # l2 loss computed only for inside case
                l2_loss = mse_loss(gaze_heatmap_pred.squeeze(1), Y_pad_data_slice_heatmap) * loss_amp_factor_mse # (args.chunk_size, 64, 64)
                l2_loss = torch.mean(l2_loss, dim=1) # (args.chunk_size, 64)
                l2_loss = torch.mean(l2_loss, dim=1) # (64)
                Y_pad_data_slice_inout = Y_pad_data_slice_inout.cuda(device).to(torch.float).squeeze()
                l2_loss = torch.mul(l2_loss, Y_pad_data_slice_inout) # zero out loss when it's outside gaze case
                l2_loss = torch.sum(l2_loss)/torch.sum(Y_pad_data_slice_inout)
                avg_l2_loss.append(l2_loss)
                    # cross entropy loss for in vs out
                Xent_loss = bcelogit_loss(inout_pred.squeeze(), Y_pad_data_slice_inout.squeeze()) * loss_amp_factor_inout
                avg_Xent_loss.append(Xent_loss)

                total_loss = l2_loss + Xent_loss
                total_loss.backward() # loss accumulation

                # update model parameters
                optimizer.step()
                optimizer.zero_grad()

                # previous_hx_size = X_pad_sizes_slice[-1]

                step += 1

                if ((batch + 1) % args.print_every == 0 or (batch + 1) == max_steps) and last_index == X_pad_data_img.shape[0]:
                    print("Epoch:{:04d}\tbatch:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Xent){:.4f}".format(ep, batch+1, max_steps, torch.mean(torch.tensor(avg_l2_loss)), torch.mean(torch.tensor(avg_Xent_loss))))

                if (batch + 1) == max_steps and last_index == X_pad_data_img.shape[0]:
                    print('Validation in progress ...')
                    model.train(False)
                    AUC = []; in_vs_out_groundtruth = []; in_vs_out_pred = []; distance = []
                    with torch.no_grad():
                        for val_batch, (val_img, val_depth, val_face, val_face_depth, val_head_channel, val_gaze_heatmap, val_gaze_field, cont_gaze, val_inout_label, val_lengths) in enumerate(val_loader):
                            print('\tprogress = ', val_batch+1, '/', len(val_loader))
                            val_pad_data = pack_padded_sequence(val_img, val_lengths, batch_first=True)
                            val_X_pad_data_img, val_X_pad_sizes = val_pad_data.data, val_pad_data.batch_sizes
                            val_X_pad_data_depth = pack_padded_sequence(val_depth, val_lengths, batch_first=True).data
                            val_X_pad_data_face = pack_padded_sequence(val_face, val_lengths, batch_first=True).data
                            val_X_pad_data_face_depth = pack_padded_sequence(val_face_depth, val_lengths, batch_first=True).data
                            val_X_pad_data_head = pack_padded_sequence(val_head_channel, val_lengths, batch_first=True).data
                            val_Y_pad_data_heatmap = pack_padded_sequence(val_gaze_heatmap, val_lengths, batch_first=True).data
                            val_Y_pad_data_gaze_field = pack_padded_sequence(val_gaze_field, val_lengths, batch_first=True).data
                            val_Y_pad_data_cont_gaze = pack_padded_sequence(cont_gaze, val_lengths, batch_first=True).data
                            val_Y_pad_data_inout = pack_padded_sequence(val_inout_label, val_lengths, batch_first=True).data

                            # (num_layers, batch_size, feature dims)
                            # val_hx = (torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device),
                            #           torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device))
                            val_last_index = 0
                            # val_previous_hx_size = args.batch_size

                            for j in range(0, val_lengths[0], args.chunk_size): # (args.chunk_size, C, H, W)
                                val_X_pad_sizes_slice = val_X_pad_sizes[j:j + args.chunk_size].cuda(device)
                                val_curr_length = np.sum(val_X_pad_sizes_slice.cpu().detach().numpy())
                                # slice padded data
                                val_X_pad_data_slice_img = val_X_pad_data_img[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_X_pad_data_slice_depth = val_X_pad_data_depth[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_X_pad_data_slice_face = val_X_pad_data_face[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_X_pad_data_slice_face_depth = val_X_pad_data_face_depth[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_X_pad_data_slice_head = val_X_pad_data_head[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_Y_pad_data_slice_heatmap = val_Y_pad_data_heatmap[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_Y_pad_data_slice_gaze_field = val_Y_pad_data_gaze_field[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_Y_pad_data_slice_cont_gaze = val_Y_pad_data_cont_gaze[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_Y_pad_data_slice_inout = val_Y_pad_data_inout[val_last_index:val_last_index + val_curr_length].cuda(device)
                                val_last_index += val_curr_length

                                # detach previous hidden states to stop gradient flow
                                # val_prev_hx = (val_hx[0][:, :min(val_X_pad_sizes_slice[0], val_previous_hx_size), :, :, :].detach(),
                                #                val_hx[1][:, :min(val_X_pad_sizes_slice[0], val_previous_hx_size), :, :, :].detach())

                                # forward pass
                                val_gaze_heatmap_pred, val_attmap, val_inout_pred, val_direction, val_gaze_field_map = \
                                    model(val_X_pad_data_slice_img, val_X_pad_data_slice_depth, val_X_pad_data_slice_head, val_X_pad_data_slice_face, \
                                          val_X_pad_data_slice_face_depth, val_Y_pad_data_slice_gaze_field, device)

                                # val_gaze_heatmap_pred, val_inout_pred, val_hx = model(val_X_pad_data_slice_img, val_X_pad_data_slice_head, val_X_pad_data_slice_face, \
                                #                                           hidden_scene=val_prev_hx, batch_sizes=val_X_pad_sizes_slice)
                                val_gaze_heatmap_pred = val_gaze_heatmap_pred.cpu()

                                for b_i in range(len(val_Y_pad_data_slice_cont_gaze)):
                                    if val_Y_pad_data_slice_inout[b_i]: # ONLY for 'inside' cases
                                        # AUC: area under curve of ROC
                                        multi_hot = torch.zeros(output_resolution, output_resolution)  # set the size of the output
                                        gaze_x = val_Y_pad_data_slice_cont_gaze[b_i, 0]
                                        gaze_y = val_Y_pad_data_slice_cont_gaze[b_i, 1]
                                        multi_hot = imutils.draw_labelmap(multi_hot, [gaze_x * output_resolution, gaze_y * output_resolution], 3, type='Gaussian')
                                        multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
                                        multi_hot = misc.to_numpy(multi_hot)

                                        scaled_heatmap = imresize(val_gaze_heatmap_pred[b_i].squeeze(), (output_resolution, output_resolution), interp = 'bilinear')
                                        auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                                        AUC.append(auc_score)

                                        # distance: L2 distance between ground truth and argmax point
                                        pred_x, pred_y = evaluation.argmax_pts(val_gaze_heatmap_pred[b_i].squeeze())
                                        norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                                        dist_score = evaluation.L2_dist(val_Y_pad_data_slice_cont_gaze[b_i].cpu(), norm_p).item()
                                        distance.append(dist_score)

                                # in vs out classification
                                in_vs_out_groundtruth.extend(val_Y_pad_data_slice_inout.cpu().numpy())
                                in_vs_out_pred.extend(val_inout_pred.cpu().numpy())

                                # val_previous_hx_size = val_X_pad_sizes_slice[-1]

                            try:
                                print("\tAUC:{:.4f}"
                                      "\tdist:{:.4f}"
                                      "\tin vs out AP:{:.4f}".
                                      format(torch.mean(torch.tensor(AUC)),
                                             torch.mean(torch.tensor(distance)),
                                             evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)))
                            except:
                                pass

                    print("Summary ")
                    print("\tAUC:{:.4f}"
                          "\tdist:{:.4f}"
                          "\tin vs out AP:{:.4f}".
                          format(torch.mean(torch.tensor(AUC)),
                                 torch.mean(torch.tensor(distance)),
                                 evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)))

                    # wandb loss
                    wandb.log({"Train Loss": (torch.mean(torch.tensor(avg_l2_loss)) + torch.mean(torch.tensor(avg_Xent_loss))) / 2}, step=(ep+1))

                    # wandb img
                    t = transforms.Resize(input_resolution)
                    wandb.log({"img": [wandb.Image(X_pad_data_img[last_index - curr_length:last_index], caption="images"),
                                       wandb.Image(X_pad_data_face[last_index - curr_length:last_index], caption="faces"),
                                       wandb.Image(X_pad_data_head[last_index - curr_length:last_index], caption="head"),
                                       wandb.Image(t(Y_pad_data_heatmap[last_index - curr_length:last_index].unsqueeze(1)), caption="gaze_heatmap"),
                                       wandb.Image(t(gaze_heatmap_pred), caption="gaze_heatmap_pred"),
                                       wandb.Image(Y_pad_data_gaze_field[last_index - curr_length:last_index], caption="gaze_heatmap_pred")]},
                                       step=(ep+1))
                    # wandb val
                    wandb.log({"Validation AUC": torch.mean(torch.tensor(AUC)),
                                "Validation dist": torch.mean(torch.tensor(distance)),
                                "Validation in vs out AP": evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)},
                                step=(ep+1))

                    # wandb learning rate
                    wandb.log({"Learning Rate": optimizer.param_groups[0]['lr']}, step=(ep+1))

        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))


def video_pack_sequences(in_batch):
    """
    Pad the variable-length input sequences to fixed length
    :param in_batch: the original input batch of sequences generated by pytorch DataLoader
    :return:
        out_batch (list): the padded batch of sequences
    """
    # Get the number of return values from __getitem__ in the Dataset
    # in_batch.shape -> (N, num_returns, seq_len, 3, 224, 224)
    num_returns = len(in_batch[0])

    # Sort the batch according to the sequence lengths. This is needed by torch func: pack_padded_sequences
    in_batch.sort(key=lambda x: -x[0].shape[0])
    shapes = [b[0].shape[0] for b in in_batch]

    # Determine the length of the padded inputs
    max_length = shapes[0]

    # Declare the output batch as a list
    out_batch = []
    # For each return value in each sequence, calculate the sequence-wise zero padding
    for r in range(num_returns):
        output_values = [] # output_values.shape -> (N, max_length, 3, 224, 224)
        lengths = [] # lengths.shape -> (N)
        for seq in in_batch: # run "N" time
            values = seq[r] # values.shape -> (seq_len, 3, 224, 224)
            seq_size = values.shape[0]
            seq_shape = values.shape[1:]
            lengths.append(seq_size)
            padding = torch.zeros((max_length - seq_size, *seq_shape))
            padded_values = torch.cat((values, padding)) # padded_values.shape -> (max_length, 3, 224, 224)
            output_values.append(padded_values)
        out_batch.append(torch.stack(output_values))
    out_batch.append(lengths)

    # out_batch.shape -> (num_returns + 1, N, max_length, 3, 224, 224)
    return out_batch


if __name__ == "__main__":
    train()
