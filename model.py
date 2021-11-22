import torch
import torch.nn as nn
from torchvision import utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import math
import copy
from lib.pytorch_convolutional_rnn import convolutional_rnn
import numpy as np
from resnet import resnet18

# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample # shortcut
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # shortcut
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckConvLSTM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample # shortcut
        self.bn_ds = nn.BatchNorm2d(planes * self.expansion)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn3(out)

        # shortcut
        if self.downsample is not None:
            # RW edit: handles batch_size==1
            if out.shape[0] > 1:
                residual = self.downsample(x)
                residual = self.bn_ds(residual)
            else:
                residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ModelSpatial(nn.Module):
    # Define a ResNet 50-ish arch [3, 4, 6, 3]
    def __init__(self, block = Bottleneck, layers_scene = [3, 4, 6, 3, 2], layers_face = [3, 4, 6, 3, 2]):
        # Resnet Feature Extractor
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatial, self).__init__()
        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # learnable weight
        # self.gamma_fov1 = nn.Parameter(torch.zeros(1))
        # self.gamma_fov2 = nn.Parameter(torch.zeros(1))
        # self.gamma_fov3 = nn.Parameter(torch.zeros(1))
        # self.gamma_fovdepth = nn.Parameter(torch.zeros(1))

        # gaze direction
        self.gaze = GazeTR()

        # scene pathway, input size = 8 means Scene Image cat with fov, depth map and Head Position
        self.conv1_scene = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(4, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50

        # attention
        self.attn = nn.Linear(1808, 1*7*7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion: # if residual size is bigger than output
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion: # if residual size is bigger than output
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)


    def forward(self, images, depth, head, face, face_depth, gaze_field, device):
        ### images -> whole image(Scene Image), head -> position image(Head Position), face -> head image(Cropped Head)
        # images.shape -> torch.Size([batch_size, 3, 224, 224]),
        # depth.shape -> torch.Size([batch_size, 1, 224, 224]),
        # head.shape -> torch.Size([batch_size, 1, 224, 224])
        # face.shape -> torch.Size([batch_size, 3, 224, 224])
        # face_depth.shape -> torch.Size([batch_size, 1, 224, 224])
        # gaze_field.shape -> torch.Size([batch_size, 1, 224, 224])
        # eye.shape -> torch.Size([batch_size, 2])
        # gaze.shape -> torch.Size([batch_size, 2])

        direction = self.gaze(face, device) # (N, 3, 224, 224) -> (N, 3)

        # infer gaze direction and normalized
        norm = torch.norm(direction[:, :2], 2, dim=1)
        normalized_direction = direction[:, :2] / norm.view([-1, 1])

        # generate gaze field map
        batch_size, channel, height, width = gaze_field.size()
        gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
        gaze_field = gaze_field.view([batch_size, -1, 2])
        gaze_field = torch.matmul(gaze_field, normalized_direction.view([batch_size, 2, 1]))
        gaze_field_map = gaze_field.view([batch_size, height, width, 1])
        gaze_field_map = gaze_field_map.permute([0, 3, 1, 2]).contiguous()

        gaze_field_map = self.relu(gaze_field_map)

        # mask with gaze_field
        gaze_field_map_2 = torch.pow(gaze_field_map, 3)
        gaze_field_map_3 = torch.pow(gaze_field_map, 5)
        # gaze_field_map_gamma = self.gamma_fov1 * gaze_field_map + self.gamma_fov2 * gaze_field_map_2 + self.gamma_fov3 * gaze_field_map_3
        # images = torch.cat([images, gaze_field_map_gamma], dim=1) # (N, 3, 224, 224) + (N, 1, 224, 224) -> (N, 4, 224, 224)
        images = torch.cat([images, gaze_field_map, gaze_field_map_2, gaze_field_map_3], dim=1) # (N, 3, 224, 224) + (N, 3, 224, 224) -> (N, 6, 224, 224)

        # face_depth = face_depth * direction[:, 2].view([batch_size, -1, 1, 1])
        face = torch.cat((face, face_depth), dim=1) # (N, 3, 224, 224) + (N, 1, 224, 224) -> (N, 4, 224, 224)
        face = self.conv1_face(face)       # (N, 4, 224, 224) -> (N, 64, 112, 112)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)          # (N, 64, 112, 112) -> (N, 64, 56, 56)
        face = self.layer1_face(face)      # (N, 64, 56, 56)   -> (N, 256, 56, 56)
        face = self.layer2_face(face)      # (N, 256, 56, 56)  -> (N, 512, 28, 28)
        face = self.layer3_face(face)      # (N, 512, 28, 28)  -> (N, 1024, 14, 14)
        face = self.layer4_face(face)      # (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        face_feat = self.layer5_face(face) # (N, 2048, 7, 7)   -> (N, 1024, 7, 7)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)

        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)

        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1)) # (N, 1808)
        attn_weights = attn_weights.view(-1, 1, 49) # (N, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel, value of attention(dim=2) to be [0-1]
        attn_weights = attn_weights.view(-1, 1, 7, 7) # (N, 1, 7, 7)

        # origin image concat with depth map and haed position (N, 6, 224, 224) + (N, 1, 224, 224) + (N, 1, 224, 224) -> (N, 8, 224, 224)
        # depth_gamma = depth * self.gamma_fovdepthm * gaze_field_map_gamma
        # im = torch.cat((images, depth * gaze_field_map, depth * gaze_field_map_2, depth * gaze_field_map_3), dim=1)
        # depth = depth * direction[:, 2].view([batch_size, -1, 1, 1])
            # get front, mid and back depth map by value
        offset = 0.2
        for idx in range(batch_size):
            if direction[idx, 2] >= 0:
                front = depth[idx]
                mask = front > -offset
                front[mask] += offset
                front = torch.clamp(front, min=0, max=1)
                depth[idx] = front
            else:
                back = depth[idx]
                mask = back < offset
                back[mask] -= offset
                back = torch.clamp(depth[idx], min=-1, max=0)
                depth[idx] = back

        im = torch.cat((images, depth), dim=1)
        im = torch.cat((im, head), dim=1)
        im = self.conv1_scene(im)           # (N, 8, 224, 224) -> (N, 64, 112, 112)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)               # (N, 64, 112, 112) -> (N, 64, 56, 56)
        im = self.layer1_scene(im)          # (N, 64, 56, 56)   -> (N, 256, 56, 56)
        im = self.layer2_scene(im)          # (N, 256, 56, 56)  -> (N, 512, 28, 28)
        im = self.layer3_scene(im)          # (N, 512, 28, 28)  -> (N, 1024, 14, 14)
        im = self.layer4_scene(im)          # (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        scene_feat = self.layer5_scene(im)  # (N, 2048, 7, 7)   -> (N, 1024, 7, 7)

        # applying attention weights on scene feat
        # attn_weights = torch.ones(attn_weights.shape)/49.0
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) * (N, 1024, 7, 7) -> (N, 1024, 7, 7)

        # attention feature concat with face feature
        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1) #  (N, 1024, 7, 7) + (N, 1024, 7, 7) -> (N, 2048, 7, 7)

        # scene + face feat -> in/out
        encoding_inout = self.compress_conv1_inout(scene_face_feat) # (N, 2048, 7, 7) -> (N, 512, 7, 7)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout) # (N, 512, 7, 7) -> (N, 1, 7, 7)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = encoding_inout.view(-1, 49) # (N, 1, 7, 7) -> (N, 49)
        encoding_inout = self.fc_inout(encoding_inout) # (N, 49) -> (N, 1)

        # scene + face feat -> encoding -> decoding
        encoding = self.compress_conv1(scene_face_feat) # (N, 2048, 7, 7) -> (N, 1024, 7, 7)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding) # (N, 1024, 7, 7) -> (N, 512, 7, 7)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        # Hout​=(Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        x = self.deconv1(encoding) # (N, 512, 7, 7) -> (N, 256, 15, 15)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x) # (N, 256, 15, 15) -> (N, 128, 31, 31)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x) # (N, 128, 31, 31) -> (N, 1, 64, 64)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x) # (N, 1, 64, 64) -> (N, 1, 64, 64)

        # x -> output heatmap, attn_weights -> mean of attention, encoding_inout -> in/out
        return x, torch.mean(attn_weights, 1, keepdim=True), encoding_inout, direction, gaze_field_map


class ModelSpatioTemporal(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, block=BottleneckConvLSTM, num_lstm_layers = 1, bidirectional = False, layers_scene = [3, 4, 6, 3, 2], layers_face = [3, 4, 6, 3, 2]):
        # Resnet Feature Extractor
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatioTemporal, self).__init__()
        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # scene pathway
        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50

        # attention
        self.attn = nn.Linear(1808, 1*7*7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        self.convlstm_scene = convolutional_rnn.Conv2dLSTM(in_channels=512,
                                                     out_channels=512,
                                                     kernel_size=3,
                                                     num_layers=num_lstm_layers,
                                                     bidirectional=bidirectional,
                                                     batch_first=True,
                                                     stride=1,
                                                     dropout=0.5)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion: # if residual size is bigger than output
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion: # if residual size is bigger than output
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)

    def forward(self, images, head, face, hidden_scene: tuple = None, batch_sizes: list = None):
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)

        # scene + face feat -> in/out
        encoding_inout = self.compress_conv1_inout(scene_face_feat)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)

        # scene + face feat -> encoding -> decoding
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        # RW edit: x should be of shape (size, channel, width, height)
        x_pad = PackedSequence(encoding, batch_sizes)
        y, hx = self.convlstm_scene(x_pad, hx=hidden_scene)
        deconv = y.data

        inout_val = encoding_inout.view(-1, 49)
        inout_val = self.fc_inout(inout_val)

        deconv = self.deconv1(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn1(deconv)
        deconv = self.relu(deconv)
        deconv = self.deconv2(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn2(deconv)
        deconv = self.relu(deconv)
        deconv = self.deconv3(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn3(deconv)
        deconv = self.relu(deconv)
        deconv = self.conv4(deconv)

        return deconv, inout_val, hx


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos


    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class GazeTR(nn.Module):
    def __init__(self):
        super(GazeTR, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6

        self.base_model = resnet18(pretrained=False, maps=maps)

        # d_model: dim of Q, K, V
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps,
                  nhead,
                  dim_feedforward,
                  dropout)

        encoder_norm = nn.LayerNorm(maps)
        # num_encoder_layer: deeps of layers

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 3)

        self.loss_op = nn.L1Loss()

    def forward(self, x_in, device):
        feature = self.base_model(x_in) # (N, 32, 7, 7)
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)

        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)

        position = torch.from_numpy(np.arange(0, 50)).to(device)

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)

        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)

        return gaze