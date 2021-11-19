from __future__ import absolute_import
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im,onehot_im.size), np.reshape(heatmap,heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float,idx)
    return pred_x, pred_y


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def angle_degree(gt, pred):
    norm_pred = (pred[0] ** 2 + pred[1] ** 2) ** 0.5
    norm_gt = (gt[0] ** 2 + gt[1] ** 2) ** 0.5

    f_cos_sim = (pred[0] * gt[0] + pred[1] * gt[1]) / (norm_gt * norm_pred + 1e-6)
    f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
    f_angle = np.arccos(f_cos_sim) * 180 / np.pi
    return f_angle
