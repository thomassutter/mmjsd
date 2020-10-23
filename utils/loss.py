
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc


def log_prob_img(output_dist, target, norm_value):
    log_prob = output_dist.log_prob(target).sum();
    mean_val_logprob = log_prob/norm_value;
    return mean_val_logprob;


def log_prob_text(output_dist, target, norm_value):
    log_prob = output_dist.log_prob(target).sum();
    mean_val_logprob = log_prob/norm_value;
    return mean_val_logprob;


def clf_loss(estimate, gt):
    loss = F.binary_cross_entropy(estimate, gt, reduction='mean');
    return loss


def calc_auc(gt, pred_proba):
    fpr, tpr, thresholds = roc_curve(gt, pred_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc;
