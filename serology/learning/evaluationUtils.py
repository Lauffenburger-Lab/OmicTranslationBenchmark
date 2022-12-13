from __future__ import division, print_function
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import torch
import torch.nn as nn
from torch.functional import F
import os
import random
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import re
from functools import partial
from multiprocessing import cpu_count, Pool
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


#Define custom metrics for evaluation
def r_square(y_true, y_pred):
    SS_res =  torch.sum((y_true - y_pred)**2)
    SS_tot = torch.sum((y_true - torch.mean(y_true))**2)
    return (1 - SS_res/(SS_tot + 1e-7))

def get_cindex(y_true, y_pred):
    g = torch.sub(y_pred.unsqueeze(-1), y_pred)
    g = (g == 0.0).type(torch.FloatTensor) * 0.5 + (g > 0.0).type(torch.FloatTensor)

    f = torch.sub(y_true.unsqueeze(-1), y_true) > 0.0
    f = torch.tril(f.type(torch.FloatTensor))

    g = torch.sum(g*f)
    f = torch.sum(f)

    return torch.where(g==0.0, torch.tensor(0.0), g/f)

# def pearson_r(y_true, y_pred):
#     x = y_true
#     y = y_pred
#     mx = torch.mean(x, dim=0)
#     my = torch.mean(y, dim=0)
#     xm, ym = x - mx, y - my
#     r_num = torch.sum(xm * ym)
#     x_square_sum = torch.sum(xm * xm)
#     y_square_sum = torch.sum(ym * ym)
#     r_den = torch.sqrt(x_square_sum * y_square_sum)
#     r = r_num / r_den
#     return torch.mean(r)

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym,dim=0)
    x_square_sum = torch.sum(xm * xm,dim=0)
    y_square_sum = torch.sum(ym * ym,dim=0)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return torch.mean(r)


def pseudoAccuracy(y_true, y_pred, eps=1e-4):
    from sklearn.metrics import accuracy_score

    y_true[torch.where(torch.abs(y_true) < eps)] = 0
    y_true[torch.where(y_true < 0)] = -1
    y_true[torch.where(y_true > 0)] = 1
    # pos = y_true > 0
    # pos = pos.long()
    # neutral = y_true == 0.
    # neutral = neutral.long()
    # neg = y_true < 0
    # neg = neg.long()

    y_pred[torch.where(torch.abs(y_pred) < eps)] = 0
    y_pred[torch.where(y_pred < 0)] = -1
    y_pred[torch.where(y_pred > 0)] = 1
    # pred_pos = y_pred > 0
    # pred_pos = pred_pos.long()
    # pred_neutral = y_pred == 0.
    # pred_neutral = pred_neutral.long()
    # pred_neg = y_pred < 0
    # pred_neg = pred_neg.long()

    # tp = torch.sum(pos * pred_pos,axis=1)
    # tn = torch.sum(neg*pred_neg,axis=1)
    # tneutral = torch.sum(neutral* pred_neutral,axis=1)

    # acc = (tp+tn+tneutral)/y_true.shape[0]
    # acc = torch.mean(acc)
    acc = []
    for i in range(y_true.shape[0]):
        acc.append(accuracy_score(y_true[i, :].numpy(), y_pred[i, :].numpy()))

    return acc

def pseudoPresicion(y_true, y_pred, eps=1e-4):
    from sklearn.metrics import precision_score

    y_true[torch.where(torch.abs(y_true) < eps)] = 0
    y_true[torch.where(y_true < 0)] = -1
    y_true[torch.where(y_true > 0)] = 1
    # pos = y_true > 0
    # pos = pos.long()
    # neutral = y_true == 0.
    # neutral = neutral.long()
    # neg = y_true < 0
    # neg = neg.long()

    y_pred[torch.where(torch.abs(y_pred) < eps)] = 0
    y_pred[torch.where(y_pred < 0)] = -1
    y_pred[torch.where(y_pred > 0)] = 1
    # pred_pos = y_pred > 0
    # pred_pos = pred_pos.long()
    # pred_neutral = y_pred == 0.
    # pred_neutral = pred_neutral.long()
    # pred_neg = y_pred < 0
    # pred_neg = pred_neg.long()

    # tp = torch.sum(pos * pred_pos,axis=1)
    # tn = torch.sum(neg*pred_neg,axis=1)
    # tneutral = torch.sum(neutral* pred_neutral,axis=1)

    # acc = (tp+tn+tneutral)/y_true.shape[0]
    # acc = torch.mean(acc)
    prec = []
    for i in range(y_true.shape[0]):
        prec.append(precision_score(y_true[i, :].numpy(), y_pred[i, :].numpy(),average=None))

    return prec