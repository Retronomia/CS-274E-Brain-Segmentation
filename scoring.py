import numpy as np
import torch.nn as nn
import math
from utils import *
from torchmetrics.functional import dice, average_precision, auroc


def metrics(y_stat, y_mask, mtype, folder, epoch):
    '''generate DICE, AUPRC, AUROC'''
    ensure_folder_exists(folder)

    # remove anything not 0 or 1
    flat_mask = torch.flatten(y_mask).int()
    tot_orig_vals = len(flat_mask)
    sel_vals = torch.logical_or(flat_mask == 0, flat_mask == 1)
    flat_mask = flat_mask[sel_vals]
    flat_stat = torch.flatten(y_stat)
    flat_stat = flat_stat[sel_vals]

    def imgwise_DICE(q, y_stat, flat_mask, sel_vals):
        quants = torch.quantile(y_stat, q, dim=0)
        y_segmented = (y_stat > quants).float()
        flat_segmented = y_segmented.flatten()
        flat_segmented = flat_segmented[sel_vals]
        testdice = dice(flat_segmented.int(), flat_mask,
                        average='none', num_classes=2)[1]
        return testdice

    qthresh = [.1, .25, .5, .75, .8, .90]
    dscores = [imgwise_DICE(t, y_stat, flat_mask, sel_vals) for t in qthresh]

    # DICE
    diceScore, diceThreshold = compute_dice_curve_recursive(
        flat_stat, flat_mask,
        plottitle=f"DICE vs L1 Threshold Curve for {mtype} Samples",
        folder=folder,
        file_name=f'dicePC_{epoch}',
        granularity=5
    )

    # AUROC
    diff_auc = compute_roc(flat_stat, flat_mask,
                           plottitle=f"ROC Curve for {mtype} Samples",
                           folder=folder,
                           file_name=f'rocPC_{epoch}'
                           )

    # AUPRC
    diff_auprc = compute_prc(
        flat_stat, flat_mask,
        plottitle=f"Precision-Recall Curve for {mtype} Samples",
        folder=folder,
        file_name=f'prcPC_{epoch}'
    )

    return diff_auc, diff_auprc, diceScore, diceThreshold, dscores, qthresh

# below here is modified from the brainweb github code
# I wanted to have the same dice algorithm


def compute_dice_curve_recursive(predictions, labels, folder=None, file_name=None, granularity=5, plottitle="DICE Curve"):
    '''Computes DICE and saves data'''
    datadict = dict()
    datadict["scores"], datadict["threshs"] = compute_dice_score(
        predictions, labels, granularity)
    datadict["best_score"], datadict["best_threshold"] = sorted(
        zip(datadict["scores"], datadict["threshs"]), reverse=True)[0]

    min_threshs, max_threshs = min(
        datadict["threshs"]), max(datadict["threshs"])
    buffer_range = math.fabs(min_threshs - max_threshs) * 0.02
    x_min, x_max = min(datadict["threshs"]) - \
        buffer_range, max(datadict["threshs"]) + buffer_range

    best_score = datadict["best_score"]
    best_threshold = datadict["best_threshold"]

    del datadict, min_threshs, max_threshs, buffer_range, x_min, x_max

    return best_score, best_threshold


def old_dice(P, G):
    '''This calculates DICE using set cardinality formula'''

    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    del psum, gsum, pgsum
    return score


def xfrange(start, stop, step):
    '''Generator for generating float steps evenly'''
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


def compute_dice_score(predictions, labels, granularity):
    def inner_compute_dice_curve_recursive(start, stop, decimal):
        _threshs = []
        _scores = []
        had_recursion = False

        # this stops at the granularity, so setting granularity=5 results in up to 4 decimal places.
        if decimal == granularity:
            return _threshs, _scores

        for i, t in enumerate(xfrange(start, stop, (1.0 / (10.0 ** decimal)))):
            temp_preds = torch.zeros_like(predictions)
            mask = predictions > t
            temp_preds[mask] = 1
            score = dice(temp_preds.int(), labels,
                         average='none', num_classes=2)[1]
            #l = labels.cpu().detach().numpy().flatten()
            #p = predictions.cpu().detach().numpy().flatten()
            #tdice = old_dice(np.where(p > t, 1, 0), l)
            #testdice = 1.0 - scipy.spatial.distance.dice(l,np.where(p > t, 1, 0))
            #print(f"Orig:{score}, New:{testdice}, dice:{tdice}")
            if i >= 2 and score <= _scores[i - 1] and not had_recursion:
                # this walks through previous step as well despite checking the 2nd and 3rd element.
                # Personally that's a little too greedy (this reruns a whole bunch of thresholds) but it works
                _subthreshs, _subscores = inner_compute_dice_curve_recursive(
                    _threshs[i - 2], t, decimal + 1)
                _threshs.extend(_subthreshs)
                _scores.extend(_subscores)
                had_recursion = True
            _scores.append(score)
            _threshs.append(t)

        return _threshs, _scores

    threshs, scores = inner_compute_dice_curve_recursive(0, 1.0, 1)
    sorted_pairs = sorted(zip(threshs, scores))
    threshs, scores = list(zip(*sorted_pairs))
    return scores, threshs


def compute_prc(predictions, labels, folder=None, file_name=None, plottitle="Precision-Recall Curve"):
    '''compute AUPRC, save data'''
    auprc = average_precision(predictions, labels, task='binary')
    return auprc


def compute_roc(predictions, labels, folder=None, file_name=None, plottitle="ROC Curve"):
    '''Compute AUROC, save data'''
    roc_auc = auroc(predictions, labels, task='binary')
    return roc_auc
