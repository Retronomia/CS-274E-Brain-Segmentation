import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn as nn
import math
from utils import *
import os




def metrics(y_stat,y_mask,type,folder,epoch):
    '''generate DICE, AUPRC, AUROC'''
    ensure_folder_exists(folder)

    #DICE
    diceScore,diceThreshold = compute_dice_curve_recursive(
        y_stat,y_mask,
        plottitle=f"DICE vs L1 Threshold Curve for {type} Samples",
        folder = folder,
        file_name = f'dicePC_{epoch}',
        granularity=5
        )


    flat_stat = y_stat.flatten()
    flat_mask = y_mask.astype(bool).astype(int).flatten()
    #AUROC
    diff_auc = compute_roc(flat_stat,flat_mask,
        plottitle=f"ROC Curve for {type} Samples",
        folder = folder,
        file_name= f'rocPC_{epoch}'
    )

    #AUPRC
    diff_auprc = compute_prc(
        flat_stat,flat_mask,
        plottitle=f"Precision-Recall Curve for {type} Samples",
        folder = folder,
        file_name=f'prcPC_{epoch}'
    )

    del flat_stat,flat_mask
    return diff_auc,diff_auprc,diceScore,diceThreshold


#below here is modified from the brainweb github code
#I wanted to have the same dice algorithm
def compute_dice_curve_recursive(predictions,labels,folder=None,file_name=None,granularity=5,plottitle="DICE Curve"):
    '''Computes DICE and saves data'''
    datadict = dict()
    datadict["scores"], datadict["threshs"] = compute_dice_score(predictions, labels, granularity)
    datadict["best_score"], datadict["best_threshold"] = sorted(zip(datadict["scores"], datadict["threshs"]), reverse=True)[0]

    
    min_threshs, max_threshs = min(datadict["threshs"]), max(datadict["threshs"])
    buffer_range = math.fabs(min_threshs - max_threshs) * 0.02
    x_min, x_max = min(datadict["threshs"]) - buffer_range, max(datadict["threshs"]) + buffer_range
    fig = plt.figure()
    plt.plot(datadict["threshs"],datadict["scores"], color='darkorange', lw=2, label='DICE vs Threshold Curve')
    plt.xlim([x_min, x_max])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Thresholds')
    plt.ylabel('DICE Score')
    plt.title(plottitle)
    plt.legend(loc="lower right")
    plt.text(x_max - x_max * 0.01, 1, f'Best dice score at {datadict["best_threshold"]:.5f} with {datadict["best_score"]:.4f}', horizontalalignment='right',
                           verticalalignment='top')
    
    if folder and file_name:
        save_fig(fig,folder,file_name)
        save_json(datadict,folder,file_name,gz=True)
        plt.close(fig)
    else:
        plt.show()

    best_score = datadict["best_score"]
    best_threshold = datadict["best_threshold"]

    del datadict,fig,min_threshs,max_threshs,buffer_range,x_min,x_max

    return best_score, best_threshold


def dice(P, G):
    '''This calculates DICE using set cardinality formula'''
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    del psum,gsum,pgsum
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

        if decimal == granularity: #this stops at the granularity, so setting granularity=5 results in up to 4 decimal places.
            return _threshs, _scores

        for i, t in enumerate(xfrange(start, stop, (1.0 / (10.0 ** decimal)))):
            score = dice(np.where(predictions > t, 1, 0), labels)
            if i >= 2 and score <= _scores[i - 1] and not had_recursion:
                #this walks through previous step as well despite checking the 2nd and 3rd element.
                #Personally that's a little too greedy (this reruns a whole bunch of thresholds) but it works
                _subthreshs, _subscores = inner_compute_dice_curve_recursive(_threshs[i - 2], t, decimal + 1)
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

def compute_prc(predictions, labels, folder=None,file_name=None, plottitle="Precision-Recall Curve"):
    '''compute AUPRC, save data'''
    datadict = dict()
    datadict["precisions"], datadict["recalls"], _thresholds = precision_recall_curve(labels, predictions)
    datadict["auprc"] = average_precision_score(labels, predictions)

    fig = plt.figure()
    plt.step(datadict["recalls"], datadict["precisions"], color='b', alpha=0.2, where='post')
    plt.fill_between(datadict["recalls"],datadict["precisions"], step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{plottitle} (area = {datadict["auprc"]:.2f}.)')

    # save a pdf to disk
    if folder and file_name:
        save_fig(fig,folder,file_name)
        save_json(datadict,folder,file_name,gz=True)
        plt.close(fig)
    else:
        plt.show()

    auprc= datadict["auprc"]
    del datadict,fig,_thresholds
    return auprc

def compute_roc(predictions, labels, folder=None, file_name=None, plottitle="ROC Curve"):
    '''Compute AUROC, save data'''
    datadict = dict()
    datadict["_fpr"], datadict["_tpr"], datadict["thresholds"] =  roc_curve(labels, predictions)
    datadict["roc_auc"] = auc(datadict["_fpr"], datadict["_tpr"])
  
    fig = plt.figure()
    plt.plot(datadict["_fpr"], datadict["_tpr"], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % datadict["roc_auc"])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plottitle)
    plt.legend(loc="lower right")

    # save a pdf to disk
    if folder and file_name:
        save_fig(fig,folder,file_name)
        save_json(datadict,folder,file_name,gz=True)
        plt.close(fig)
    else:
        plt.show()

    roc_auc = datadict["roc_auc"]

    del datadict,fig

    return roc_auc