import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn as nn
import math
#import SimpleITK as sitk
from utils import *
import scipy.spatial
import os
from scipy.stats import bootstrap
from torchmetrics.functional import dice,average_precision,auroc,recall,f1_score


def metrics(y_stat,y_mask,mtype,folder,epoch):
    '''generate DICE, AUPRC, AUROC'''
    ensure_folder_exists(folder)

    flat_mask = torch.flatten(y_mask).int() #y_mask.astype(bool).astype(int).flatten()
    tot_orig_vals = len(flat_mask)
    sel_vals= torch.logical_or(flat_mask ==0, flat_mask==1) #np.logical_or(flat_mask ==0, flat_mask==1)
    flat_mask = flat_mask[sel_vals]
    flat_stat = torch.flatten(y_stat) #y_stat.flatten()
    flat_stat = flat_stat[sel_vals]
    

    #num = 0
    #denom = y_mask.shape[0]
    #for img in y_mask:
    #    if any(e == 1 for e in np.unique(img)):
    #        num+=1
    #q = torch.sum(flat_mask)/tot_orig_vals #num/denom
    #print("Quantile:",q.item())
    def imgwise_DICE(q,y_stat,flat_mask,sel_vals):
        quants = torch.quantile(y_stat,q,dim=0)
        y_segmented = (y_stat > quants).float()
        flat_segmented = y_segmented.flatten()
        flat_segmented = flat_segmented[sel_vals]
        testdice = dice(flat_segmented.int(),flat_mask,average='none',num_classes=2)[1]
        return testdice
        #print("Alt dice method: ",testdice.item())

    qthresh = [.1,.25,.5,.75,.8,.90]
    dscores = [imgwise_DICE(t,y_stat,flat_mask,sel_vals) for t in qthresh]
    #y_thresh = (y_stat > diceThreshold).astype(int)
    #print(getlAVD(y_mask,y_thresh))
    

    #DICE
    diceScore,diceThreshold = compute_dice_curve_recursive(
        flat_stat,flat_mask,
        plottitle=f"DICE vs L1 Threshold Curve for {mtype} Samples",
        folder = folder,
        file_name = f'dicePC_{epoch}',
        granularity=5
        )
 
    #AUROC
    diff_auc = compute_roc(flat_stat,flat_mask,
        plottitle=f"ROC Curve for {mtype} Samples",
        folder = folder,
        file_name= f'rocPC_{epoch}'
    )

    #AUPRC
    diff_auprc = compute_prc(
        flat_stat,flat_mask,
        plottitle=f"Precision-Recall Curve for {mtype} Samples",
        folder = folder,
        file_name=f'prcPC_{epoch}'
    )
    # del flat_stat,flat_mask
    #f_recall = recall(flat_stat > diceThreshold,flat_mask,task='binary')
    #f_f1_score = f1_score(flat_stat > diceThreshold,flat_mask,task='binary')

    return diff_auc,diff_auprc,diceScore,diceThreshold,dscores,qthresh#,f_recall,f_f1_score



def getlAVD(testImage, resultImage):   
    """Volume statistics."""
    # Compute statistics of both images
    predSum = np.sum(resultImage,axis=(1,2,3))
    testSum = np.sum(testImage,axis=(1,2,3))
    print(testSum)
    avd = np.abs(np.log(predSum-testSum/testSum))*100
    print(avd.shape)
    #AVDs = float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100  
    #fail here plz
    meanAVD = np.mean(avd)
    bootstrap_ci = bootstrap((avd,), np.median, confidence_level=0.95,random_state=1, method='percentile',n_resamples=2000)
    confLow,confUpper = bootstrap_ci.confidence_interval
    #view 95% boostrapped confidence interval
    print(confLow,confUpper)
    #return meanAVD,confLow,confUpper
    return meanAVD,confLow,confUpper

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
    '''fig = plt.figure()
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
        plt.show()'''

    best_score = datadict["best_score"]
    best_threshold = datadict["best_threshold"]

    del datadict,min_threshs,max_threshs,buffer_range,x_min,x_max

    return best_score, best_threshold


def old_dice(P, G):
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
            temp_preds = torch.zeros_like(predictions)
            mask = predictions > t
            temp_preds[mask] = 1
            score = dice(temp_preds.int(),labels,average='none',num_classes=2)[1]
            #l = labels.cpu().detach().numpy().flatten()
            #p = predictions.cpu().detach().numpy().flatten()
            #tdice = old_dice(np.where(p > t, 1, 0), l)
            #testdice = 1.0 - scipy.spatial.distance.dice(l,np.where(p > t, 1, 0)) 
            #print(f"Orig:{score}, New:{testdice}, dice:{tdice}")
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
    #datadict = dict()
    #datadict["precisions"], datadict["recalls"], _thresholds = precision_recall_curve(labels, predictions)
    #datadict["auprc"] = average_precision_score(labels, predictions)

    '''fig = plt.figure()
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
        plt.show()'''

    #auprc= datadict["auprc"]
    #del datadict,_thresholds
    auprc = average_precision(predictions, labels,task='binary')
    return auprc

def compute_roc(predictions, labels, folder=None, file_name=None, plottitle="ROC Curve"):
    '''Compute AUROC, save data'''
    #datadict = dict()
    #datadict["_fpr"], datadict["_tpr"], datadict["thresholds"] =  roc_curve(labels, predictions)
    #datadict["roc_auc"] = auc(datadict["_fpr"], datadict["_tpr"])
  
    '''fig = plt.figure()
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
        plt.show()'''

    #roc_auc = datadict["roc_auc"]
    #del datadict
    roc_auc = auroc(predictions, labels,task='binary')
    return roc_auc