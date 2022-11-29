import os
import PIL
import tempfile
import random
from monai.utils import set_determinism
from monai.apps import download_and_extract
import numpy as np
import torch 
import matplotlib.pyplot as plt
import h5py
from monai.transforms.utils import rescale_array
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import math
import gc
import sys
import psutil


def memDebugger():
    print("CPU Usage (%):", psutil.cpu_percent())
    print(psutil.virtual_memory())
    pid = os.getpid()
    py = psutil.Process(pid)
    memory = py.memory_info()[0] / (2**30)
    print("Memory (GB):",memory)
    print("=========================")
    print("Torch objects:")
    #objs = gc.get_objects()
    for obj in gc.get_objects():
        #print(type(obj),sys.getsizeof(obj))
        if torch.is_tensor(obj):
            print(type(obj),obj.size(),sys.getsizeof(obj))
        #elif type(obj) is list:
        #    print(type(obj),len(obj))
        #elif type(obj) is dict:
        #    print(type(obj),len(obj),obj.keys())
        #else:
        #    print(type(obj))


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.image_files[index][0],self.image_files[index][1]


def resetSeeds():
    set_determinism(seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

def downloadMedNIST(root_dir):
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

def loadDSprites():
    resetSeeds()
    filename = "dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"

    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())

        mask_imgs = f['imgs'][()]
        latent_imgs = f['latents']

    num_key_frames = len(mask_imgs)

    np.random.shuffle(mask_imgs)
    print(f"Number of frames: {num_key_frames}")
    return mask_imgs

def loadMedNISTData(root_dir):
    downloadMedNIST(root_dir)
    data_dir = os.path.join(root_dir, "MedNIST")

    class_names = sorted(x for x in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, x)))

    image_files = dict()
    num_each = dict()
    for class_name in class_names:
        image_files[class_name] = [
            os.path.join(data_dir, class_name, x)
            for x in os.listdir(os.path.join(data_dir, class_name))
        ]
        num_each[class_name] = len(image_files[class_name])

    data_len = sum([v for v in num_each.values()])
    image_width, image_height = PIL.Image.open(image_files[class_names[0]][0]).size

    plt.subplots(2,3, figsize=(6, 6))
    for i, class_name in enumerate(class_names):
        sel_rand = np.random.randint(num_each[class_name])
        im = PIL.Image.open(image_files[class_name][sel_rand])
        arr = np.array(im)
        plt.subplot(2,3, i + 1)
        plt.xlabel(f"{class_name}:{sel_rand}")
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

    print(f"Total image count: {data_len}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    return image_files

def splitData(split_tuple: tuple,image_files: dict,mask_imgs: list,sprite_chances: tuple):
    resetSeeds()
    train_sp,val_sp,test_sp = sprite_chances
    class_name = 'HeadCT'

    train_frac,val_frac,test_frac = split_tuple
    if train_frac + val_frac + test_frac >1:
        raise ValueError(f"Proportions of class {class_name} ({train_frac},{val_frac},{test_frac}) sum to greater than 1.")
    
    dat_len = len(image_files[class_name])
    split_indices = np.arange(dat_len)
    np.random.shuffle(split_indices)

    train_indices = split_indices[0:int(dat_len*train_frac)]
    val_indices = split_indices[int(dat_len*train_frac):int(dat_len*train_frac)+int(dat_len*val_frac)]
    test_indices = split_indices[int(dat_len*train_frac)+int(dat_len*val_frac):int(dat_len*train_frac)+int(dat_len*val_frac)+int(dat_len*test_frac)]

    def genArray(type_indices,p_sprite,image_files,class_name,mask_imgs):
        made_arr = []
        for i in type_indices[0:int(len(type_indices)*p_sprite)]: #add sprite
            tempmask = mask_imgs[i]*1
            tempmask = np.expand_dims(tempmask,axis=0)
            tempimg = np.array(PIL.Image.open(image_files[class_name][i]))
            tempimg = rescale_array(tempimg,0,1)
            tempimg = np.expand_dims(tempimg,axis=0)
            tempimg[tempmask.astype(bool)] = 1   #random.uniform(0,1)
            made_arr.append((tempimg,tempmask))
        for i in type_indices[int(len(type_indices)*p_sprite):]: #no sprite
            tempimg = np.array(PIL.Image.open(image_files[class_name][i]))
            tempimg = rescale_array(tempimg,0,1)
            tempimg = np.expand_dims(tempimg, axis=0)
            makenorm = tempimg * 0
            made_arr.append((tempimg,makenorm))

        made_arr = np.array(made_arr)
        return made_arr

    add_train = genArray(train_indices,train_sp,image_files,class_name,mask_imgs)
    add_val = genArray(val_indices,val_sp,image_files,class_name,mask_imgs)
    add_test = genArray(test_indices,test_sp,image_files,class_name,mask_imgs)

    print(f"For {class_name} added {len(add_train)} train, {len(add_val)} val, {len(add_test)} test.")
    print(f"Train has {int(len(train_indices)*train_sp)}/{len(train_indices)} sprited images, Val has {int(len(val_indices)*val_sp)}/{len(val_indices)} sprited images, Test has {int(len(test_indices)*test_sp)}/{len(test_indices)} sprited images, ")
    
    return add_train,add_val,add_test

def json_reformatter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'{type(obj)} is not serializable')

def score(model,loader,loss_function,chosen_loss,device):
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y_mask = torch.tensor([], dtype=torch.long, device=device)
        y_true = torch.tensor([], dtype=torch.long, device=device)
        mu = torch.tensor([], dtype=torch.long, device=device)
        sigma = torch.tensor([], dtype=torch.long, device=device)
        loss_values = []
        for data, ground_truths in loader:
            temp_mu,temp_sigma = None,None
            val_images = data.to(device)
            truths = ground_truths.to(device)
            if chosen_loss=="KL":
                temp_pred,temp_mu,temp_sigma = model(val_images)
                mu = torch.cat([mu, temp_mu], dim=0)
                sigma = torch.cat([sigma,temp_sigma], dim=0)
            else:
                temp_pred = model(val_images)
            y_pred = torch.cat([y_pred, temp_pred], dim=0)
            y_true = torch.cat([y_true, val_images], dim=0)
            y_mask = torch.cat([y_mask, truths], dim=0)
            
            if chosen_loss == "custom" or chosen_loss == "custom2":
                loss = loss_function(temp_pred,val_images,truths)
            elif chosen_loss == "KL":
                loss = loss_function(temp_pred,val_images,temp_mu,temp_sigma)
            else:
                loss = loss_function(temp_pred,val_images)
            loss_values.append(loss.item())
            del val_images,truths,loss,temp_pred,temp_mu,temp_sigma

        del mu,sigma
        return loss_values, y_pred,y_mask,y_true

def metrics(y_stat,y_mask,type,fol,filename):
    folder = "images"

    folder = os.path.join(folder,fol)
    if not os.path.exists(folder):
        os.makedirs(folder)

    diceScore,diceThreshold = compute_dice_curve_recursive(
        y_stat,y_mask,
        plottitle=f"DICE vs L1 Threshold Curve for {type} Samples",
        filename=os.path.join(folder, f'dicePC_{filename}.png'),
        granularity=5
    )
    flat_stat = y_stat.flatten()
    flat_mask = y_mask.astype(bool).astype(int).flatten()
    #print("Computing AUROC:")
    diff_auc = compute_roc(flat_stat,flat_mask,
            plottitle=f"ROC Curve for {type} Samples",
            filename=os.path.join(folder, f'rocPC_{filename}.png'))

    #print("Computing AUPRC:")
    diff_auprc = compute_prc(
        flat_stat,flat_mask,
        plottitle=f"Precision-Recall Curve for {type} Samples",
        filename=os.path.join(folder, f'prcPC_{filename}.png')
    )

    del flat_stat,flat_mask
    return diff_auc,diff_auprc,diceScore,diceThreshold


def train(model,train_loader,optimizer,loss_function,chosen_loss,device):
    model.train()
    epoch_loss = 0
    step = 0
    num_steps = len(train_loader)
    for batch_data, ground_truths in train_loader:
        step += 1
        inputs = batch_data.to(device)
        truths = ground_truths.to(device)
        mu,sigma = None,None

        optimizer.zero_grad()
        if chosen_loss == "KL":
            outputs,mu,sigma = model(inputs)
        else:
            outputs = model(inputs)
        if chosen_loss == "custom" or chosen_loss == "custom2":
            loss = loss_function(outputs,inputs,truths)
        elif chosen_loss=="KL":
            loss = loss_function(outputs,inputs,mu,sigma)
        else:
            loss = loss_function(outputs, inputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{num_steps}, "f"train_loss: {loss.item():.4f}")
        
        del inputs,outputs,truths,mu,sigma,loss
        #if step >= 2:
        #    break

    epoch_loss /= step
    del step
    return epoch_loss

def objective(trial,model,lr,betas,weight_decay,chosen_loss,gamma,encoderdict,train_loader,val_loader,model_name):
    best_metric=None
    max_epochs = 10
    val_interval = 1
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=betas,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    if chosen_loss=="L1":
        loss_function = nn.L1Loss()
        score_function = nn.L1Loss(reduction='none')
    elif chosen_loss=="KL":
        def kl_loss(pred,real,mu,sigma):
            l1_loss = nn.L1Loss(reduction='none')(pred,real)
            rec = torch.sum(l1_loss, axis=[1, 2, 3])
            kl = .5 * torch.sum(torch.square(mu)+torch.square(sigma)-torch.log(torch.square(sigma))-1,axis=1)
            return torch.mean(rec+kl)

        loss_function = kl_loss
        score_function = nn.L1Loss(reduction='none')
    elif chosen_loss=="custom": 
        def anomaly_score(pred,real):
            return torch.abs(pred-real)
        def maskloss(pred,real,mask,reduction='mean'):
            def formula(pred,real,mask):
                anom = anomaly_score(pred,real)
                anom_mod = anom.clone()
                anom_mod[anom_mod==0]=1e-6
                return (1 - mask)*anom + mask/anom_mod
            res = formula(pred,real,mask)
            if reduction=='mean':
                return torch.mean(res)
            else:
                return res
        loss_function = maskloss
        score_function = anomaly_score
    elif chosen_loss=="custom2": 
        def anomaly_score(pred,real):
            return torch.abs(pred-real)
        def maskloss(pred,real,mask,reduction='mean'):
            def formula(pred,real,mask):
                anom = anomaly_score(pred,real)
                #anom_mod = anom.clone()
                anom[anom==1]=.99
                anom[anom==0]=1e-4
                return -(1 - mask)*torch.log(1-anom) - mask*torch.log(anom)
            res = formula(pred,real,mask)
            if reduction=='mean':
                return torch.mean(res)
            else:
                return res
        loss_function = maskloss
        score_function = anomaly_score
    else:
        raise ValueError(f"chosen loss {chosen_loss} invalid")


    customname = f"{trial.number}-({lr},{gamma},{encoderdict['num_layers']},{encoderdict['kernel_size']},"\
        f"{encoderdict['stride']},{encoderdict['padding']},{encoderdict['dilation']},{chosen_loss},{betas},{weight_decay})"
    modelfolder = "./models"
    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)
    
    datadict = dict()
    datadict["val_losses"] = []
    datadict["train_losses"] = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = train(model,train_loader,optimizer,loss_function,chosen_loss,device)
        print(f"TRAIN: epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        datadict["train_losses"].append(epoch_loss)
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                loss_values, y_pred,y_mask,y_true = score(model,val_loader,loss_function,chosen_loss,device)
            
                y_stat = score_function(y_pred,y_true).cpu().numpy()
                y_mask = np.array([i.numpy() for i in decollate_batch(y_mask.cpu(), detach=False)])

                avg_reconstruction_err = np.mean(loss_values)
                datadict["val_losses"].append(avg_reconstruction_err)
                if best_metric == None or avg_reconstruction_err < best_metric:
                    best_metric = avg_reconstruction_err
                    best_metric_epoch = epoch + 1
                
                diff_auc,diff_auprc,diceScore,diceThreshold = metrics(y_stat,y_mask,"Validation",customname,str(epoch+1))
                scheduler.step()
                print(
                    f"current epoch: {epoch + 1}",
                    f"\ncurrent {chosen_loss} loss mean: {avg_reconstruction_err:.4f}",
                    f"\nAUROC: {diff_auc:.4f}",
                    f"\nAUPRC: {diff_auprc:.4f}",
                    f"\nDICE score: {diceScore:.4f}",
                    f"\nThreshold: {diceThreshold:.4f}",
                    f"\nbest {chosen_loss} loss mean: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                trial.report(avg_reconstruction_err, epoch)
                del loss_values,y_pred,y_mask,y_true,y_stat,avg_reconstruction_err
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    storeResults(model,model_name,modelfolder,best_metric,best_metric_epoch,customname,datadict,epoch,chosen_loss)
                    del datadict,optimizer,scheduler,loss_function,score_function
                    raise optuna.exceptions.TrialPruned()
    #Run completes all the way
    storeResults(model,model_name,modelfolder,best_metric,best_metric_epoch,customname,datadict,epoch,chosen_loss)
    del datadict,optimizer,scheduler,loss_function,score_function
    return best_metric



def storeResults(model,model_name,modelfolder,best_metric,best_metric_epoch,customname,datadict,epoch,chosen_loss):
    print("Storing Results...")
    mname = f'trainRecErr.png'
    folder = './images'
    folder = os.path.join(folder,customname)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder,mname) +".json", "w") as fp:
        json.dump(datadict,fp, indent = 4) 
    fig = plt.figure()
    eplen = range(1,epoch+2)
    plt.plot(eplen,datadict["train_losses"], color='darkorange', lw=2)
    plt.xlabel('Epochs')
    plt.ylabel(f'{chosen_loss}')
    plt.title(f'Avg Train Reconstruction Error ({chosen_loss})')
    #plt.show()
    # save a pdf to disk
    fig.savefig(os.path.join(folder,mname))
    plt.close(fig)
    #val
    mname = f'valRecErr.png'
    folder = './images'
    folder = os.path.join(folder,customname)
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig = plt.figure()
    eplen = range(1,epoch+2)
    plt.plot(eplen,datadict["val_losses"], color='darkorange', lw=2)
    plt.xlabel('Epochs')
    plt.ylabel(f'{chosen_loss}')
    plt.title(f'Avg Val Reconstruction Error ({chosen_loss})')
    #plt.show()
    # save a pdf to disk
    fig.savefig(os.path.join(folder,mname))
    plt.close(fig)
    print(f"train completed, best_metric: {best_metric:.4f} "f"at epoch: {best_metric_epoch}")
    modelsavedloc = os.path.join(modelfolder,f"{model_name}_{customname}.pth")
    torch.save(model.state_dict(), modelsavedloc)
    print(f"Saved model at {modelsavedloc}.")



#below here is modified from the brainweb github code
#I wanted to have the same dice algorithm

def compute_dice_curve_recursive(predictions, labels, filename=None, plottitle="DICE Curve", granularity=5):
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
    #plt.show()

    # save a pdf to disk
    if filename:
        fig.savefig(filename)
        with open(filename+".json", "w") as fp:
            json.dump(datadict,fp, indent = 4,default=json_reformatter) 

    plt.close(fig)
    
    best_score = datadict["best_score"]
    best_threshold = datadict["best_threshold"]

    del datadict,fig,min_threshs,max_threshs,buffer_range,x_min,x_max

    return best_score, best_threshold

def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    #print(f"pgsum {pgsum}, psum {psum}, gsum {gsum}")
    del psum,gsum,pgsum
    return score

def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

def compute_dice_score(predictions, labels, granularity):
    def inner_compute_dice_curve_recursive(start, stop, decimal):
        _threshs = []
        _scores = []
        had_recursion = False

        if decimal == granularity:
            return _threshs, _scores

        for i, t in enumerate(xfrange(start, stop, (1.0 / (10.0 ** decimal)))):
            #print(f"Trying {i},{t}")
            score = dice(np.where(predictions > t, 1, 0), labels)
            if i >= 2 and score <= _scores[i - 1] and not had_recursion:
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

def compute_prc(predictions, labels, filename=None, plottitle="Precision-Recall Curve"):
    datadict = dict()
    datadict["precisions"], datadict["recalls"], datadict["thresholds"] = precision_recall_curve(labels, predictions)
    datadict["auprc"] = average_precision_score(labels, predictions)

    fig = plt.figure()
    plt.step(datadict["recalls"], datadict["precisions"], color='b', alpha=0.2, where='post')
    plt.fill_between(datadict["recalls"],datadict["precisions"], step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{plottitle} (area = {datadict["auprc"]:.2f}.)')
    #plt.show()

    # save a pdf to disk
    if filename:
        fig.savefig(filename)
        with open(filename+".json", "w") as fp:
            json.dump(datadict,fp, indent = 4,default=json_reformatter) 

    plt.close(fig)

    auprc= datadict["auprc"]
    del datadict,fig
    return auprc

def compute_roc(predictions, labels, filename=None, plottitle="ROC Curve"):
    datadict = dict()
    datadict["_fpr"], datadict["_tpr"], datadict["thresholds"] = roc_curve(labels, predictions)
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
    #plt.show()

    # save a pdf to disk
    if filename:
        fig.savefig(filename)
        with open(filename+".json", "w") as fp:
            json.dump(datadict,fp, indent = 4,default=json_reformatter) 
                
    plt.close(fig)

    roc_auc = datadict["roc_auc"]

    del datadict,fig

    return roc_auc