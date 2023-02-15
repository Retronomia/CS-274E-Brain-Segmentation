#Credit to:
#https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb for brain data plus how to load it
#https://github.com/deepmind/dsprites-dataset sprite dataset
#https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py for demonstrating how to set up Optuna code
#https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI for model inspiration/code for metric evaluation (Baur, et al. 2021)
import torch
import os
from monai.data import decollate_batch, DataLoader
from utils import *
from loss_funcs import *
import matplotlib.pyplot as plt
from data_loader import loadData
import time
import optuna
from scoring import metrics

def score(model,loader,loss_function,chosen_loss,device):
    '''get loss, reconstructions, masks, true image values'''
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y_mask = torch.tensor([], dtype=torch.long, device=device)
        y_true = torch.tensor([], dtype=torch.long, device=device)
        mu = torch.tensor([], dtype=torch.long, device=device)
        sigma = torch.tensor([], dtype=torch.long, device=device)
        z = torch.tensor([], dtype=torch.long, device=device)
        z_rec = torch.tensor([], dtype=torch.long, device=device)
        loss_values = []
        for data, ground_truths in loader:
            temp_mu,temp_sigma,temp_z,temp_z_rec = None,None,None,None
            val_images = data.to(device)
            truths = ground_truths.to(device)
            if chosen_loss=="KL_Loss" or chosen_loss=="KL_SP_Loss":
                temp_pred,temp_mu,temp_sigma = model(val_images)
                mu = torch.cat([mu, temp_mu], dim=0)
                sigma = torch.cat([sigma,temp_sigma], dim=0)
            elif chosen_loss=="CAE_Loss" or chosen_loss == "CAE_SP_Loss":
                temp_pred,temp_z,temp_z_rec = model(val_images)
                z = torch.cat([z, temp_z], dim=0)
                z_rec = torch.cat([z_rec,temp_z_rec], dim=0)
            else:
                temp_pred = model(val_images)
            y_pred = torch.cat([y_pred, temp_pred], dim=0)
            y_true = torch.cat([y_true, val_images], dim=0)
            y_mask = torch.cat([y_mask, truths], dim=0)
            
            if chosen_loss == "Custom_Loss":
                loss = loss_function(temp_pred,val_images,truths)
            elif chosen_loss == "KL_Loss":
                loss = loss_function(temp_pred,val_images,temp_mu,temp_sigma)
            elif chosen_loss == "KL_SP_Loss":
                loss = loss_function(temp_pred,val_images,truths,temp_mu,temp_sigma)
            elif chosen_loss == "CAE_Loss":
                loss = loss_function(temp_pred,val_images,temp_z,temp_z_rec)
            elif chosen_loss == "CAE_SP_Loss":
                loss = loss_function(temp_pred,val_images,truths,temp_z,temp_z_rec)
            else:
                loss = loss_function(temp_pred,val_images)
            loss_values.append(loss.item())
            del val_images,truths,loss,temp_pred,temp_mu,temp_sigma,temp_z,temp_z_rec
        del mu,sigma,z,z_rec
        return loss_values, y_pred,y_mask,y_true


def train(model,train_loader,optimizer,loss_function,loss_name,device):
    '''Run through one epoch of training dataset on model'''
    model.train()
    epoch_loss = 0
    step = 0
    num_steps = len(train_loader)
    for batch_data, ground_truths in train_loader:
        step += 1
        inputs = batch_data.to(device)

        truths = ground_truths.to(device)
        mu,sigma,z,z_rec = None,None,None,None

        optimizer.zero_grad()
        try:
            if loss_name == "KL_Loss":
                outputs,mu,sigma = model(inputs)
                loss = loss_function(outputs,inputs,mu,sigma)
            elif loss_name == "KL_SP_Loss":
                outputs,mu,sigma = model(inputs)
                loss = loss_function(outputs,inputs,truths,mu,sigma)
            elif loss_name == "Custom_Loss":
                outputs = model(inputs)
                loss = loss_function(outputs,inputs,truths)
            elif loss_name=="CAE_Loss":
                outputs,z,z_rec  = model(inputs)
                loss = loss_function(outputs,inputs,z,z_rec)
            elif loss_name=="CAE_SP_Loss":
                outputs,z,z_rec  = model(inputs)
                loss = loss_function(outputs,inputs,truths,z,z_rec)
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, inputs)
        except Exception as e:
            print(str(e))
            raise optuna.exceptions.TrialPruned()
        #print(outputs)
        loss.backward()
        optimizer.step()
    
        epoch_loss += loss.item()
        #print(f"{step}/{num_steps}, "f"train_loss: {loss.item():.4f}")
        
        del inputs,outputs,truths,mu,sigma,loss,z,z_rec

    epoch_loss /= step
    del step
    return epoch_loss


def objective(trial,loaderdict,device):
    exp_name = loaderdict['exp_name']

    train_x,val_x,test_x,loader = loadData(exp_name)

    batch_size = loaderdict['batch_size']
    if type(train_x) is tuple:
        train_ds = tuple(loader(t) for t in train_x)
        train_loaders = tuple(DataLoader(t, batch_size=batch_size,shuffle=True) for t in train_ds)
    else:
        train_ds = loader(train_x)
        train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True) 
        val_ds = loader(val_x)
        val_loader  = DataLoader(val_ds, batch_size=batch_size)

    del train_x,val_x,test_x

    model_name = loaderdict['model_name']
    modeltype = load_class(model_name)

    encoderdict = loaderdict['encoderdict']
    
    try:
        print("making model...")
        model = modeltype(**encoderdict).to(device)
    except Exception as e: 
        print(str(e))
        raise optuna.exceptions.TrialPruned()

    if type(loaderdict['loss_name']) is tuple:
        loss_names = loaderdict['loss_name']
        loss_functions = tuple( load_loss(l)() for l in loss_names)
    else:
        loss_name = loaderdict['loss_name']
        loss_function = load_loss(loss_name)()
    score_function=nn.L1Loss(reduction='none')

    
    optimizerdict = loaderdict['optimizerdict']
    learnerdict = loaderdict['learnerdict']
    
    optimizer = torch.optim.Adam(model.parameters(),**optimizerdict)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,**learnerdict)

    max_epochs = loaderdict['max_epochs']

    dir_name = Path('./experiments')
    folder_name = loaderdict['folder_name']
    save_json(loaderdict,dir_name/folder_name,'experiment_info',gz=False)
    #now should have everything :D
    best_metric=None
    val_interval = 1

    datadict = dict()
    datadict["val_losses"] = []
    datadict["train_losses"] = []

    for epoch in range(max_epochs):
        if type(loaderdict['loss_name']) is tuple:
            loss_name = loss_names[(epoch+1)%2]
            loss_function = loss_functions[(epoch+1)%2]
        if type(train_ds) is tuple:
            train_loader = train_loaders[(epoch+1)%2]
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        try:
            epoch_loss = train(model,train_loader,optimizer,loss_function,loss_name,device)
        except Exception as e:
            print(str(e))
            raise optuna.exceptions.TrialPruned()
        print(f"TRAIN: epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        datadict["train_losses"].append(epoch_loss)
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                loss_values, y_pred,y_mask,y_true = score(model,val_loader,loss_function,loss_name,device)
            
                y_stat = score_function(y_pred,y_true).cpu().numpy()
                y_mask = np.array([i.numpy() for i in decollate_batch(y_mask.cpu(), detach=False)])

                avg_reconstruction_err = np.mean(loss_values)
                datadict["val_losses"].append(avg_reconstruction_err)
                if best_metric == None or avg_reconstruction_err < best_metric:
                    best_metric = avg_reconstruction_err
                    best_metric_epoch = epoch + 1
                
                diff_auc,diff_auprc,diceScore,diceThreshold = metrics(y_stat,y_mask,"Validation",dir_name/folder_name,str(epoch+1))
                #
                y_true_np = np.array([i.numpy() for i in decollate_batch(y_true.cpu(),detach=False)])
                y_pred_np = np.array([i.numpy() for i in decollate_batch(y_pred.cpu())])
                if (epoch+1) % 1 == 0:
                    def plotims(nums):
                        fig = plt.figure(figsize=(10,10))
                        plt.rc('font', size=8)
                        def subp(n,num,maxr):
                            ax1 = fig.add_subplot(maxr,3,3*(n-1)+1)
                            ax1.imshow(y_stat[num][0], vmin=0, vmax=1)
                            ax1.grid(False)
                            ax1.set_xticks([])
                            ax1.set_yticks([])
                            ax1.set_title(f"Score Image({num})")

                            ax2 = fig.add_subplot(maxr,3,3*(n-1)+2)
                            ax2.imshow(y_pred_np[num][0], cmap="gray", vmin=0, vmax=1)
                            ax2.grid(False)
                            ax2.set_xticks([])
                            ax2.set_yticks([])
                            ax2.set_title(f"Reconstructed Image({num})")
                            
                            ax3 = fig.add_subplot(maxr,3,3*(n-1)+3)
                            ax3.imshow(y_true_np[num][0], cmap="gray", vmin=0, vmax=1)
                            ax3.set_xticks([])
                            ax3.set_yticks([])
                            ax3.grid(False)
                            ax3.set_title(f"Original Image({num})")
                        row = 1
                        maxr = len(nums)
                        for num in nums:
                            subp(row,num,maxr)
                            row+=1
                        save_fig(fig,dir_name/folder_name,f'training_{epoch+1}',suffix='.jpg')
                        plt.close(fig)
                        #plt.show()
                    plotims([0,2,22,30])
                del y_true_np,y_pred_np
                #
                scheduler.step()
                print(
                    f"current epoch: {epoch + 1}",
                    f"\ncurrent {loss_name} loss mean: {avg_reconstruction_err:.4f}",
                    f"\nAUROC: {diff_auc:.4f}",
                    f"AUPRC: {diff_auprc:.4f}",
                    f"DICE score: {diceScore:.4f}",
                    f"Threshold: {diceThreshold:.4f}",
                    f"\nbest {loss_name} loss mean: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                trial.report(avg_reconstruction_err, epoch)
                del loss_values,y_pred,y_mask,y_true,y_stat,avg_reconstruction_err
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    storeResults(model,dir_name/folder_name,best_metric,best_metric_epoch,datadict,epoch,loss_name)
                    del datadict,optimizer,loss_function,score_function #scheduler
                    raise optuna.exceptions.TrialPruned()
    #Run completes all the way
    storeResults(model,dir_name/folder_name,best_metric,best_metric_epoch,datadict,epoch,loss_name)
    del datadict,optimizer,loss_function,score_function #scheduler
    return best_metric
    

def storeResults(model,folder,best_metric,best_metric_epoch,datadict,epoch,loss_name):
    '''Store results from objective run'''
    print("Storing Results...")
    dat_name = "trainRunDat"
    save_json(datadict,folder,dat_name)

    #train
    train_name = f'trainRecErr'
    fig = plt.figure()
    eplen = range(1,epoch+2)
    plt.plot(eplen,datadict["train_losses"], color='darkorange', lw=2)
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_name}')
    plt.title(f'Avg Train Loss Error ({loss_name})')
    #plt.show()
    save_fig(fig,folder,train_name)
    plt.close(fig)

    #val
    val_name = f'valRecErr'
    fig = plt.figure()
    eplen = range(1,epoch+2)
    plt.plot(eplen,datadict["val_losses"], color='darkorange', lw=2)
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_name}')
    plt.title(f'Avg Val Loss Error ({loss_name})')
    #plt.show()
    save_fig(fig,folder,val_name)
    plt.close(fig)

    print(f"train completed, best_metric: {best_metric:.4f} "f"at epoch: {best_metric_epoch}")
    modelsavedloc = folder / "model.pth"
    torch.save(model.state_dict(), modelsavedloc)
    print(f"Saved model at {modelsavedloc}.")



def test(folder_name,parent_dir,device):
    parent_dir = Path(parent_dir)

    loaderdict = read_json(parent_dir/folder_name/'experiment_info.json',gz=False)
    exp_name = loaderdict['exp_name']

    train_x,val_x,test_x,loader = loadData(exp_name)

    batch_size = loaderdict['batch_size']
    if type(test_x) is tuple:
        test_x = test_x[0]
    test_ds = loader(test_x)
    test_loader = DataLoader(test_ds, batch_size=batch_size,shuffle=True) 

    del train_x,val_x,test_x

    model_name = loaderdict['model_name']
    modeltype = load_class(model_name)

    encoderdict = loaderdict['encoderdict']
    
    try:
        print("making model...")
        model = modeltype(**encoderdict).to(device)
        st_d = torch.load(parent_dir/folder_name/'model.pth')
        model.load_state_dict(st_d)
        model.eval()
    except Exception as e: 
        print(str(e))
        raise e

    if type(loaderdict['loss_name']) is tuple:
        loss_name= loaderdict['loss_name'][0]
    else:
        loss_name = loaderdict['loss_name']
    loss_function = load_loss(loss_name)()
    score_function= nn.L1Loss(reduction='none')
    #now should have everything
    with torch.no_grad():

        loss_values, y_pred,y_mask,y_true = score(model,test_loader,loss_function,loss_name,device)
    
        y_stat = score_function(y_pred,y_true).cpu().numpy()
        y_mask = np.array([i.numpy() for i in decollate_batch(y_mask.cpu(), detach=False)])

        avg_reconstruction_err = np.mean(loss_values)
        diff_auc,diff_auprc,diceScore,diceThreshold = metrics(y_stat,y_mask,"Test",parent_dir/folder_name,f"test")
        #
        y_true_np = np.array([i.numpy() for i in decollate_batch(y_true.cpu(),detach=False)])
        y_pred_np = np.array([i.numpy() for i in decollate_batch(y_pred.cpu())])

        epochnum = 0
        prevnum = 1
        while os.path.exists(parent_dir/folder_name/f"dicePC_{prevnum}.gz"):
            epochnum+=1
            prevnum+=1
        tempjson = read_json(parent_dir/folder_name/f"dicePC_{epochnum}.gz",gz=True)
        threshold = tempjson['best_threshold']

        def plotims(num):
            fig = plt.figure(figsize=(20,5))
            ax1 = fig.add_subplot(1,4,1)

            threshplot = y_stat[num][0].copy()
            threshplot[threshplot < threshold] = 0
            ax1.imshow(threshplot,vmin=0, vmax=1)
            ax1.grid(False)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"Thresholded L1 Image ({threshold})", size=20)

            ax1 = fig.add_subplot(1,4,2)
            ax1.imshow(y_stat[num][0],vmin=0, vmax=1)
            ax1.grid(False)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title("L1 Image",size=20)
            
            ax2 = fig.add_subplot(1,4,3)
            ax2.imshow(y_pred_np[num][0], cmap="gray", vmin=0, vmax=1)
            ax2.grid(False)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title("Reconstructed Image", size=20)
            
            ax3 = fig.add_subplot(1,4,4)
            ax3.imshow(y_true_np[num][0], cmap="gray", vmin=0, vmax=1)
            ax3.grid(False)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title("Original Image", size=20)
            #plt.show()
            save_fig(fig,parent_dir/folder_name,f'testimgs_{num}',suffix='.jpg')
            plt.close(fig)
        plotims(125)
        plotims(19)
        plotims(26)
        plotims(234)
        plotims(49)
        plotims(670)
        plotims(69)
        plotims(71)
        plotims(78)
        plotims(82)
        plotims(83)
        plotims(89)
        plotims(92)
        plotims(93)
        plotims(132)
        plotims(504)
        del y_true_np,y_pred_np
        #
        print(
            f"\n{loss_name} loss mean: {avg_reconstruction_err:.4f}",
            f"\nAUROC: {diff_auc:.4f}",
            f"\nAUPRC: {diff_auprc:.4f}",
            f"\nDICE score: {diceScore:.4f}",
            f"\nThreshold: {diceThreshold:.4f}",
        )
        del loss_values,y_pred,y_mask,y_true,y_stat
    del loss_function,score_function
    return avg_reconstruction_err