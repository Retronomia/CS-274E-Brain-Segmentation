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
import scipy
from scoring import metrics
from tqdm import tqdm

def predict_vals(model,data,ground_truths,loss_function,chosen_loss,device):
    temp_mu,temp_sigma,temp_z,temp_z_rec = None,None,None,None
    val_images = data.to(device)
    truths = ground_truths.to(device)
    if chosen_loss=="KL_Loss" or chosen_loss=="KL_SP_Loss":
        temp_pred,temp_mu,temp_sigma = model(val_images)
    elif chosen_loss=="CAE_Loss" or chosen_loss == "CAE_SP_Loss":
        temp_pred,temp_z,temp_z_rec = model(val_images)
    else:
        temp_pred = model(val_images.float())

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
    
    return temp_pred,loss.item()

def score(model,loader,loss_function,chosen_loss,score_function,filepath,epoch,mtype,device):
    '''get loss, reconstructions, masks, true image values'''
    model.eval()
    madeexc = False
    with torch.no_grad():
        loss_values = torch.tensor([], dtype=torch.float32, device=device)
        diff_aucs = torch.tensor([], dtype=torch.float32, device=device)
        diff_auprcs = torch.tensor([], dtype=torch.float32, device=device)
        diceScores = torch.tensor([], dtype=torch.float32, device=device)
        diceThresholds = torch.tensor([], dtype=torch.float32, device=device)
        imgdScores = torch.tensor([], dtype=torch.float32, device=device)
        imgdThresholds = torch.tensor([], dtype=torch.float32, device=device)
        #recalls = torch.tensor([], dtype=torch.float32, device=device)
        #f1_scores = torch.tensor([], dtype=torch.float32, device=device)
        batchnum = 0
        for data, ground_truths in tqdm(loader, desc="Predictions and Scoring"):
            batchnum+=1
            val_images = data.to(device)
            truths = ground_truths.to(device)
            temp_pred,loss = predict_vals(model,val_images,truths,loss_function,chosen_loss,device)

            y_stat = score_function(temp_pred,val_images) #.cpu().numpy()
            #y_mask = np.array([i.numpy() for i in decollate_batch(truths.cpu(), detach=False)])

            diff_auc,diff_auprc,diceScore,diceThreshold,dscores,qthresh = metrics(y_stat,truths,mtype,filepath,epoch)

            loss_values = torch.cat([loss_values, torch.tensor([loss],device=device)], dim=0)
            diff_aucs = torch.cat([diff_aucs, torch.tensor([diff_auc],device=device)], dim=0)
            diff_auprcs = torch.cat([diff_auprcs, torch.tensor([diff_auprc],device=device)], dim=0)
            diceScores = torch.cat([diceScores, torch.tensor([diceScore],device=device)], dim=0)
            diceThresholds = torch.cat([diceThresholds, torch.tensor([diceThreshold],device=device)], dim=0)
            imgdScores = torch.cat([imgdScores, torch.tensor([dscores],device=device)], dim=0)
            #recalls = torch.cat([recalls, torch.tensor([recall],device=device)], dim=0)
            #f1_scores = torch.cat([f1_scores, torch.tensor([f1_score],device=device)], dim=0)
            if len(imgdThresholds) == 0:
                imgdThresholds =  torch.tensor(qthresh,device=device) #torch.cat([imgdThresholds, torch.tensor([qthresh],device=device)], dim=0)
            if not madeexc:
                def plotims(num):
                    fig = plt.figure(figsize=(20,5))

                    #bscore = torch.max(dscores)
                    bscoreidx = torch.argmax(torch.Tensor(dscores))
                    bquant = qthresh[bscoreidx]
                    quants = torch.quantile(y_stat,bquant,dim=0)
                    threshplot = (y_stat[num][0] > quants)
                    mask = torch.where(truths[num][0]==2)
                    threshplot[mask]=0
                    ax1 = fig.add_subplot(2,3,1)
                    ax1.imshow(threshplot[0].cpu(),vmin=0, vmax=1)
                    ax1.grid(False)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_title(f"Thresholded L1 Image (Q={bquant})",size=20)


                    ax1 = fig.add_subplot(2,3,2)
                    threshplot = y_stat[num][0].clone()
                    threshplot[threshplot < diceThreshold] = 0
                    threshplot[threshplot >= diceThreshold] = 1
                    threshplot[mask]==0
                    ax1.imshow(threshplot.cpu(),vmin=0, vmax=1)
                    ax1.grid(False)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_title(f"Thresholded L1 Image ({diceThreshold:.5f})", size=20)

                    ax1 = fig.add_subplot(2,3,3)
                    ax1.imshow(y_stat[num][0].cpu(),vmin=0, vmax=1)
                    ax1.grid(False)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_title("L1 Image",size=20)
                    
                    ax2 = fig.add_subplot(2,3,4)
                    ax2.imshow(temp_pred[num][0].cpu(), cmap="gray", vmin=0, vmax=1)
                    ax2.grid(False)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_title("Reconstructed Image", size=20)
                    
                    ax3 = fig.add_subplot(2,3,5)
                    ax3.imshow(val_images[num][0].cpu(), cmap="gray", vmin=0, vmax=1)
                    ax3.grid(False)
                    ax3.set_xticks([])
                    ax3.set_yticks([])
                    ax3.set_title("Original Image", size=20)

                    ax4 = fig.add_subplot(2,3,6)
                    truthplt = truths[num][0].clone()
                    truthplt[mask] = 0
                    ax4.imshow(truthplt.cpu(), cmap="gray", vmin=0, vmax=1)
                    ax4.grid(False)
                    ax4.set_xticks([])
                    ax4.set_yticks([])
                    ax4.set_title("Image Mask", size=20)

                    #plt.show()
                    save_fig(fig,filepath,f'{mtype}({epoch})_{batchnum}_{num}',suffix='.jpg')
                    plt.close(fig)
                for plotnum in [0,1,2]: #125,19,26,234,49,670,69,71,78,82,83,89,92,93,132,504
                    try:
                        plotims(plotnum)
                    except Exception as e:
                        print("WARNING:",str(e))
                        pass
                madeexc = True
            

        statdict = dict()
        #statdict['losses'] = loss_values.cpu().detach().numpy()
        statdict['mean_losses'] = torch.mean(loss_values).item()
        statdict['std_losses'] = torch.std(loss_values).item()
      
        #statdict['auc'] = diff_aucs.cpu().detach().numpy()
        statdict['mean_auc'] = torch.mean( diff_aucs).item()
        statdict['std_auc'] = torch.std( diff_aucs).item()

        #statdict['auprc'] = diff_auprcs.cpu().detach().numpy()
        statdict['mean_auprc'] = torch.mean(diff_auprcs).item()
        statdict['std_auprc'] = torch.std(diff_auprcs).item()

        #statdict['dice_thresholds'] = diceThresholds.cpu().detach().numpy()
        statdict['mean_dice_thresholds'] = torch.mean(diceThresholds).item()
        statdict['std_dice_thresholds'] = torch.std(diceThresholds).item()

        #statdict['dice_scores'] =  diceScores.cpu().detach().numpy()
        statdict['mean_dice_scores'] = torch.mean(diceScores).item()
        statdict['std_dice_scores'] = torch.std(diceScores).item()
        #imgdScores
        #statdict['imgdice_scores'] = imgdScores.cpu().detach().numpy()
        statdict['mean_imgdice_scores'] = torch.mean(imgdScores ,axis=0).cpu().detach().numpy()
        statdict['std_imgdice_scores'] = torch.std(imgdScores ,axis=0).cpu().detach().numpy()
        statdict['imgdice_thresholds'] = imgdThresholds.cpu().detach().numpy()

        #statdict['mean_recall'] = torch.mean(recalls).item()
        #statdict['std_recall'] = torch.std(recalls).item()

        #statdict['mean_f1_scores'] = torch.mean(f1_score).item()
        #statdict['std_f1_scores'] = torch.std(f1_score).item()
        #imgdThresholds
        save_json(statdict,filepath,f'data_summary_{epoch}')

        return statdict

def train(model,train_loader,optimizer,loss_function,loss_name,device):
    '''Run through one epoch of training dataset on model'''
    model.train()
    epoch_loss = 0
    step = 0
    num_steps = len(train_loader)
    for batch_data, ground_truths in tqdm(train_loader,desc="Training"):
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
                outputs = model(inputs.float()) #FIX
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
    del test_x

    batch_size = loaderdict['batch_size']
    if type(train_x) is tuple:
        train_ds = tuple(loader(t) for t in train_x)
        train_loaders = tuple(DataLoader(t, batch_size=batch_size,shuffle=True) for t in train_ds)
        val_x = val_x[0]
    else:
        train_ds = loader(train_x)
        train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True)  #REMOVE POST
    val_ds = loader(val_x)
    val_loader  = DataLoader(val_ds, batch_size=batch_size,shuffle=False)

    del train_x,val_x

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,**learnerdict)

    max_epochs = loaderdict['max_epochs']

    dir_name = Path('./experiments')
    folder_name = loaderdict['folder_name']
    save_json(loaderdict,dir_name/folder_name,'experiment_info',gz=False)
    #now should have everything :D
    best_metric=None
    best_metric_epoch=0
    val_interval = 100

    datadict = dict()
    datadict["val_losses"] = []
    datadict["train_losses"] = []
    datadict["val_epochs"] = []
    datadict["learning_rates"] = []

    for epoch in range(max_epochs):
        datadict["learning_rates"].append(scheduler.get_last_lr())
        if type(loaderdict['loss_name']) is tuple:
            loss_name = loss_names[(epoch+1)%len(loaderdict['loss_name'])]
            loss_function = loss_functions[(epoch+1)%len(loaderdict['loss_name'])]
        if type(train_ds) is tuple:
            train_loader = train_loaders[(epoch+1)%len(train_ds)]
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
            datadict["val_epochs"].append(epoch + 1)
            model.eval()
            with torch.no_grad():
                statdict = score(model,val_loader,loss_function,loss_name,score_function,dir_name/folder_name,f'{epoch+1}',"Validation",device)

                avg_reconstruction_err = np.mean(statdict['mean_losses'])
                datadict["val_losses"].append(avg_reconstruction_err)
                if best_metric == None or avg_reconstruction_err < best_metric:
                    best_metric = avg_reconstruction_err
                    best_metric_epoch = epoch + 1

                '''if (epoch+1) % 1 == 0:
                    def plotims(nums):
                        fig = plt.figure(figsize=(10,10))
                        plt.set_cmap('gray')
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
                    plotims([0,2,22,30])'''

                print(
                    f"current epoch: {epoch + 1}",
                    f"\ncurrent {loss_name} loss mean: {avg_reconstruction_err:.4f}",
                    f"\nAUROC mean: {statdict['mean_auc']:.4f}, std: {statdict['std_auc']:.4f}",
                    f"\nAURPC mean: {statdict['mean_auprc']:.4f}, std: {statdict['std_auprc']:.4f}",
                    f"\nDICE score mean: {statdict['mean_dice_scores']:.4f}, std: {statdict['std_dice_scores']:.4f}",
                    f"\nDICE threshold mean: {statdict['mean_dice_thresholds']:.4f}, std: {statdict['std_dice_thresholds']:.4f}",
                    f"\nimg-wise DICE score means: {statdict['mean_imgdice_scores']}",
                    f"\nimg-wise DICE score stds: {statdict['std_imgdice_scores']}",
                    f"\nimg-wise DICE quantiles:{ statdict['imgdice_thresholds']}",
                    #f"\nf1-score mean: {statdict['mean_f1_scores']:.4f}, std: {statdict['std_f1_scores']:.4f}",
                    #f"\nrecall mean: {statdict['mean_recall']:.4f}, std: {statdict['std_recall']:.4f}",
                    f"\nbest {loss_name} loss mean: {best_metric if best_metric != None else 0 :.4f} at epoch: {best_metric_epoch}"
                )
                trial.report(avg_reconstruction_err, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    storeResults(model,dir_name/folder_name,best_metric,best_metric_epoch,datadict,epoch,loss_name)
                    del datadict,optimizer,loss_function,score_function, statdict
                    raise optuna.exceptions.TrialPruned()
        scheduler.step()
    #Run completes all the way
    storeResults(model,dir_name/folder_name,best_metric,best_metric_epoch,datadict,epoch,loss_name)
    del datadict,optimizer,loss_function,score_function, statdict
    return best_metric
    

def storeResults(model,folder,best_metric,best_metric_epoch,datadict,epoch,loss_name):
    '''Store results from objective run'''
    print("Storing Results...")
    dat_name = "trainRunDat"
    save_json(datadict,folder,dat_name)

    #LR
    lr_name = f'Learning Rate'
    fig = plt.figure()
    eplen = range(1,epoch+2)
    plt.plot(eplen,datadict["learning_rates"], color='purple', lw=2)
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_name}')
    plt.title(f'Learning Rate')
    #plt.show()
    save_fig(fig,folder,lr_name)
    plt.close(fig)

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

    
    #LR
    lr_loss_name = f'Loss vs Learning Rate'
    fig = plt.figure()
    plt.plot(datadict["learning_rates"],datadict["train_losses"], color='blue', lw=2)
    plt.xlabel('Learning Rate')
    plt.ylabel(f'Avg Train Loss Error ({loss_name})')
    plt.title(f'Loss vs Learning Rate')
    #plt.show()
    save_fig(fig,folder,lr_loss_name)
    plt.close(fig)

    #val
    val_name = f'valRecErr'
    fig = plt.figure()
    plt.plot(datadict["val_epochs"],datadict["val_losses"], color='darkorange', lw=2)
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_name}')
    plt.title(f'Avg Val Loss Error ({loss_name})')
    #plt.show()
    save_fig(fig,folder,val_name)
    plt.close(fig)

    print(f"train completed, best_metric: {best_metric if best_metric != None else 0 :.4f} "f"at epoch: {best_metric_epoch}")
    modelsavedloc = folder / "model.pth"
    torch.save(model.state_dict(), modelsavedloc)
    print(f"Saved model at {modelsavedloc}.")


def test(folder_name,parent_dir,device):
    parent_dir = Path(parent_dir)

    loaderdict = read_json(parent_dir/folder_name/'experiment_info.json',gz=False)
    exp_name = loaderdict['exp_name']

    train_x,val_x,test_x,loader = loadData(exp_name)
    del train_x,val_x

    batch_size = loaderdict['batch_size']
    if type(test_x) is tuple:
        test_x = test_x[0]
    test_ds = loader(test_x)
    test_loader = DataLoader(test_ds, batch_size=batch_size,shuffle=False) 

    del test_x

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
        statdict = score(model,test_loader,loss_function,loss_name,score_function,parent_dir/folder_name,"test","Test",device)

        avg_reconstruction_err = np.mean(statdict['mean_losses'])

        '''epochnum = 0
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
        for plotnum in [125,19,26,234,49,670,69,71,78,82,83,89,92,93,132,504]:
            try:
                plotims(plotnum)
            except:
                pass
        del y_true_np,y_pred_np'''
        #
        print(
            f"\n{loss_name} loss mean: {avg_reconstruction_err:.4f}",
            f"\nAUROC mean: {statdict['mean_auc']:.4f}, std: {statdict['std_auc']:.4f}",
            f"\nAURPC mean: {statdict['mean_auprc']:.4f}, std: {statdict['std_auprc']:.4f}",
            f"\nDICE score mean: {statdict['mean_dice_scores']:.4f}, std: {statdict['std_dice_scores']:.4f}",
            f"\nDICE threshold mean: {statdict['mean_dice_thresholds']:.4f}, std: {statdict['std_dice_thresholds']:.4f}",
            f"\nimg-wise DICE score means: {statdict['mean_imgdice_scores']}",
            f"\nimg-wise DICE score stds: {statdict['std_imgdice_scores']}",
            f"\nimg-wise DICE quantiles:{ statdict['imgdice_thresholds']}",
            #f"\nf1-score mean: {statdict['mean_f1_scores']:.4f}, std: {statdict['std_f1_scores']:.4f}",
            #f"\nrecall mean: {statdict['mean_recall']:.4f}, std: {statdict['std_recall']:.4f}",
        )
    del loss_function,score_function
    return avg_reconstruction_err