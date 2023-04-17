from data_loader import loadData
from utils import *
from monai.data import decollate_batch, DataLoader
from loss_funcs import load_loss
from run_experiment import objective,test
import time
import optuna
from optuna.trial import TrialState
import argparse
import pickle
import string
import datetime
folder_name = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def loadobjective(trial):
    global folder_name,device,args_model_name,args_exp_name,args_loss_name
    exp_name = args_exp_name 
    model_name = args_model_name
    batch_size = 4
 

    # Generate values
    optimizerdict = dict()
    optimizerdict['lr'] =  .005 #trial.suggest_float("lr", 0.0008, 0.002, step=0.0001) #0.001
    b1= 0.95 #trial.suggest_float("b1", 0.70, 0.95, step=0.05) #0.71
    b2= 0.97 #trial.suggest_float("b2", 0.90, 0.99, step=0.01) #0.965 
    optimizerdict['betas']=(b1,b2)
    optimizerdict['weight_decay']= 0.0001 #trial.suggest_float("weight_decay", 0, 0.0003,step=0.0001) #0.0002
    
    learnerdict = dict()
    learnerdict['gamma']= .95 #trial.suggest_float("gamma",.7,1,step=.05) #.75
    learnerdict['step_size']= 5
    
    encoderdict = dict()
    encoderdict['num_layers'] = 5 #4 #5
    encoderdict['kernel_size'] = [1,2,3,4,5]   #[1,2,3,4] #[1,2,3,4,5] 
    encoderdict['stride'] = [1,2,1,1,1]   #[1,2,1,1] #[1,2,1,1,1] 
    encoderdict['padding'] = [1,2,2,2,2]  #[1,2,2,2] #[1,2,2,2,2] 
    encoderdict['dilation'] = [1,1,1,1,1]  #[1,1,1,1] #[1,1,1,1,1] 
    encoderdict['latent'] = 128 
    encoderdict['image_shape']= 240 #64 #240


    loss_name = args_loss_name


    loaderdict = dict()
    loaderdict['use_tqdm'] = args_use_tqdm
    loaderdict['trial_num'] = trial.number
    loaderdict['exp_name'] = exp_name
    loaderdict['model_name'] = model_name
    loaderdict['loss_name'] = loss_name
    loaderdict['batch_size']= batch_size
    loaderdict['optimizerdict'] = optimizerdict
    loaderdict['learnerdict'] = learnerdict
    loaderdict['encoderdict'] = encoderdict
    max_epochs = 500
    loaderdict['max_epochs'] = max_epochs


    #time_stamp = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    folder_name=study_name + '-' + str(loaderdict['trial_num'])+"-"+exp_name+"-"+model_name+'-'+loss_name #+'-'+time_stamp
    loaderdict['folder_name']=folder_name

    print(f"Testing model with parameters: {loaderdict}")
    #del train_x,val_x,test_x,train_ds,val_ds
    return objective(trial,loaderdict,device)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='wmh_usp', help='experiment name')
    parser.add_argument('--model_name', type=str, default='AutoEnc', help='model name')
    parser.add_argument('--loss_name', type=str, default='L1_Loss', help='loss name')
    parser.add_argument('--tqdm',action='store_true',help='use tqdm')
    parser.add_argument('--comment',type=str,default='None',help='reminder print out for logs')
    args = parser.parse_args()
    args_model_name = args.model_name
    args_exp_name = args.exp_name
    args_loss_name = args.loss_name
    args_use_tqdm = args.tqdm
    print("Comment:",args.comment)

    study_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
 
    vae_study = optuna.create_study(direction="minimize",study_name=study_name)

    start = time.time()
    vae_study.optimize(loadobjective, n_trials=1,timeout=60*60*24,gc_after_trial=True,show_progress_bar=False)

    pruned_trials = vae_study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = vae_study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(vae_study.trials))

    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = vae_study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open('experiments/'+study_name+'.pickle','wb') as f:
        pickle.dump(vae_study,f)

    best_folder = study_name + '-' + str(trial.number)+"-"+args_exp_name+"-"+args_model_name+'-'+args_loss_name
    #folder_name = "0-wmh_usp-AutoEnc-L1_Loss-2023-02-21 08-53-14"
    test(best_folder,'experiments',device)
    end = time.time()
    print("Time to complete:",datetime.timedelta(seconds=end-start))