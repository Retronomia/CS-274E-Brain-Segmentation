from data_loader import loadData
from utils import *
from monai.data import decollate_batch, DataLoader
from loss_funcs import load_loss
from run_experiment import objective,test
import time
import optuna
from optuna.trial import TrialState

folder_name = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadobjective(trial):
    global folder_name,device
    exp_name = "wmh_usp"
    model_name = "AutoEnc"
    batch_size = 8


    # Generate values
    optimizerdict = dict()
    optimizerdict['lr'] = 0.001
    b1= 0.71
    b2= 0.965 
    optimizerdict['betas']=(b1,b2)
    optimizerdict['weight_decay']=0.0002
    
    learnerdict = dict()
    learnerdict['gamma']=.75
    
    encoderdict = dict()
    encoderdict['num_layers'] = 4
    encoderdict['kernel_size'] = [3,2,4,2] 
    encoderdict['stride'] = [1,2,1,1] 
    encoderdict['padding'] = [1,1,1,2] 
    encoderdict['dilation'] = [1,1,1,1] 
    encoderdict['latent'] = 128 
    encoderdict['image_shape']=240


    loss_name = "L1_Loss"


    loaderdict = dict()
    loaderdict['trial_num'] = trial.number
    loaderdict['exp_name'] = exp_name
    loaderdict['model_name'] = model_name
    loaderdict['loss_name'] = loss_name
    loaderdict['batch_size']= batch_size
    loaderdict['optimizerdict'] = optimizerdict
    loaderdict['learnerdict'] = learnerdict
    loaderdict['encoderdict'] = encoderdict
    max_epochs = 1
    loaderdict['max_epochs'] = max_epochs


    time_stamp = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    folder_name=str(loaderdict['trial_num'])+"-"+exp_name+"-"+model_name+'-'+loss_name+'-'+time_stamp
    loaderdict['folder_name']=folder_name

    print(f"Testing model with parameters: {loaderdict}")
    #del train_x,val_x,test_x,train_ds,val_ds
    return objective(trial,loaderdict,device)


vae_study = optuna.create_study(direction="minimize",study_name="teststudy")

vae_study.optimize(loadobjective, n_trials=1,timeout=14400,gc_after_trial=True,show_progress_bar=False)

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


test(folder_name,'experiments',device)