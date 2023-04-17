# MRI Segmentation with Autoencoders


## Requirements
Python version 3.7.13.

See requirements.txt for required libraries and versions.


## Instructions

### Datasets

Place data into data_folder:
- MedNIST (https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb)
- WMH Challenge (https://wmh.isi.uu.nl/data/)
    - modified using (https://github.com/FeliMe/brain_sas_baseline)

This stays as dsprites-dataset-master and is already in the repo:
- Dsprites (https://github.com/deepmind/dsprites-dataset)
### experiment.py Arguments

In addition to these arguments, modify experiment.py (see def loadobjective).

--exp_name: Data + experiment type. Default wmh_usp.
- mnist_sp
- mnist_usp
- mnist_semi
- wmh_sp
- wmh_usp
- wmh_semi

These are initially handled in data_loader.py.

--model_name: name of model to be used. Default AutoEnc
- AutoEnc
- AutoEnc_NoBottleneck
- ConstrainedAE
- SkipAE
- UNet
- VAE
- VQModel

To add your own model, simply add a python file + class of same name (python file can be lowercased for first character, class is always uppercased first letter) to the models folder.

Afterwards, modify run_experiment.py as necessary.

Then the new argument option is simply the name of the model.

--loss_name: name of loss to be used. Default L1_Loss.

If you are running a semisupervised, input supervised first then unsupervised, e.g. Custom_Loss L1_Loss 


Unsupervised:
- L1_Loss
- L2_Loss
- Bernoulli_Loss
- KL_Loss
- CAE_Loss
- VQ_Model_Loss

Supervised:
- Custom_Loss
- KL_SP_Loss
- CAE_SP_Loss
- VQ_Model_SP_Loss

To add your own loss, simply add a custom loss functiont to loss_funcs.py.

Afterwards, modify run_experiment.py as necessary.

Then the new argument option is simply the name of the function.

--tqdm: whether or not to use tqdm loading bars. No arguments, just use --tqdm if you want this enabled.

--comment: string. optional print out for remembering what an experiment was specifically trying to achieve when looking through log files.

### Running models

To run the code, run experiment.py with the desired arguments.

Experiments can be found in the experiments folder.

To open an experiment's .gz file, refer to open_gz.py.