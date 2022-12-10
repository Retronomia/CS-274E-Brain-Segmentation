Python Version: 3.7.13
Required Libraries:
-PIL (Pillow 9.3.0)
-tempfile
-seaborn==0.12.1
-monai==1.0.1
-numpy==1.21.6
-optuna==3.0.3
-torch==1.13.0
-matplotlib==3.5.3
-plotly==5.11.0
-h5py==3.7.0
-sklearn==0.0.post1 (scikit-image==0.19.3, scikit-learn==1.0.2)
-torchvision==0.14.0
-nbformat==5.7.0
=================================
Instructions:

To run the code, open the autoencoder.ipynb and make sure to run all of the cells up to "Required File Loads End Here."
Now, you can rerun any of the training sections of the notebook.

To create a custom model:
-copy and modify one the loadobjective functions used in each section
-copy and modify the code that immediately comes after those functions
To test a model:
-test(model_name,root_dir)
root_dir is one of the variables that should be loaded in at the top of the notebook
model_name is the name of the model folder in ./images (or ./models minus the file extension)
Valid experiment_names:
-Unsupervised: ["ae_usp_.1","vae_usp_.1","cae_usp_.1","nobottleae_usp_.1","skipae_usp_.1"]
-Supervised: ["ae_sp_.1","vae_sp_.1","cae_sp_.1","nobottleae_sp_.1","skipae_sp_.1"]
-Semi-Supervised: ["ae_semi_.1","vae_semi_.1","cae_semi_.1","nobottleae_semi_.1","skipae_semi_.1"]
Warning: If you rerun an experiment with the exact same parameters it will overwrite all of the preexisting files for the model (if there are any)