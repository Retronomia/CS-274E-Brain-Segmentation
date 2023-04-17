#!/bin/bash

#SBATCH --job-name=mriproj_vqvae      # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=irosen@uci.edu  # Where to send mail	(for notification only)
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --nodelist=ava-m3          # select your node (or not)
#SBATCH --output=logs/job_%j.log   # output log

/home/irosen/.conda/envs/mriproj/bin/python -u experiment.py --exp_name wmh_sp --model_name VQModel --loss_name VQ_Model_SP_Loss --comment "vq model supervised"
