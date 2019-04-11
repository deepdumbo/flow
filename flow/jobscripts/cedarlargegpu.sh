#!/bin/bash
#SBATCH --account=def-macgowan
#SBATCH --nodes=1
#SBATCH --gres=gpu:lgpu:4         # Number of GPUs (per node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24        # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --mem=0                   # Request the full memory of the node
#SBATCH --time=0-02:00            # time (DD-HH:MM)

# cedar cluster job script using a large GPU node (whole node job)

module load cuda/9.0.176 cudnn/7.0 python/3.6
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
source ~/tfenv/bin/activate
nvidia-smi
python -u ./unet_xyt_train.py
