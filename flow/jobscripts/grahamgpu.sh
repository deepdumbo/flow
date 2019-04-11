#!/bin/bash
#SBATCH --account=def-macgowan
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem=10000M              # memory (per node)
#SBATCH --time=0-12:00            # time (DD-HH:MM)

# graham cluster sample job script using GPU

# For TensorFlow
module load cuda/9.0.176 cudnn/7.0 python/3.6
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
source ~/tfenv/bin/activate
nvidia-smi

# Run python code here
python -u ./mognet_train.py