#!/bin/bash
#SBATCH --account=def-macgowan
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem=8G                  # memory (per node)
#SBATCH --time=0-00:20            # time (DD-HH:MM)

# beluga cluster sample job script using GPU

# Submit this shell script within the Anaconda environment
# $ conda activate fire
# $ sbatch belugagpu.sh

# flow repository
export PYTHONPATH="$HOME/flow:$PYTHONPATH"

nvidia-smi

# Run python code here
which python
python ~/flow/flow/mains/neonatal_recon/train_unet.py