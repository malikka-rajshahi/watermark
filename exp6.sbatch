#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=compute  # OPTIONAL can be removed to run on big/normal partition
#SBATCH --job-name=Experiment6

source activate watermark
./experiments/scripts/experiment-6.sh exp6/opt facebook/opt-1.3b french
