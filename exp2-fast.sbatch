#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=compute  # OPTIONAL can be removed to run on big/normal partition
#SBATCH --job-name=Experiment2-Fast

source activate watermark
./experiments/scripts/experiment-2.sh exp2/fast facebook/opt-1.3b 35
