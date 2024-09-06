#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH --job-name=Experiment6-Mod
#SBATCH --mail-type=END
#SBATCH --mail-user=mr6177@nyu.edu
#SBATCH --output=slurm_%j.out

conda activate /scratch/projects/hegdelab/mr6177/watermark/penv
./experiments/scripts/experiment-6-mod.sh exp6_mod/opt facebook/opt-1.3b french
