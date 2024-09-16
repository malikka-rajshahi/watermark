#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH --job-name=Batches-Gumbel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr6177@nyu.edu
#SBATCH --output=slurm_%j.out

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ./penv;
export PATH=./penv/bin:$PATH;

./experiments/scripts/testing-beta.sh test/ facebook/opt-1.3b french 0.01
