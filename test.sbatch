#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH --job-name=Experiment6-Test
#SBATCH --mail-type=END
#SBATCH --mail-user=mr6177@nyu.edu
#SBATCH --output=slurm_%j.out

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ./penv;
export PATH=./penv/bin:$PATH;

python experiments/c4-experiment.py --save test/test-mod.p --n 256 --method gumbel_mod --batch_size 20 --m 50 --rt_translate --model facebook/opt-1.3b --language french --buffer_tokens 100 --seed 1
