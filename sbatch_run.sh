#!/bin/bash -l

#SBATCH --account mvaal --partition tier3
##SBATCH -n 1
##SBATCH -c 8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=300g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
##SBATCH --gpus-per-task=2

module purge
conda activate dplearning_clone

python3 -u main.py /home/bk9618/learning-with-noisy-labels-benchmark/data --dataset-name $dataset --epochs $epochs --batch-size $batch --learning-rate-weights $lr --version $version --projector $projector 
