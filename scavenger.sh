#!/bin/bash
#SBATCH --output=./tfg.out
#SBATCH --error=./efg.err
#SBATCH --job-name=fg_s
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --partition=scavenger
#SBATCH --time=3-00:00:00


# conda activate fordataset
python ForgeryNet.py