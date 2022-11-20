#!/bin/bash
#SBATCH --output=./tdfd5.out # STDOUT
#SBATCH --error=./edfd5.err # STDERR
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dfd5_m
#SBATCH --mem=30GB
#SBATCH --nodes=1
#SBATCH --partition=fvl
#SBATCH --qos=medium
#SBATCH --time=1-23:59:00
#SBATCH --constraint=2080

# conda activate fordataset
python FaceForensics++.py 5