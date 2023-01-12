#!/bin/bash
#SBATCH --output=./tdf1.out # STDOUT
#SBATCH --error=./edf1.err # STDERR
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dfd1_l
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --partition=fvl
#SBATCH --qos=medium
#SBATCH --time=1-23:59:00
#SBATCH --constraint=2080

# conda activate fordataset
python DeeperForensics-1.0.py