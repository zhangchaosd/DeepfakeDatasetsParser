#!/bin/bash
#SBATCH --output=./df1.0.out
#SBATCH --error=./df1.0.err
#SBATCH --job-name=df1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --mem=512GB
#SBATCH --nodes=1
#SBATCH --partition=scavenger
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=gpu16

# conda activate fordataset
python DeeperForensics-1.0.py -path '/share/home/zhangchao/datasets_io03_ssd/DeeperForensics-1.0' -samples 120 -scale 1.3 -detector dlib -workers 64