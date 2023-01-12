#!/bin/bash
#SBATCH --output=./deepfTIMIT.out
#SBATCH --error=./deepfTIMIT.err
#SBATCH --job-name=timit
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --partition=fvl
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00

# conda activate fordataset
python DeepfakeTIMIT.py -path '/share/home/zhangchao/datasets_io03_ssd/DeepfakeTIMIT' -samples 120 -scale 1.3 -detector dlib -workers 8