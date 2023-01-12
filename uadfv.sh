#!/bin/bash
#SBATCH --output=./uadfv.out
#SBATCH --error=./uadfv.err
#SBATCH --job-name=ua_f
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --partition=fvl
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00

# conda activate fordataset
python UADFV.py -path '/share/home/zhangchao/datasets_io03_ssd/UADFV' -samples 120 -scale 1.3 -detector dlib -workers 8