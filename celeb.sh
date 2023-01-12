#!/bin/bash
#SBATCH --output=./celebdf.out
#SBATCH --error=./celebdf.err
#SBATCH --job-name=celeb
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --partition=fvl
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=gpu08


# conda activate fordataset
python CelebDF.py -path '/share/home/zhangchao/datasets_io03_ssd/Celeb-DF' -samples 120 -scale 1.3 -detector dlib -workers 8
# python CelebDF.py -path '/share/home/zhangchao/datasets_io03_ssd/Celeb-DF-v2' -samples 120 -scale 1.3 -detector dlib -workers 8