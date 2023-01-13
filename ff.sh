#!/bin/bash
#SBATCH --output=./ff.out
#SBATCH --error=./ff.err
#SBATCH --job-name=ff
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --partition=fvl
#SBATCH --qos=high
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=gpu08


# conda activate fordataset
python FaceForensics++.py -path '/share/home/zhangchao/datasets_io03_ssd/ff++' -subset FF -samples 120 -scale 1.3 -detector dlib -workers 32
# python FaceForensics++.py -path '/share/home/zhangchao/datasets_io03_ssd/ff++' -subset DFD -samples 120 -scale 1.3 -detector dlib -workers 32