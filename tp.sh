#!/bin/bash
#SBATCH --output=./ttar.out
#SBATCH --error=./etar.err
#SBATCH --job-name=tar_s
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=0
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --partition=scavenger
#SBATCH --time=3-00:00:00

cd /share/home/zhangchao/datasets_io03_ssd/ForgeryNet/Training/spatial_localize
tar -xvf 1.tar
tar -xvf 2.tar
tar -xvf 3.tar
tar -xvf 4.tar
tar -xvf 5.tar
tar -xvf 6.tar
tar -xvf 7.tar
tar -xvf 8.tar
tar -xvf 9.tar
tar -xvf 10.tar
tar -xvf 11.tar
tar -xvf 12.tar
tar -xvf 13.tar
tar -xvf 14.tar
tar -xvf 15.tar