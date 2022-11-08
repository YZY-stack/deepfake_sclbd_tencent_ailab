#!/bin/bash
#SBATCH -J train_deepfake
#SBATCH -p p-A100 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=pgpu22
#SBATCH --gres=gpu:1

python3 train.py
