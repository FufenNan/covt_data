#!/bin/bash
#SBATCH --job-name=data_transform
#SBATCH --account=bdqf-dtai-gh      
#SBATCH --partition=ghx4            
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00             
#SBATCH --output=/u/yli8/jiaqi/CoVT/train/logs/data_transform_%j.out
#SBATCH --error=/u/yli8/jiaqi/CoVT/train/logs/data_transform_%j.err

source ~/.bashrc
conda activate covt
cd /u/yli8/jiaqi/CoVT/train/src
python data_transform.py
