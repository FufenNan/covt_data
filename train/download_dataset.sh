#!/bin/bash
#SBATCH --job-name=download_covt
#SBATCH --account=bdqf-dtai-gh      
#SBATCH --partition=ghx4            
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=04:00:00             
#SBATCH --output=/u/yli8/jiaqi/CoVT/train/logs/download_covt_%j.out
#SBATCH --error=/u/yli8/jiaqi/CoVT/train/logs/download_covt_%j.err

source ~/.bashrc
conda activate covt

# 设置 HuggingFace cache 到 NVMe
export HF_HOME=/u/yli8/jiaqi/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# 切到 NVMe 目录
cd /u/yli8/jiaqi/CoVT/train

# 运行 Python 下载
python download_dataset.py


# huggingface repo download Wakals/CoVT-Dataset --repo-type dataset --revision main --local-dir /u/yli8/jiaqi/CoVT-Dataset