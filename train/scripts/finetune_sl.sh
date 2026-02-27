#!/bin/bash
#SBATCH --job-name=covt_finetune
#SBATCH --account=bdqf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=40:00:00
#SBATCH --output=/u/yli8/jiaqi/CoVT/train/logs/finetune_%j.out
#SBATCH --error=/u/yli8/jiaqi/CoVT/train/logs/finetune_%j.err

mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi
export WANDB_API_KEY="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"

source ~/.bashrc
conda activate covt
cd /u/yli8/jiaqi/CoVT/train
export BNB_DISABLE=1

set -e

NUM_DEVICES=1
BATCH_PER_DEVICE=1
GLOBAL_BATCH_SIZE=12
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR_STAGE1="output/lora_vision_test/lora_stage1"
MERGED_STAGE1_MODEL="output/lora_merged/lora_stage1_merged"
OUT_DIR_STAGE234="output/lora_vision_test/lora_stage234"
FINAL_MERGED_MODEL="output/lora_merged/lora_stage234_merged"
DATA_PATH="/u/yli8/jiaqi/CoVT-Dataset/data_new.json"
IMAGE_FOLDER="/u/yli8/jiaqi/CoVT-Dataset/images"
VISUAL_MODEL_ID="['sam', 'depth', 'dino']"

mkdir -p "$OUT_DIR_STAGE1"
mkdir -p "$MERGED_STAGE1_MODEL"
mkdir -p "$OUT_DIR_STAGE234"
mkdir -p "$FINAL_MERGED_MODEL"


echo "==== [1/4] First stage training: max_steps=6000 ===="
MODEL_NAME="$BASE_MODEL" \
MODEL_PATH="$BASE_MODEL" \
GPU_IDS="$GPU_IDS" \
NUM_DEVICES="$NUM_DEVICES" \
BATCH_PER_DEVICE="$BATCH_PER_DEVICE" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
OUTPUT_DIR="$OUT_DIR_STAGE1" \
RUN_NAME="stage1_train" \
STAGE_0_STEP=6000 \
STAGE_1_STEP=6000 \
STAGE_2_STEP=6000 \
VQA_ONLY_STAGE=6000 \
MAX_STEPS=6000 \
DATA_PATH="$DATA_PATH" \
IMAGE_FOLDER="$IMAGE_FOLDER" \
VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/run.sh

echo "==== [2/4] First merge LoRA ===="
MODEL_NAME="$BASE_MODEL" \
MODEL_PATH="$OUT_DIR_STAGE1" \
SAVE_MODEL_PATH="$MERGED_STAGE1_MODEL" \
VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/merge_lora.sh

echo "==== [3/4] Joint training of stage 2/3/4: max_steps=10000 ===="
MODEL_NAME="$MERGED_STAGE1_MODEL" \
MODEL_PATH="$MERGED_STAGE1_MODEL" \
GPU_IDS="$GPU_IDS" \
NUM_DEVICES="$NUM_DEVICES" \
BATCH_PER_DEVICE="$BATCH_PER_DEVICE" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
OUTPUT_DIR="$OUT_DIR_STAGE234" \
RUN_NAME="stage234_train" \
STAGE_0_STEP=0 \
STAGE_1_STEP=3000 \
STAGE_2_STEP=6000 \
VQA_ONLY_STAGE=8000 \
MAX_STEPS=10000 \
DATA_PATH="$DATA_PATH" \
IMAGE_FOLDER="$IMAGE_FOLDER" \
VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/run.sh

echo "==== [4/4] Second merge LoRA (final) ===="
MODEL_NAME="$MERGED_STAGE1_MODEL" \
MODEL_PATH="$OUT_DIR_STAGE234" \
SAVE_MODEL_PATH="$FINAL_MERGED_MODEL" \
VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/merge_lora.sh

echo "==== All processes completed, final model: $FINAL_MERGED_MODEL ===="
echo "Job finished at: $(date)"