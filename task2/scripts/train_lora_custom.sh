#!/bin/sh

######## BEGIN TODO: experiment configuration ########
# Task 2-1: self-created "pncut" geometric paper-cut style dataset.
# Keep this configuration region as the only experiment-specific edit.
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="./sample_data/artistic-custom"
export OUTPUT_DIR="./runs/artistic_custom"
MIXED_PRECISION="fp16"
IMAGE_COLUMN="image"
CAPTION_COLUMN="text"
RESOLUTION=512
RANDOM_FLIP="true"
TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=100
VALIDATION_EPOCHS=10
CHECKPOINTING_STEPS=400
LEARNING_RATE="1e-04"
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
SEED=42
CHECKPOINTS_TOTAL_LIMIT=2
VALIDATION_PROMPT="a lighthouse by the sea in pncut style"
######## END TODO: experiment configuration ########

set -- \
  --mixed_precision="$MIXED_PRECISION" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --image_column="$IMAGE_COLUMN" \
  --caption_column="$CAPTION_COLUMN" \
  --resolution="$RESOLUTION" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --num_train_epochs="$NUM_TRAIN_EPOCHS" \
  --validation_epochs="$VALIDATION_EPOCHS" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  --learning_rate="$LEARNING_RATE" \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps="$LR_WARMUP_STEPS" \
  --seed="$SEED" \
  --checkpoints_total_limit="$CHECKPOINTS_TOTAL_LIMIT" \
  --validation_prompt="$VALIDATION_PROMPT"

if [ "$RANDOM_FLIP" = "true" ]; then
  set -- "$@" --random_flip
fi

accelerate launch --mixed_precision="$MIXED_PRECISION" train_lora.py "$@"
