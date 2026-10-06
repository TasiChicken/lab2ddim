#!/bin/sh

######## BEGIN TODO: experiment configuration ########
# Optional public-dataset route retained from the starter.
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
DATASET_CONFIG_NAME=""
export OUTPUT_DIR="./runs/sd-naruto-model-lora"
MIXED_PRECISION="fp16"
IMAGE_COLUMN="image"
CAPTION_COLUMN="text"
RESOLUTION=512
RANDOM_FLIP="true"
TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=100
VALIDATION_EPOCHS=1
CHECKPOINTING_STEPS=2000
LEARNING_RATE="1e-04"
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
SEED=42
CHECKPOINTS_TOTAL_LIMIT=2
VALIDATION_PROMPT="a cute bear in Naruto style"
######## END TODO: experiment configuration ########

set -- \
  --mixed_precision="$MIXED_PRECISION" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --dataset_name="$DATASET_NAME" \
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

if [ -n "$DATASET_CONFIG_NAME" ]; then
  set -- "$@" --dataset_config_name="$DATASET_CONFIG_NAME"
fi

if [ "$RANDOM_FLIP" = "true" ]; then
  set -- "$@" --random_flip
fi

accelerate launch --mixed_precision="$MIXED_PRECISION" train_lora.py "$@"
