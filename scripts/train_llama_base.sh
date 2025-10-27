#!/bin/bash

# Modular configuration for training LLaMA base with LoRA adapters

# ---- Configurable parameters ----
# BASE_MODEL="meta-llama/Llama-3.2-1B"
# BASE_MODEL="meta-llama/Llama-3.2-3B"
BASE_MODEL="meta-llama/Llama-3.1-8B"
DATA_PATH="ft-training_set/math_7k.json"
BATCH_SIZE=128
MICRO_BATCH_SIZE=4
NUM_EPOCHS=3
LEARNING_RATE=3e-4
CUTOFF_LEN=512
VAL_SET_SIZE=120
ADAPTER_NAME="lora"
CUDA_DEVICE=0
OUTPUT_DIR="/data1/soheil/tce/trained_models/llama8b-lora-math_7k-3ep-128bs-lr3e-4"

# ---- Training command ----
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python finetune.py \
  --base_model "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --micro_batch_size $MICRO_BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --cutoff_len $CUTOFF_LEN \
  --val_set_size $VAL_SET_SIZE \
  --adapter_name $ADAPTER_NAME
