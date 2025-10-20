#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --dataset SVAMP \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --save_dir 'math_llama1b_inst_no_sft' \
    --limit 10 \
    --num_beams 1 \
    --temperature 0.0 \
    --max_new_tokens 512