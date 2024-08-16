#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# YV_

# Copied from applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_node/run_1.3b_lora.sh

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-350m \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0.1 \
   --num_train_epochs 8 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 0 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --only_optimize_lora \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
