#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# YV_
# Copied from: applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_node/run_1.3b_lora.sh
# Modified using: applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh

ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --num_padding_at_beginning 1 \
   --gradient_accumulation_steps 2 \
   --deepspeed \
   --actor_lora_dim 128 \
   --enable_hybrid_engine \
   --actor_gradient_checkpointing \
   --actor_dropout 0.0 \
   --output_dir $OUTPUT 
    &> $OUTPUT/training.log