#!/usr/bin/env bash
# ./scripts/run_nlp.sh train features

METHODS='features sidetune finetune scratch'

PHASE=$1
METHOD=$2
SIZE=${3:-1}

if [ "$SIZE" == "1" ]; then
    SIZE=""
    NUM_EPOCHS=2
    CACHE_DIR="/mnt/data/squad2"
elif [ "$SIZE" == "few125" ]; then
    NUM_EPOCHS=100
    CACHE_DIR="/mnt/data/squad2${SIZE}"
else
    NUM_EPOCHS=5
    CACHE_DIR="/mnt/data/squad2_${SIZE}"
fi

if [ "$PHASE" == "train" ]; then
    CMD="python -m torch.distributed.launch --master_port=6011 --nproc_per_node=3 ./nlp/run_squad.py \
            --model_type bert \
            --model_name_or_path bert-large-uncased-whole-word-masking \
            --do_train \
            --do_eval \
            --do_lower_case \
            --train_file /mnt/data/squad2_${SIZE}/train${SIZE}-v2.0.json \
            --predict_file /mnt/data/squad2_${SIZE}/dev-v2.0.json \
            --version_2_with_negative \
            --learning_rate 3e-5 \
            --num_train_epochs ${NUM_EPOCHS} \
            --max_seq_length 384 \
            --doc_stride 128 \
            --per_gpu_eval_batch_size=1 \
            --per_gpu_train_batch_size=1 \
            --save_steps 10000 \
            --cache_dir ${CACHE_DIR} \
            --output_dir /mnt/models/wwm_uncased_${METHOD}_squad${SIZE}/ \
            --${METHOD}"
elif [ "$PHASE" == "eval" ] || [ "$PHASE" == "test" ]; then
     CMD="python -m nlp.run_squad \
            --model_type bert \
            --model_name_or_path /mnt/models/wwm_uncased_${METHOD}_squad${SIZE}/ \
            --config_name /mnt/models/wwm_uncased_${METHOD}_squad${SIZE}/config.json \
            --do_eval \
            --do_lower_case \
            --train_file /mnt/data/squad2_${SIZE}/train${SIZE}-v2.0.json \
            --predict_file /mnt/data/squad2_${SIZE}/dev-v2.0.json \
            --version_2_with_negative \
            --learning_rate 3e-5 \
            --num_train_epochs 2 \
            --max_seq_length 384 \
            --doc_stride 128 \
            --output_dir /mnt/models/wwm_uncased_${METHOD}_squad${SIZE}/ \
            --per_gpu_eval_batch_size=2 \
            --cache_dir ${CACHE_DIR} \
            --${METHOD}"
else
    echo BAD PHASE
    exit
fi

echo $CMD
bash -c "$CMD"
