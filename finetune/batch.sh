#!/bin/bash


export DATA_PATH="../species" #healthdata" #species_data" #../healthdata"  #/rnadata" #data" #dnabert2_lnc"  #data e.g., ./sample_data
export MAX_LENGTH=15 #1000 Please set the number as 0.25 * your sequence length.
                                                                                        # e.g., set it as 250 if your DNA sequences have 1000 nucleotide bases
                                                                                        # This is because the tokenized will reduce the sequence length by about 5 times
export LR=3e-5

# Training use DataParallel
python3 train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path  ${DATA_PATH} \
    --kmer -1 \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 5 \
    --save_steps 200 \
    --fp16 \
    --output_dir output/species2 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False


