#   一些指令
conda create -n envname python=3.11 -y
conda activate swag_env
ssh peter900218@100.110.140.73
peter900218
scp ~/projects/adl/*.py ~/projects/adl/*.sh peter900218@100.110.140.73:~/yoyo/adl/
scp peter900218@100.110.140.73:/tmp1/peter900218/adl/submission_full.csv ~/projects/adl/
scp ~/projects/adl/inference.py peter900218@100.110.140.73:/tmp1/peter900218/adl/

scp peter900218@100.110.140.73:~/yoyo/adl/paragraph_selection_model_best \
       yoyo/adl/answer_extraction_model_best \
       ~/projects/adl/

scp -r peter900218@100.110.140.73:/tmp1/peter900218/adl ~/projects/adl/download
tensorboard --logdir ./paragraph_selection_model_report --port 6006 --host 0.0.0.0
http://100.110.140.73:6006

#      參數設定
##     報告參數
```
CUDA_VISIBLE_DEVICES=2 python run_swag_no_trainer.py \
    --train_file train.json \
    --validation_file valid.json \
    --context_file context.json \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --output_dir ./paragraph_selection_model_report \
    --max_seq_length 512 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 6 \
    --per_device_eval_batch_size 6 \
    --learning_rate 1.5e-5 \
    --num_train_epochs 2 \
    --num_warmup_steps 300 \
    --weight_decay 0.02 \
    --lr_scheduler_type cosine_with_restarts \
    --with_tracking \
    --report_to tensorboard \
    --checkpointing_steps epoch \
    --seed 42
```
```
CUDA_VISIBLE_DEVICES=1 python qa.py \
    --train_file train.json \
    --validation_file valid.json \
    --context_file context.json \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --output_dir ./qa_model_report \
    --max_seq_length 512 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 3 \
    --per_device_eval_batch_size 20 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --num_warmup_steps 200 \
    --weight_decay 0.01 \
    --max_answer_length 80 \
    --doc_stride 200 \
    --lr_scheduler_type linear \
    --with_tracking \
    --report_to tensorboard \
    --checkpointing_steps 50 \
    --seed 42 
```
##     bert參數
```
CUDA_VISIBLE_DEVICES=2 python run_swag_no_trainer.py \
       --train_file train.json \
       --validation_file valid.json \
       --context_file context.json \
       --model_name_or_path ckiplab/bert-base-chinese \
       --output_dir ./paragraph_selection_model_bert \
       --max_seq_length 512 \
       --per_device_train_batch_size 6 \
       --gradient_accumulation_steps 3 \
       --per_device_eval_batch_size 12 \
       --learning_rate 1.5e-5 \
       --num_train_epochs 2 \
       --num_warmup_steps 300 \
       --weight_decay 0.02 \
       --lr_scheduler_type cosine_with_restarts \
       --with_tracking \
       --report_to tensorboard \
       --checkpointing_steps epoch \
       --seed 42
```
```
 CUDA_VISIBLE_DEVICES=1 python qa.py \     
       --train_file train.json \     
       --validation_file valid.json \     
       --context_file context.json \     
       --model_name_or_path ckiplab/bert-base-chinese \     
       --output_dir ./qa_model_bert \     
       --max_seq_length 512 \     
       --per_device_train_batch_size 10 \     
       --gradient_accumulation_steps 3 \     
       --per_device_eval_batch_size 20 \     
       --learning_rate 3e-5 \
       --num_train_epochs 2 \
       --num_warmup_steps 200 \
       --weight_decay 0.01 \
       --max_answer_length 80 \
       --doc_stride 200 \
       --lr_scheduler_type linear \
       --with_tracking \
       --report_to tensorboard \     
       --checkpointing_steps 50 \  
       --seed 42
```
##     不用pretrain
```
CUDA_VISIBLE_DEVICES=2 python run_swag_no_trainer.py \
       --train_file train.json \
       --validation_file valid.json \
       --context_file context.json \
       --model_type bert \
       --output_dir ./paragraph_selection_model_bert \
       --max_seq_length 512 \
       --per_device_train_batch_size 6 \
       --gradient_accumulation_steps 3 \
       --per_device_eval_batch_size 12 \
       --learning_rate 1.5e-5 \
       --num_train_epochs 2 \
       --num_warmup_steps 300 \
       --weight_decay 0.02 \
       --lr_scheduler_type cosine_with_restarts \
       --with_tracking \
       --report_to tensorboard \
       --checkpointing_steps epoch \
       --seed 42
```
```
 CUDA_VISIBLE_DEVICES=1 python qa.py \
       --train_file train.json \
       --validation_file valid.json \     
       --context_file context.json \
       --model_type bert \
       --output_dir ./qa_model_bert \
       --max_seq_length 512 \
       --per_device_train_batch_size 10 \
       --gradient_accumulation_steps 3 \
       --per_device_eval_batch_size 20 \
       --learning_rate 3e-5 \
       --num_train_epochs 2 \
       --num_warmup_steps 200 \
       --weight_decay 0.01 \
       --max_answer_length 80 \
       --doc_stride 200 \
       --lr_scheduler_type linear \
       --with_tracking \
       --report_to tensorboard \
       --checkpointing_steps 50 \
       --seed 42
```
##      實際參數
```
CUDA_VISIBLE_DEVICES=2 python run_swag_no_trainer.py \
    --train_file full_train.json \
    --validation_file full_train.json \
    --context_file context.json \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --output_dir ./paragraph_selection_model_full \
    --max_seq_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 9 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1.5e-5 \
    --num_train_epochs 2 \
    --num_warmup_steps 300 \
    --weight_decay 0.02 \
    --lr_scheduler_type cosine_with_restarts \
    --with_tracking \
    --report_to tensorboard \
    --checkpointing_steps epoch \
    --seed 42
```
```
CUDA_VISIBLE_DEVICES=1 python qa.py \
    --train_file full_train.json \
    --validation_file full_train.json \
    --context_file context.json \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --output_dir ./qa_model_full \
    --max_seq_length 512 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 3 \
    --per_device_eval_batch_size 20 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --num_warmup_steps 200 \
    --weight_decay 0.01 \
    --max_answer_length 80 \
    --doc_stride 200 \
    --lr_scheduler_type linear \
    --with_tracking \
    --report_to tensorboard \
    --checkpointing_steps 50 \
    --seed 42 
```
##     從頭訓練
```
CUDA_VISIBLE_DEVICES=3 python qa.py \
    --train_file train.json \
    --validation_file valid.json \
    --context_file context.json \
    --model_type bert \
    --tokenizer_name ckiplab/bert-base-chinese \
    --output_dir ./qa_model_nopretrain \
    --max_seq_length 512 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 3 \
    --per_device_eval_batch_size 20 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --num_warmup_steps 500 \
    --weight_decay 0.01 \
    --max_answer_length 80 \
    --doc_stride 200 \
    --lr_scheduler_type linear \
    --with_tracking \
    --report_to tensorboard \
    --checkpointing_steps 50 \
    --seed 42
```