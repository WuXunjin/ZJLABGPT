wandb disabled
CUDA_VISIBLE_DEVICES=0,1,2,3 

nohup deepspeed --num_gpus=4 fastchat/train/train_lora.py --deepspeed ./deepspeed-config.json --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --model_name_or_path ./pyllama_data/output/7B --data_path ./data/sharegpt_clean_split.json --fp16 True --output_dir ./output --num_train_epochs 1 --per_device_train_batch_size 14 --per_device_eval_batch_size 14 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2400 --save_total_limit 5 --learning_rate 2e-5 --weight_decay 0. --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 512 --gradient_checkpointing True >> lora.log  2>&1 &

