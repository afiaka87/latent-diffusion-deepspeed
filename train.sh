export TOKENIZERS_PARALLELISM=false # required to avoid errors with transformers lib
# data_dir="data/coco/my_data"
data_dir="s3://my-s3/path/to/webdataset/{00000..00161}.tar"
log_dir="my-logs"
deepspeed --include localhost:0,1,2,3,4,5,6,7 finetune.py \
    --data_dir $data_dir \
    --log_dir $log_dir \
    --image_size 256 \
    --batch_size 512 \
    --ga_steps 8 \
    --kl_model 'kl-f8.pt' \
    --bert_path 'bert.ckpt' \
    --resume_ckpt 'diffusion.pt' \
    --num_epochs 10 \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.01 \
    --seed 0 \
    --log_interval 10 \
    --save_interval 1000 \
    --sample_interval 100 \
    --num_workers 8 \
    --max_steps 1_000_000 \
    --warmup_steps 1000 \
    --use_webdataset \
    --deepspeed \
    --use_fp16 \
    --wandb_project "latent-diffusion-deepspeed" \
    --wandb_entity ""