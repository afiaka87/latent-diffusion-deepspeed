export TOKENIZERS_PARALLELISM=false # required to avoid errors with transformers lib
data_dir="data/coco/my_data"
# data_dir="s3://my-s3/path/to/webdataset/{00000..00161}.tar"
log_dir="my-logs"
deepspeed --include localhost:0,1,2,3,4,5,6,7 latent_diffusion_deepspeed/finetune.py \
    --data_dir $data_dir \
    --log_dir $log_dir \
    --image_size 256 \
    --batch_size 64 \
    --ga_steps 1 \
    --kl_model 'kl-f8.pt' \
    --bert_path 'bert.ckpt' \
    --resume_ckpt 'diffusion.pt' \
    --num_epochs 100 \
    --lr 2e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.01 \
    --seed 0 \
    --log_interval 10 \
    --save_interval 1000 \
    --num_workers 0 \
    --max_steps 100000 \
    --random_flip \
    --random_crop \
    --use_webdataset \
    --deepspeed