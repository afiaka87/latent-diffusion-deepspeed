
# `--prompts` can be a `.txt` file containing a line separated list of prompts or a string
CUDA_VISIBLE_DEVICES=1 python sample.py \
    --prompt="/home/afiaka87/latent-diffusion-deepspeed/eleu-shuf.txt" \
    --batch_size=4 \
    --images_per_row 8 \
    --width=256 \
    --height=256 \
    --timestep_respacing=50 \
    --guidance_scale=8.0 \
    --kl_model="kl-f8.pt" \
    --bert_model="bert.ckpt" \
    --ldm_model="/opt/afiaka87/FINETUNE_GLIDE_XL-logs-ds-cp/finetune-ldm-coco-art-faces-epoch-80.pt" \
    --log_dir="/opt/afiaka87/FINETUNE_GLIDE_SL-samples" \
    --seed=0 \
    --use_fp16=True \
    --wandb_project="glid-3-xl-digital" \
    --wandb_entity="dalle-pytorch-replicate"