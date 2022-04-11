# latent-diffusion-deepspeed 

Finetune the 1.4B latent diffusion text2img-large checkpoint from CompVis using deepspeed.

## Download weights
```sh
wget --continue https://dall-3.com/models/glid-3-xl/bert.pt && mv bert.pt bert.ckpt
wget --continue https://dall-3.com/models/glid-3-xl/kl-f8.pt
wget --continue https://dall-3.com/models/glid-3-xl/diffusion.pt
```

## Installation

Grab the repo
```sh
git clone https://github.com/afiaka87/latent-diffusion-deepspeed
cd latent-diffusion-deepseed && pip install -e .
```

Go to [pytorch.org](https://pytorch.org/get-started/locally/) and install the correct version of pytorch for your system. 
You can check which CUDA version your system supports with `nvidia-smi`. Testing was done with:
```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Install deepspeed
```sh
pip install deepspeed
```

Install latent-diffusion
```sh
git clone https://github.com/CompVis/latent-diffusion.git
cd latent-diffusion
pip install -e . && cd ..
```

Install glid-3-xl
```sh
git clone https://github.com/Jack000/glid-3-xl.git
cd glid-3xl
pip install -e . && cd ..
```

## Finetune

Modify `train.sh` then run:
`source train.sh`

```sh
export TOKENIZERS_PARALLELISM=false # required to avoid errors with transformers lib
# data_dir="data/coco/my_data" # you can use coco-style e.g. data/00001.png -> data/00001.txt pairs.
data_dir="s3://my-s3/path/to/webdataset/{00000..00161}.tar" # or webdataset. `--max_steps` and `--use_webdataset` required for wds
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
``` 


### Converting Deepspeed checkpoints back to pytorch

```sh
# assuming a deepspeed checkpoint directory of `latent-diffusion-ds-cp`
cd latent-diffusion-ds-cp
python zero_to_fp32.py . my_checkpoint.pt
mv my_checkpoint.pt ..
```

### Sample from your trained checkpoint

The `--prompt` arg can be either a caption as a string, or a file ending with `.txt` that contains a line separated list of captions.

```sh
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --prompt="coco-captions.txt" \
    --batch_size=16 \
    --width=256 \
    --height=256 \
    --timestep_respacing=200 \
    --guidance_scale=5.0 \
    --kl_model="kl-f8.pt" \
    --bert_model="bert.ckpt" \
    --ldm_model="my_checkpoint.pt" \
    --log_dir="output_dir/" \
    --seed=0 \
    --use_fp16=True \
    --wandb_project="ldm-sampling"
```

