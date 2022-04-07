# latent-diffusion-deepspeed (broken - WIP)
Finetune the 1.4B latent diffusion text2img-large checkpoint from CompVis using deepspeed. (work-in-progress)


## Installation

Go to [pytorch.org](https://pytorch.org/get-started/locally/) and install the correct version of pytorch for your system. 
You can check which CUDA version your system supports with `nvidia-smi`. Testing was done with:
```sh
# install pytorch==1.11.1+cu113
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Install apex automatic mixed precision.
```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Install deepspeed
```sh
pip install deepspeed
```

Install this repo
```sh
git clone https://github.com/afiaka87/latent-diffusion-deepspeed
cd latent-diffusion-deepspeed
pip install -e .
```

## Finetune
```sh
export TOKENIZERS_PARALLELISM=false # required to avoid errors with transformers lib
deepspeed simple_train.py \
    --data_dir data/coco \
    --batch_size 32 \
    --image_size 256 \
    --kl_model kl-f8.pt \
    --bert_path bert.ckpt \
    --resume_ckpt diffusion.pt \
    --log_dir log/coco \
    --num_epochs 100 \
    --lr 1e-8 \
    --weight_decay 0.0 \
    --seed 0 \
    --log_interval 10 \
    --num_workers 0 \
    --device cuda \
    --deepspeed
```
