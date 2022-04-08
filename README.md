# latent-diffusion-deepspeed (broken - WIP)
Finetune the 1.4B latent diffusion text2img-large checkpoint from CompVis using deepspeed. (work-in-progress)

## Download weights
```sh
wget --continue https://dall-3.com/models/glid-3-xl/bert.pt -o bert.ckpt
wget --continue https://dall-3.com/models/glid-3-xl/kl-f8.pt
wget --continue https://dall-3.com/models/glid-3-xl/diffusion.pt
```

## Installation

Grab the repo
```sh
git clone https://github.com/afiaka87/latent-diffusion-deepspeed
cd latent-diffusion-deepseed
```

Go to [pytorch.org](https://pytorch.org/get-started/locally/) and install the correct version of pytorch for your system. 
You can check which CUDA version your system supports with `nvidia-smi`. Testing was done with:
```sh
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

Install latent-diffusion
```sh
git clone https://github.com/CompVis/latent-diffusion.git
cd latent-diffusion
pip install -e . & cd ..
```

Install glid-3-xl
```sh
git clone https://github.com/Jack000/glid-3-xl.git
cd glid-3xl
pip install -e .
cd ..
```

Install this repo
```sh
pip install -e . 
```

## Finetune

Modify `train.sh` then run:
`source train.sh`