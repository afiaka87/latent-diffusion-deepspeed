import numpy as np
import argparse
import random
import time

import torch
import wandb
from dalle_pytorch import distributed_utils
from crowsonkb.adamw_ema import AdamWEMA

from latent_diffusion_deepspeed.deepspeed_config import deepspeed_config_from_args
from latent_diffusion_deepspeed.image_text_datasets import create_dataloader
from latent_diffusion_deepspeed.model_util import (load_ldm_bert,
                                                   load_ldm_encoder,
                                                   load_model_and_diffusion,
                                                   sample_diffusion)
from latent_diffusion_deepspeed.train_util import save_model
import sys
sys.path.append("custom_clip")
from clip_custom import clip


def ldm_encode_data_gn(dataloader, encoder, bert, clip_model, device, use_fp16):
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for text, batch in dataloader:
            model_kwargs = {}
            text_emb = bert.encode(list(text)).to(device)
            text_blank = bert.encode(['']*batch.shape[0]).to(device)
            for i in range(batch.shape[0]):
                if random.randint(0, 100) < 20:
                    text_emb[i] = text_blank[i]


            clip_text = clip.tokenize(text, truncate=True).to(device)
            clip_emb = clip_model.encode_text(clip_text)

            model_kwargs["context"] = text_emb
            model_kwargs["clip_embed"] = clip_emb

            batch = batch.to(device)
            emb = encoder.encode(batch).sample()
            emb *= 0.18215
            yield emb, model_kwargs

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def train_step(model, diffusion, x_start, device, model_kwargs={}):
    model_kwargs["context"].to(device)
    model_kwargs["clip_embed"].to(device)
    timesteps = torch.randint(
        0, len(diffusion.betas) - 1, (x_start.shape[0],), device=device)
    scaled_timesteps = diffusion._scale_timesteps(timesteps).to(device)
    noise = torch.randn_like(x_start, device=device)
    x_t = diffusion.q_sample(x_start, timesteps, noise=noise).to(device)
    epsilon = model(x_t.to(device), scaled_timesteps.to(
        device), **model_kwargs).to(device)
    return torch.nn.functional.mse_loss(epsilon, noise.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--kl_model", type=str, default="kl-f8.pt")
    parser.add_argument("--bert_path", type=str, default="bert.ckpt")
    parser.add_argument("--resume_ckpt", type=str, default="diffusion.pt")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--ga_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_webdataset", action="store_true")
    parser.add_argument("--deepspeed", "-deepspeed", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--local_rank", "-local_rank",
                        type=int, default=0)  # stub for distributed
    parser.add_argument("--wandb_project", type=str,
                        default="latent-diffusion-deepspeed")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    data_dir = args.data_dir
    if args.data_dir.startswith("s3://"):
        data_dir = f"pipe:aws s3 cp {args.data_dir} -"
        args.use_webdataset = True  # force webdataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup deepspeed distributed training
    print(f"Initializing distributed training with local rank {args.local_rank}")
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()
    distr_backend.check_batch_size(args.batch_size)
    is_root_rank = distr_backend.is_local_root_worker()
    wandb_run = None
    if is_root_rank:
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               sync_tensorboard="finetune-ldm-logs/finetune-ldm", tensorboard=True)
    
    # from clip_custom import clip # make clip end up on the right device

    print(f"Loading CLIP.")
    clip_model, _ = clip.load('ViT-L/14', device=device, jit=False)
    clip_model.eval().requires_grad_(False)

    del clip_model.visual

    # Load backbone models
    # requires a device bc bug in latent-diffusion
    print(f"Loading bert.")
    bert = load_ldm_bert(device=device, bert_path=args.bert_path,
                         requires_grad=False)
    print(f"Loading LDM first stage encoder with local rank {args.local_rank}")
    encoder = load_ldm_encoder(args.kl_model, requires_grad=False)
    if args.use_fp16:  # Careful, this needs to be done _before_ loading the dataset
        bert = bert.half()
        encoder = encoder.half()
    else:
        clip_model.float() # CLIP is half precision, so we need to convert the encoder to FP32
    bert.to(device)
    encoder.to(device)

    # Load data
    print(f"Loading data with local rank {args.local_rank}")
    dataloader = create_dataloader(
        distr_backend,
        data_dir,
        args.batch_size,
        args.image_size,
        dataset_length=args.max_steps,  # TODO
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        use_webdataset=args.use_webdataset,
        num_workers=args.num_workers,
    )

    # Load the diffusion model (will be converted to fp16 if necessary)
    print(f"Loading diffusion model with local rank {args.local_rank}")
    model, diffusion = load_model_and_diffusion(
        model_path=args.resume_ckpt, use_fp16=args.use_fp16)
    model.to(device)

    print(f"Loading optimizer with local rank {args.local_rank}")
    optimizer = AdamWEMA(model.parameters(
    ), lr=args.lr, weight_decay=args.weight_decay, ema_decay=args.ema_decay, ema_power=1.)

    # Prepare pytorch vs. deepspeed optimizer, dataloader, model
    # if not args.deepspeed:
    #     model.train()  # make sure model is in train mode, only for non-deepspeed
    # else:
    #     model, optimizer, distr_data, _ = distributed_setup(
    #         model, optimizer, data, distr_backend, args, use_webdataset=args.use_webdataset)
    # if not args.use_webdataset:
    print(f"Distributing model with local rank {args.local_rank}")
    # model, optimizer, distr_data, _ = distributed_setup(model, optimizer, data, distr_backend, args, use_webdataset=args.use_webdataset)
    # def distributed_setup(model, optimizer, data, distr_backend, args, use_webdataset):
    # training_data = None
    # if not use_webdataset: # esoteric bug in deepspeed
    # TODO
    # training_data = data
        
    deepspeed_config = deepspeed_config_from_args(args)
    (model, optimizer, distributed_dataloader, _) = distr_backend.distribute(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=[x for x in model.parameters() if x.requires_grad],
        training_data=dataloader,
        lr_scheduler=None, # TODO: allow for pytorch scheduler
        config_params=deepspeed_config,
    )
    # return distributed_model, distributed_optimizer, distributed_dataloader, distributed_scheduler
    # data = distr_data  # returns None if using torch Dataset, deepspeed thing
    print(f"Starting training with local rank {args.local_rank}")
    # Train loop
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch} with local rank {args.local_rank}")
        data = ldm_encode_data_gn(distributed_dataloader, encoder, bert, clip_model, device, args.use_fp16)

        for i, (x_start, model_kwargs) in enumerate(data):
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                loss = train_step(model, diffusion, x_start,
                                  device=device, model_kwargs=model_kwargs)
            if args.deepspeed:
                model.backward(loss)
                model.step()
                accumulated_loss = distr_backend.average_all(loss)
            else:
                loss.backward()
                optimizer.step()
                model.zero_grad()
                accumulated_loss = loss
            if i % args.log_interval == 0 and is_root_rank:
                print(f"epoch {epoch} step {i} loss {accumulated_loss.item()}")

            if i % args.sample_interval == 0 and is_root_rank:
                current_generations = sample_diffusion(text="", bert=bert, ldm=encoder, model=model, clip_model=clip_model, custom_clip=clip, batch_size=4, prefix="finetune-samples", device=device,
                                                       timestep_respacing="30", ddpm=False, guidance_scale=1.0, shape=(256, 256), save_last=True, wandb_run=wandb_run, images_per_row=4)
                if wandb_run is not None:
                    wandb_run.log({
                        "current_generation": wandb.Image(current_generations),
                    })
            if i % args.save_interval == 0: #TODO
                save_model(model=model, path=args.log_dir, is_root=is_root_rank,
                            epoch=epoch, using_deepspeed=args.deepspeed, opt=optimizer)
                print(f"saved model to {args.log_dir} at epoch {epoch} step {i}")

    save_model(model=model, path=args.log_dir, is_root=is_root_rank,
               epoch=epoch, using_deepspeed=args.deepspeed, opt=optimizer)
    print(f"Finished training. saved model to {args.log_dir} at epoch {epoch} step {i}")


if __name__ == "__main__":
    main()
