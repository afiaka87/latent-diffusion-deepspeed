import argparse
import os
import random
import shutil
from glob import glob
from pathlib import Path

import torch
import wandb
from dalle_pytorch import distributed_utils
from guided_diffusion.image_text_datasets import create_dataloader

from latent_diffusion_deepspeed.deepspeed_config import distributed_setup
from latent_diffusion_deepspeed.model_util import (load_ldm_bert,
                                                   load_ldm_encoder,
                                                   load_model_and_diffusion,
                                                   sample_diffusion)
from latent_diffusion_deepspeed.train_util import (ldm_encode_data_gn,
                                                   save_model)


@torch.no_grad()
def ldm_encode_data_gn(dataloader, encoder, bert, device, use_fp16):
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for text, batch in dataloader:
            text_emb = bert.encode(list(text)).to(device)
            text_blank = bert.encode(['']*batch.shape[0]).to(device)
            for i in range(batch.shape[0]):
                if random.randint(0, 100) < 20:
                    text_emb[i] = text_blank[i]

            # model_kwargs["context"] = text_emb
            model_kwargs = {"context": text_emb}
            batch = batch.to(device)
            emb = encoder.encode(batch).sample()
            emb *= 0.18215
            yield emb, model_kwargs


def train_step(model, diffusion, x_start, device, model_kwargs={}):
    model_kwargs["context"].to(device)
    timesteps = torch.randint(
        0, len(diffusion.betas) - 1, (x_start.shape[0],), device=device)
    scaled_timesteps = diffusion._scale_timesteps(timesteps).to(device)
    noise = torch.randn_like(x_start, device=device)
    x_t = diffusion.q_sample(x_start, timesteps, noise=noise).to(device)
    epsilon = model(x_t.to(device), scaled_timesteps.to(device),
                    **model_kwargs).to(device).requires_grad_(True)
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
    parser.add_argument("--test_prompt", type=str, default="")
    parser.add_argument("--wandb_project", type=str,
                        default="latent-diffusion-deepspeed")
    parser.add_argument("--wandb_entity", type=str, default="")

    args = parser.parse_args()

    data_dir = args.data_dir
    if args.data_dir.startswith("s3://"):
        data_dir = f"pipe:aws s3 cp {args.data_dir} -"
        args.use_webdataset = True  # force webdataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup deepspeed distributed training
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()
    distr_backend.check_batch_size(args.batch_size)
    is_root_rank = distr_backend.is_local_root_worker()
    wandb_run = None
    if is_root_rank:
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               sync_tensorboard="finetune-ldm-logs/finetune-ldm", tensorboard=True)

    # Load backbone models
    # requires a device bc bug in latent-diffusion
    bert = load_ldm_bert(device=device, bert_path=args.bert_path,
                         requires_grad=False)
    encoder = load_ldm_encoder(args.kl_model, requires_grad=False)
    if args.use_fp16:  # Careful, this needs to be done _before_ loading the dataset
        bert = bert.half()
        encoder = encoder.half()
    bert.to(device)
    encoder.to(device)

    # Load data
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
    data = ldm_encode_data_gn(dataloader, encoder, bert, device, args.use_fp16)

    # Load the diffusion model (will be converted to fp16 if necessary)
    model, diffusion = load_model_and_diffusion(
        model_path=args.resume_ckpt, use_fp16=args.use_fp16)
    model.to(device)

    # Prepare pytorch vs. deepspeed optimizer, dataloader, model
    optimizer = None
    if not args.deepspeed:
        model.train()  # make sure model is in train mode, only for non-deepspeed
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        model, optimizer, distr_data, _ = distributed_setup(
            model, optimizer, data, distr_backend, args, use_webdataset=args.use_webdataset)
        if not args.use_webdataset:
            data = distr_data # returns None if using torch Dataset, deepspeed thing

    # Train loop
    for epoch in range(args.num_epochs):
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
                current_generations = sample_diffusion(idx=i, text=args.test_prompt, bert=bert, ldm=encoder, model=model, batch_size=1,
                                                       device=device, timestep_respacing="100", ddpm=False, guidance_scale=4.0, shape=(256, 256))
                if wandb_run is not None:
                    wandb_run.log({
                        "current_generation": wandb.Image(current_generations, caption=args.test_prompt),
                    })
            if i % args.save_interval == 0:
                save_model(model=model, path=args.log_dir, is_root=is_root_rank,
                           epoch=epoch, using_deepspeed=args.deepspeed, opt=optimizer)

    save_model(model=model, path=args.log_dir, is_root=is_root_rank,
               epoch=epoch, using_deepspeed=args.deepspeed, opt=optimizer)
    print(f"saved model to {args.log_dir}")


if __name__ == "__main__":
    main()
