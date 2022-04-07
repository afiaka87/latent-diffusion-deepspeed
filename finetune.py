import os
from re import A
from dalle_pytorch import distributed_utils
import argparse
from time import time
from deepspeed_config import deepspeed_config_from_args, distributed_setup
from guided_diffusion.image_text_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop
import torch
import random

from encoders.modules import BERTEmbedder

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def load_encoder(model, requires_grad=False):
    encoder = torch.load(model, map_location="cpu")
    encoder.eval()
    set_requires_grad(encoder, requires_grad)
    del encoder.decoder
    del encoder.loss
    return encoder


def load_bert(device, bert_path='bert.ckpt', requires_grad=False):
    assert os.path.exists(bert_path), "bert path does not exist"
    bert = BERTEmbedder(1280, 32, device=device)
    sd = torch.load(bert_path, map_location="cpu")
    bert.load_state_dict(sd)
    bert.eval()
    set_requires_grad(bert, requires_grad)
    return bert


def load_model_and_diffusion(model_path):
    options = model_and_diffusion_defaults()
    options.update({
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "image_size": 32,
        "learn_sigma": False,
        "noise_schedule": "linear",
        "num_channels": 320,
        "num_heads": 8,
        "num_res_blocks": 2,
        "resblock_updown": False,
        "use_fp16": False,
        "use_scale_shift_norm": False,
    })
    model, diffusion = create_model_and_diffusion(**options)

    if len(model_path) > 0:
        assert os.path.exists(model_path), "model path does not exist"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model, diffusion

def load_latent_data(encoder, bert, data_dir, batch_size, image_size, device):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=False,
    )
    for batch, model_kwargs, text in data:
        batch = batch.to(device)
        text_emb = bert.encode(list(text)).to(device)
        text_blank = bert.encode(['']*batch.shape[0]).to(device)
        for i in range(batch.shape[0]):
             if random.randint(0,100) < 20:
                 text_emb[i] = text_blank[i]
        model_kwargs["context"] = text_emb
        emb = encoder.encode(batch).sample()
        emb *= 0.18215
        yield emb, model_kwargs

def train_step(model, diffusion, x_start, device="cuda", model_kwargs={}):
    model_kwargs["context"].to(device)
    timesteps = torch.randint(0, len(diffusion.betas) - 1, (x_start.shape[0],), device=device)
    scaled_timesteps = diffusion._scale_timesteps(timesteps).to(device)

    noise = torch.randn_like(x_start, device=device)
    x_t = diffusion.q_sample(x_start, timesteps, noise=noise).to(device)

    epsilon = model(x_t, scaled_timesteps, **model_kwargs).to(device)
    return torch.nn.functional.mse_loss(epsilon, noise.detach())

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/coco")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--kl_model", type=str, default="kl-f8.pt")
    parser.add_argument("--bert_path", type=str, default="bert.ckpt")
    parser.add_argument("--resume_ckpt", type=str, default="diffusion.pt")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deepspeed", "-deepspeed", action="store_true")
    parser.add_argument("--local_rank", "-local_rank", type=int, default=0)

    args = parser.parse_args()
    device = torch.device(args.device)

    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()
    is_root_rank = distr_backend.is_local_root_worker()
    
    bert = load_bert(device="cuda", bert_path=args.bert_path, requires_grad=False)
    encoder = load_encoder(args.kl_model, requires_grad=False)
    data = load_latent_data(encoder, bert, args.data_dir, args.batch_size, args.image_size, device)
    model, diffusion = load_model_and_diffusion(args.resume_ckpt)
    bert.to(device)
    encoder.to(device)
    if not args.deepspeed:
        model.to(device)
        model.train()

    optimizer = None
    if not args.deepspeed:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.deepspeed:
        model, optimizer = distributed_setup(model, optimizer, data, distr_backend, args)

    for epoch in range(args.num_epochs):
        for i, (x_start, model_kwargs) in enumerate(data):
            x_start = x_start.to(device)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                loss = train_step(model, diffusion, x_start, device="cuda", model_kwargs=model_kwargs)
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
            if i % args.save_interval == 0 and is_root_rank:
                save_model(model, os.path.join(args.log_dir, f"diffusion-{epoch}-{i}.pt"))
                print(f"saved model to {args.log_dir}")

                

if __name__ == "__main__":
    main()
