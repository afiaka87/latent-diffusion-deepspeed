import argparse
import os

import torch
from tqdm import tqdm

import wandb
from finetune import load_ldm_bert, load_ldm_encoder, load_model_and_diffusion
from latent_diffusion_deepspeed.model_util import sample_diffusion
import sys
sys.path.append("custom_clip")
from clip_custom import clip

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="",
                        help="a string or .txt file containing multiple prompts")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--timestep_respacing", type=str, default="200")
    parser.add_argument("--ddpm", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--kl_model", type=str, default="kl-f8.pt")
    parser.add_argument("--bert_model", type=str, default="bert.ckpt")
    parser.add_argument("--ldm_model", type=str, default="diffusion.pt")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fp16", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str,
                        default="ldm-sampling")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument('--images_per_row', type=int, default=8)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    if ".txt" in args.prompt:
        prompts = open(args.prompt).readlines()
        print(f"{len(prompts)} prompts loaded")
    else:
        prompts = [args.prompt]
        print(f"Loaded 1 prompt - {args.prompt}")

    assert len(prompts) > 0, "No prompts loaded"
    assert args.images_per_row > 0, "images_per_row must be > 0"
    assert args.images_per_row % 2 == 0, "images_per_row must be even"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = None
    # Setup wandb, dont pass entity if None
    if args.wandb_entity is not None:
        wandb_run = wandb.init(project=args.wandb_project,
                               entity=args.wandb_entity)
    else:
        wandb_run = wandb.init(project=args.wandb_project)


    clip_model, _ = clip.load('ViT-L/14', device=device, jit=False)
    clip_model.eval().requires_grad_(False)

    del clip_model.visual
    bert = load_ldm_bert(
        device=device, bert_path=args.bert_model, requires_grad=False)
    encoder = load_ldm_encoder(args.kl_model, requires_grad=False)
    if args.use_fp16:
        bert = bert.half()
        encoder = encoder.half()
    bert.to(device)
    encoder.to(device)

    # Load the diffusion model (will be converted to fp16 if necessary)
    model, _ = load_model_and_diffusion(
        model_path=args.ldm_model, use_fp16=args.use_fp16)
    model.eval()
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()

    for prompt in tqdm(prompts):
        print(f"Sampling {prompt}")
        # Create a directory for the prompt
        log_dir = os.path.join(
            args.log_dir, prompt.strip().replace(" ", "_")).replace(",", "_")
        os.makedirs(log_dir, exist_ok=True)

        # Sample the diffusion
        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            # TODO we only pass wandb_run if we are training, then log with it manually ourselves to avoid custom logic for training vs inference
            output_path = sample_diffusion(text=prompt, bert=bert, ldm=encoder, model=model, clip_model=clip_model, custom_clip=clip, batch_size=args.batch_size,
                                           device=device, prefix=log_dir, timestep_respacing=args.timestep_respacing, ddpm=args.ddpm, guidance_scale=args.guidance_scale, shape=(args.width, args.height), wandb_run=None, images_per_row=args.images_per_row)
            wandb_run.log({"grid": wandb.Image(output_path, caption=prompt)})
            print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
