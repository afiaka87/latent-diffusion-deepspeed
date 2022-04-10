import argparse
import wandb
import torch
from dalle_pytorch import distributed_utils
from finetune import load_bert, load_encoder, load_model_and_diffusion, load_latent_data
from latent_diffusion_deepspeed.model_util import sample_diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--timestep_respacing", type=str, default="100")
    parser.add_argument("--ddpm", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--kl_model", type=str, default="kl-f8.pt")
    parser.add_argument("--bert_model", type=str, default="bert.ckpt")
    parser.add_argument("--ldm_model", type=str, default="diffusion.pt")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="latent-diffusion-deepspeed")
    parser.add_argument("--wandb_entity", type=str, default="")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup wandb
    wandb_run = wandb.init(project=args.wandb_project,
                           entity=args.wandb_entity)

    # Load backbone models
    # requires a device bc bug in latent-diffusion
    bert = load_bert(device=device, bert_path=args.bert_path,
                     requires_grad=False)
    encoder = load_encoder(args.kl_model, requires_grad=False)
    if args.use_fp16:  # Careful, this needs to be done _before_ loading the dataset
        bert = bert.half()
        encoder = encoder.half()

    bert.to(device)
    encoder.to(device)

    # Load the diffusion model (will be converted to fp16 if necessary)
    model, diffusion = load_model_and_diffusion(
        model_path=args.resume_ckpt, use_fp16=args.use_fp16)
    model.to(device)

    samples = sample_diffusion(idx=0, text=args.prompt, bert=bert, ldm=encoder, model=model, batch_size=args.batch_size,
                               device=device, prefix=args.log_dir, timestep_respacing=args.timestep_respacing, ddpm=args.ddpm, guidance_scale=args.guidance_scale, shape=(args.width, args.height))


if __name__ == "__main__":
    main()
