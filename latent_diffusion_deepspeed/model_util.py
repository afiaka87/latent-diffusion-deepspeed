import wandb
import numpy as np
import torchvision
import itertools
import os
import torch
from torchvision.transforms.functional import to_pil_image
from encoders.modules import BERTEmbedder

from guided_diffusion.script_util import create_gaussian_diffusion, create_model_and_diffusion, model_and_diffusion_defaults


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def load_ldm_encoder(model, requires_grad=False):
    assert os.path.exists(model), "model not found, download from https://dall-3.com/models/glid-3-xl/kl-f8.pt"
    encoder = torch.load(model, map_location="cpu")
    encoder.eval()
    set_requires_grad(encoder, requires_grad)
    del encoder.loss
    return encoder


def load_ldm_bert(device, bert_path='bert.ckpt', requires_grad=False):
    assert os.path.exists(bert_path), "bert not found, download from https://dall-3.com/models/glid-3-xl/bert.pt"
    bert = BERTEmbedder(1280, 32, device=device)
    sd = torch.load(bert_path, map_location="cpu")
    bert.load_state_dict(sd)
    bert.eval()
    set_requires_grad(bert, requires_grad)
    return bert

def diffusion_options(use_fp16, timestep_respacing):
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
        "timestep_respacing": timestep_respacing,
        "use_fp16": use_fp16,
        "use_scale_shift_norm": False,
    })
    return options


def load_model_and_diffusion(model_path, use_fp16=True):
    options = diffusion_options(use_fp16, timestep_respacing=str(1000))
    model, diffusion = create_model_and_diffusion(**options)
    if len(model_path) > 0:
        assert os.path.exists(model_path), "model not found, download from https://dall-3.com/models/glid-3-xl/diffusion.pt"
        model.load_state_dict(torch.load(
            model_path, map_location="cpu"), strict=False)
    if use_fp16:
        model.convert_to_fp16()
    model.requires_grad_(True)
    return model, diffusion


@torch.inference_mode()
def sample_diffusion(text, bert, ldm, model, batch_size, device, prefix="output", timestep_respacing="50", ddpm=False, guidance_scale=10.0, shape=(256, 256), save_last=True, wandb_run=None, images_per_row=8):
    sampling_options = diffusion_options(False, timestep_respacing)
    sampling_diffusion = create_gaussian_diffusion(
        steps=sampling_options["diffusion_steps"],
        learn_sigma=sampling_options["learn_sigma"],
        noise_schedule=sampling_options["noise_schedule"],
        use_kl=sampling_options["use_kl"],
        predict_xstart=sampling_options["predict_xstart"],
        rescale_timesteps=sampling_options["rescale_timesteps"],
        rescale_learned_sigmas=sampling_options["rescale_learned_sigmas"],
        timestep_respacing=timestep_respacing,
    )
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    height, width = shape
    text_emb = bert.encode([text]*batch_size).to(device)
    text_blank = bert.encode(['']*batch_size).to(device)
    kwargs = {"context": torch.cat([text_emb, text_blank], dim=0)}
    # Create a classifier-free guidance sampling function

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    final_step = int(timestep_respacing.replace("ddim", "")) - 1
    if timestep_respacing.startswith('ddim'):
        sample_fn = sampling_diffusion.ddim_sample_loop_progressive
    elif ddpm:
        sample_fn = sampling_diffusion.p_sample_loop_progressive
    else:
        # PLMS sampling skips the first steps
        final_step = int(timestep_respacing.replace("ddim", "")) - 3
        sample_fn = sampling_diffusion.plms_sample_loop_progressive

    samples = sample_fn(
        model_fn,
        (batch_size*2, 4, int(height/8), int(width/8)),
        clip_denoised=False,
        model_kwargs=kwargs,
        cond_fn=None,
        device=device,
        progress=True,
    )
    def decode_sample(image):
        image /= 0.18215
        im = image.unsqueeze(0)
        decoded_image = ldm.decode(im)
        return decoded_image.squeeze(0).add(1).div(2).clamp(0, 1)


    output_path = os.path.join(prefix, f"grid.png")
    for timestep_idx, sample in enumerate(samples):
        batch_gn = sample['pred_xstart'][:batch_size]
        batch = [decode_sample(out) for out in batch_gn]
        grid = torchvision.utils.make_grid(batch, nrow=images_per_row)
        if save_last and timestep_idx == final_step:
            torchvision.utils.save_image(grid, output_path)
            if wandb_run is not None: wandb_run.log({"sample": wandb.Image(output_path, caption=text)})
        else:
            torchvision.utils.save_image(grid, output_path)
    return output_path