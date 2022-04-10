import os
import torch
import encoders # latent diffusion auto-encoder
from torchvision.transforms.functional import to_pil_image
from guided_diffusion.script_util import (create_gaussian_diffusion, create_model_and_diffusion,
                                          model_and_diffusion_defaults)
def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def load_ldm_encoder(model, requires_grad=False):
    encoder = torch.load(model, map_location="cpu")
    encoder.eval()
    set_requires_grad(encoder, requires_grad)
    del encoder.loss
    return encoder


def load_ldm_bert(device, bert_path='bert.ckpt', requires_grad=False):
    assert os.path.exists(bert_path), "bert path does not exist"
    bert = encoders.modules.BERTEmbedder(1280, 32, device=device)
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
        assert os.path.exists(model_path), "model path does not exist"
        model.load_state_dict(torch.load(
            model_path, map_location="cpu"), strict=False)
    if use_fp16:
        model.convert_to_fp16()
    model.requires_grad_(True)
    return model, diffusion


@torch.inference_mode()
def sample_diffusion(idx, text, bert, ldm, model, batch_size, device, prefix="output", timestep_respacing="50", ddpm=False, guidance_scale=10.0, shape=(256, 256)):
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

    if timestep_respacing.startswith('ddim'):
        sample_fn = sampling_diffusion.ddim_sample_loop_progressive
    elif ddpm:
        sample_fn = sampling_diffusion.p_sample_loop_progressive
    else:
        sample_fn = sampling_diffusion.plms_sample_loop_progressive

    samples = sample_fn(
        model_fn,
        (batch_size*2, 4, int(height/8), int(width/8)),
        clip_denoised=False,
        model_kwargs=kwargs,
        cond_fn=None,
        device=device,
        progress=False,
    )
    for sample in samples:
        for k, image in enumerate(sample['pred_xstart'][:batch_size]):
            image /= 0.18215
            im = image.unsqueeze(0)
            out = ldm.decode(im)
            out = to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))
            os.makedirs(os.path.dirname( f"{prefix}/{idx}/{k}.png"), exist_ok=True)
            out.save(f"{prefix}/{idx}/{k}.png")

    out.save("current_sample.png")
    return out
