import torch
import os

import sys
sys.path.append("custom_clip")
sys.path.append("glid-3-xl")
sys.path.append("latent-diffusion")

# from finetune import load_ldm_bert, load_ldm_encoder, load_model_and_diffusion
from latent_diffusion_deepspeed.model_util import load_ldm_bert, load_ldm_encoder, load_model_and_diffusion, sample_diffusion
from clip_custom import clip

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        use_fp16 = True
        bert_model = "bert.ckpt"
        ldm_model = "finetune-ldm-coco-art-faces-epoch-80.pt"
        kl_model = "kl-f8.pt"
        device = torch.device("cuda")

        clip_model, _ = clip.load('ViT-L/14', device=device, jit=False)
        clip_model.eval().requires_grad_(False)
        del clip_model.visual

        bert = load_ldm_bert(device=device, bert_path=bert_model, requires_grad=False)
        encoder = load_ldm_encoder(kl_model, requires_grad=False)

        bert = bert.half()
        encoder = encoder.half()

        bert.to(device)
        encoder.to(device)

        # Load the diffusion model (will be converted to fp16 if necessary)
        model, _ = load_model_and_diffusion(model_path=ldm_model, use_fp16=use_fp16)
        model.eval()
        model.to(device)
        model.convert_to_fp16()
        self.model, self.bert, self.encoder, self.clip_model = model, bert, encoder, clip_model


    def predict(
        self,
        prompt: str = Input(description="Prompt to generate", default=""),
        batch_size: int = Input(description="Batch size", default=8),
        width: int=Input(description="Width of the image", default=256),
        height: int=Input(description="Height of the image", default=256),
        timestep_respacing: str=Input(description="Timestep respacing", default="200"),
        ddpm: bool=Input(description="Use DDPM for deterministic sampling", default=False),
        guidance_scale: float=Input(description="Guidance scale", default=5.0, ge=0, le=50),
        seed: int=Input(description="Seed for random number generator", default=0),
    ) -> Path:
        images_per_row = batch_size // 2
        torch.manual_seed(seed)
        log_dir = os.path.join("outputs", prompt.strip().replace(" ", "_")).replace(",", "_")
        os.makedirs(log_dir, exist_ok=True)

        with torch.cuda.amp.autocast():
            output_path = sample_diffusion(text=prompt, bert=self.bert, ldm=self.encoder, model=self.model, clip_model=self.clip_model, custom_clip=clip, batch_size=batch_size,
                                        device="cuda", prefix=log_dir, timestep_respacing=timestep_respacing, ddpm=ddpm, guidance_scale=guidance_scale, shape=(width, height), wandb_run=None, images_per_row=images_per_row)
            return Path(output_path)

