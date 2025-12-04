#!/usr/bin/env python
import os
import argparse

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from torchvision.utils import save_image

# this is your patched step
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/root/autodl-tmp/ddpo-pytorch/cache_sd15"
CKPT_DIR = "/root/autodl-tmp/ddpo-pytorch/logs/run_7_2025.12.01_13.01.44/checkpoints/checkpoint_4"
OUT_DIR = "ddim_out_steps/ddpo_IS_compress_pmpt_bird"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "a bird"
NEG_PROMPT = ""
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
HEIGHT = 512
WIDTH = 512
ETA = 0.0  # standard DDIM
BASE_DIR = "/root/autodl-tmp/ddpo-pytorch/cache_sd15/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"

def decode_latents(vae, latents):
    """Latents -> pixel images in [0, 1]."""
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    latents = latents / scaling_factor
    image = vae.decode(latents).sample  # [-1, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


@torch.no_grad()
def sample_trajectory(
    pretrained_model,
    revision,
    ckpt_dir,
    prompt,
    negative_prompt,
    num_steps,
    guidance_scale,
    eta,
    use_lora,
    outdir,
    height,
    width,
    device,
):
    os.makedirs(outdir, exist_ok=True)

    # 1. Base pipeline (same as in train.py)
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     pretrained_model,
    #     revision=revision,
    #     cache_dir="/root/autodl-tmp/ddpo-pytorch/cache_sd15"
    # )
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_DIR,
        local_files_only=True,     # <--- prevent downloads
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    pipe.to(device)

    # 2. Load finetuned weights from checkpoint_* (mirror load_model_hook)
    if use_lora:
        # training save hook did: pipeline.unet.save_attn_procs(output_dir)
        pipe.unet.load_attn_procs(ckpt_dir)
        print(f"Loaded LoRA attn processors from {ckpt_dir}")
    else:
        # training save hook did: models[0].save_pretrained(os.path.join(output_dir, 'unet'))
        ft_unet = UNet2DConditionModel.from_pretrained(ckpt_dir, subfolder="unet")
        pipe.unet.load_state_dict(ft_unet.state_dict())
        del ft_unet
        print(f"Loaded full UNet weights from {ckpt_dir}/unet")

    pipe.unet.to(device)

    # 3. Text embeddings (same style as trainer)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    text_ids = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    prompt_embeds = text_encoder(text_ids)[0]  # (1, seq, dim)

    neg_ids = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    negative_prompt_embeds = text_encoder(neg_ids)[0]

    # # classifier-free guidance: concat [neg, pos]
    # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # # 4. Prepare DDIM schedule + initial latents
    # scheduler = pipe.scheduler
    # scheduler.set_timesteps(num_steps, device=device)
    # timesteps = scheduler.timesteps  # e.g. [999, 979, ...]

    # bsz = 1
    # latent_shape = (
    #     bsz,
    #     pipe.unet.in_channels,
    #     height // 8,
    #     width // 8,
    # )
    # latents = torch.randn(latent_shape, device=device, dtype=prompt_embeds.dtype)
    # latents = latents * scheduler.init_noise_sigma

    # # 5. DDIM loop with logging per timestep
    # for i, t in enumerate(timesteps):
    #     # duplicate latents for CFG
    #     latent_model_input = torch.cat([latents] * 2, dim=0)
    #     latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    #     # UNet forward → ε_θ(x_t, t)
    #     noise_pred = pipe.unet(
    #         latent_model_input,
    #         t,
    #         encoder_hidden_states=prompt_embeds,
    #     ).sample  # (2, C, H, W)

    #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #     noise_pred_cfg = noise_pred_uncond + guidance_scale * (
    #         noise_pred_text - noise_pred_uncond
    #     )

    #     # save predicted noise
    #     torch.save(
    #         noise_pred_cfg.detach().cpu(),
    #         os.path.join(outdir, f"eps_step_{i:03d}_t{int(t)}.pt"),
    #     )

    #     # DDIM update using your patched step (also returns log_prob if you care)
    #     latents, log_prob = ddim_step_with_logprob(
    #         scheduler,
    #         model_output=noise_pred_cfg,
    #         timestep=t,
    #         sample=latents,
    #         eta=eta,
    #         prev_sample=None,
    #     )

    #     # save latent
    #     torch.save(
    #         latents.detach().cpu(),
    #         os.path.join(outdir, f"latent_step_{i:03d}_t{int(t)}.pt"),
    #     )

    #     # decode to pixel and save image
    #     img = pipe.vae.decode(
    #         latents / pipe.vae.config.scaling_factor, return_dict=False
    #     )[0]

    #     do_denormalize = [True] * img.shape[0]
    #     img = pipe.image_processor.postprocess(
    #     img, output_type="pil", do_denormalize=do_denormalize
    # )
    #     # img = decode_latents(pipe.vae, latents)  # (1, 3, H, W), [0,1]
    #     # save_image(img, os.path.join(outdir, f"img_step_{i:03d}_t{int(t)}.png"))

    #     img[0].save(os.path.join(outdir, f"img_step_{i:03d}_t{int(t)}.png"))

    #     noise = pipe.vae.decode(
    #         noise_pred_cfg / pipe.vae.config.scaling_factor, return_dict=False
    #     )[0]

    #     do_denormalize = [True] * noise.shape[0]
    #     noise = pipe.image_processor.postprocess(
    #     noise, output_type="pil", do_denormalize=do_denormalize
    # )
    #     # img = decode_latents(pipe.vae, latents)  # (1, 3, H, W), [0,1]
    #     # save_image(img, os.path.join(outdir, f"img_step_{i:03d}_t{int(t)}.png"))

    #     noise[0].save(os.path.join(outdir, f"noise_step_{i:03d}_t{int(t)}.png"))

    #     # (optional) save log_prob too
    #     torch.save(
    #         log_prob.detach().cpu(),
    #         os.path.join(outdir, f"logprob_step_{i:03d}_t{int(t)}.pt"),
    #     )

    #     print(f"[step {i:03d}] t={int(t)} saved eps/latent/img/logprob")
    images, eps_list, latents_list, log_probs = pipeline_with_logprob(
        pipe,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=NUM_STEPS,        # ideally config.sample.num_steps
        guidance_scale=GUIDANCE_SCALE,        # match config.sample.guidance_scale
        eta=ETA,                              # match config.sample.eta
        output_type="pt",
    )
    for i, latents in enumerate(latents_list):



        # decode to pixel and save image
        img = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False
        )[0]

        do_denormalize = [True] * img.shape[0]
        img = pipe.image_processor.postprocess(
        img, output_type="pil", do_denormalize=do_denormalize
    )
        # img = decode_latents(pipe.vae, latents)  # (1, 3, H, W), [0,1]
        # save_image(img, os.path.join(outdir, f"img_step_{i:03d}_t{int(t)}.png"))

        img[0].save(os.path.join(outdir, f"img_step_t{int(i)}.png"))

        print(f"[step {i:03d}] saved eps/latent/img/logprob")

    print(f"Done. Trajectory saved to {outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Same as config.pretrained.model in training.")
    parser.add_argument("--revision", type=str, default="main",
                        help="Same as config.pretrained.revision.")
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR,
                        help="Path to checkpoint_* directory (the thing you passed to resume_from).")
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--use_lora", default=True,
                        help="Match config.use_lora from training.")
    parser.add_argument("--outdir", type=str, default=OUT_DIR)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_trajectory(
        pretrained_model=args.pretrained_model,
        revision=args.revision,
        ckpt_dir=args.ckpt_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        use_lora=args.use_lora,
        outdir=args.outdir,
        height=args.height,
        width=args.width,
        device=device,
    )


if __name__ == "__main__":
    main()
