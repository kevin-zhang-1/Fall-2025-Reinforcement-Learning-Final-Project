#!/usr/bin/env python
import os
import argparse

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel

# this is your patched step
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
import torch, random, numpy as np

SEED = 43

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/root/autodl-tmp/ddpo-pytorch/cache_sd15"
# CKPT_DIR = "/root/autodl-tmp/ddpo-pytorch/logs/run_5_aes_iskl_2025.12.07_01.18.11/checkpoints/checkpoint_0"
CKPT_DIR = "/root/autodl-tmp/ddpo-pytorch/logs/run_4_aes_sf_2025.12.06_05.04.33/checkpoints/checkpoint_0"
# CKPT_DIR = "/root/autodl-tmp/ddpo-pytorch/logs/run_5_aes_is_2025.12.05_12.41.08/checkpoints/checkpoint_0"

OUT_DIR = "ddim_out/ddpo_SF_aesthetics_empire_state/"
# OUT_DIR = "ddim_out/ddpo_ISKL_aesthetics_pmpt_empire_state/"

os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "empire_state"
NEG_PROMPT = ""
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
HEIGHT = 512
WIDTH = 512
ETA = 0.0  # standard DDIM
BASE_DIR = "/root/autodl-tmp/ddpo-pytorch/cache_sd15/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"

# list of ckpt numbers you want to test
CKPT_NUMS = [1, 3, 5, 7, 9]
# CKPT_NUMS = [0, 1, 2, 3, 4]


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
    last=True,
    ckpt_label="",  # <-- new: used to name the output
):
    os.makedirs(outdir, exist_ok=True)

    # 1. Base pipeline (same as in train.py)
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
    # -----------------------------
    # LOAD CKPT (if provided)
    # -----------------------------
    if ckpt_dir is None:
        print("Using BASE pretrained model (no checkpoint loaded)")
    else:
        if use_lora:
            pipe.unet.load_attn_procs(ckpt_dir)
            print(f"Loaded LoRA from {ckpt_dir}")
        else:
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

    images, eps_list, latents_list, log_probs = pipeline_with_logprob(
        pipe,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        output_type="pt",
    )

    # prefix like "ckpt3_" so files do not overwrite each other
    fname_prefix = f"{ckpt_label}_" if ckpt_label else ""

    if last:
        latents = latents_list[-1]
        t = len(latents_list)

        img = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False
        )[0]

        do_denormalize = [True] * img.shape[0]
        img = pipe.image_processor.postprocess(
            img, output_type="pil", do_denormalize=do_denormalize
        )

        img[0].save(os.path.join(outdir, f"{fname_prefix}img_step_t{int(t)}.png"))

        print(f"[ckpt {ckpt_label} step {t:03d}] saved img")
    else:
        for i, latents in enumerate(latents_list):
            img = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor, return_dict=False
            )[0]

            do_denormalize = [True] * img.shape[0]
            img = pipe.image_processor.postprocess(
                img, output_type="pil", do_denormalize=do_denormalize
            )

            img[0].save(os.path.join(outdir, f"{fname_prefix}img_step_{i:03d}_t{i}.png"))
            print(f"[ckpt {ckpt_label} step {i:03d}] saved img")

    print(f"Done for ckpt {ckpt_label}. Trajectory saved to {outdir}")


def main():
    set_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Same as config.pretrained.model in training.")
    parser.add_argument("--revision", type=str, default="main",
                        help="Same as config.pretrained.revision.")
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR,
                        help="Path to one checkpoint_* directory; script will infer the root /checkpoints/.")
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--use_lora", default=True,
                        help="Match config.use_lora from training.")
    parser.add_argument("--outdir", type=str, default=OUT_DIR)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    args = parser.parse_args()

    # Use the 3rd GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n=== Sampling BASELINE (no checkpoint) ===")

    sample_trajectory(
        pretrained_model=args.pretrained_model,
        revision=args.revision,
        ckpt_dir=None,          # <-- NEW: no weights loaded
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        use_lora=False,         # <-- IMPORTANT: do NOT load LoRA for baseline
        outdir=args.outdir,
        height=args.height,
        width=args.width,
        device=device,
        last=True,
        ckpt_label="baseline",  # <-- filename prefix
    )
    # infer the checkpoints root: ".../checkpoints" from ".../checkpoints/checkpoint_3"
    ckpt_root = os.path.dirname(args.ckpt_dir)

    for n in CKPT_NUMS:
        ckpt_dir = os.path.join(ckpt_root, f"checkpoint_{n}")
        ckpt_label = f"e{(n+1)*20}"
        print(f"\n=== Sampling from {ckpt_dir} ({ckpt_label}) ===")

        sample_trajectory(
            pretrained_model=args.pretrained_model,
            revision=args.revision,
            ckpt_dir=ckpt_dir,
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
            last=True,          # only final image
            ckpt_label=ckpt_label,
        )


if __name__ == "__main__":
    main()
