import os
import glob
from PIL import Image
import numpy as np
import imageio

noise_dir = "/root/autodl-tmp/ddpo-pytorch/ddim_out_steps/ddpo_IS_compress_pmpt_cat"

# find all noise images, sorted
noise_paths = sorted(glob.glob(os.path.join(noise_dir, "noise_step_*.png")))

imgs = []
sum_acc = None

for path in noise_paths:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0   # 0–1

    imgs.append(arr)

    if sum_acc is None:
        sum_acc = arr
    else:
        sum_acc += arr

# ----- Save summed image -----
# normalize to [0,1]
summed = sum_acc / len(imgs)
summed = np.clip(summed, 0, 1)

summed_img = (summed * 255).astype(np.uint8)
Image.fromarray(summed_img).save(os.path.join(noise_dir, "summed_noise.png"))
print("Saved:", os.path.join(noise_dir, "summed_noise.png"))

# ----- Make GIF -----
gif_path = os.path.join(noise_dir, "noise_evolution.gif")

# convert imgs list back to uint8
gif_frames = [(img * 255).astype(np.uint8) for img in imgs]
# print(gif_frames[0].shape)
imageio.mimsave(gif_path, gif_frames, fps=8)
print("Saved GIF:", gif_path)
