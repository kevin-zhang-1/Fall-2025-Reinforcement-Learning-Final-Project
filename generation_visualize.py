import os
import glob
import re
import imageio.v2 as imageio
from PIL import Image
import numpy as np

img_dir = "/root/autodl-tmp/ddpo-pytorch/ddim_out_steps/ddpo_IS_compress_pmpt_dog"
paths = glob.glob(os.path.join(img_dir, "img_step_t*.png"))

# numeric sort based on tXXXX
def get_t_index(path):
    name = os.path.basename(path)
    # extract number after 't'
    m = re.search(r"t(\d+)", name)
    return int(m.group(1)) if m else 999999

img_paths = sorted(paths, key=get_t_index)


print(f"Found {len(img_paths)} frames")

gif_path = os.path.join(img_dir, "img_steps_evolution.gif")
writer = imageio.get_writer(gif_path, mode="I", fps=6)  # change fps if you want

for p in img_paths:
    img = Image.open(p).convert("RGB")
    frame = np.array(img)          # shape (H, W, 3), uint8
    writer.append_data(frame)

writer.close()
print("GIF saved to:", gif_path)
