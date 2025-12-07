#!/usr/bin/env python3
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# Folder where your images are saved
ALGO = "ISKL"
PMPT = "monkey"
OUT_DIR = f"ddim_out/ddpo_{ALGO}_aesthetics_compare/"
SOURCE_DIR = f"ddim_out/ddpo_{ALGO}_aesthetics_pmpt_{PMPT}"

# Labels matching the prefixes you used when saving
# e.g. "baseline_img_step_t50.png", "ckpt1_img_step_t50.png", ...
CKPT_LABELS = ["baseline", "e40", "e80", "e120", "e160", "e200"]

# Optional: where to save the comparison image
COMPARE_PATH = os.path.join(OUT_DIR, f"aesthetics_{ALGO}_prompt_{PMPT}.png")

TITLE = f"Aesthetics_DDPO_{ALGO}_prompt_{PMPT}"

def find_image_for_label(source_dir, label):
    """
    Find the first image whose filename starts with '{label}_img_step_t'
    e.g. 'ckpt3_img_step_t50.png'.
    """
    pattern = os.path.join(source_dir, f"{label}_img_step_t*.png")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No image found for label '{label}' with pattern {pattern}")
    return matches[0]


def make_side_by_side(source_dir, labels, save_path=None, show=False):
    images = []
    titles = labels  # same order

    for label in labels:
        img_path = find_image_for_label(source_dir, label)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    n = len(images)

    # Wider figure without vertical padding
    width = 5
    plt.figure(figsize=(width * (n-1), width))

    # Global title
    plt.suptitle(TITLE, fontsize=26, y=0.98)

    for i, (img, title) in enumerate(zip(images, titles), start=1):
        ax = plt.subplot(1, n, i)

        ax.imshow(img)
        ax.axis("off")

        # ⬇️ Title at the bottom (bigger font)
        ax.set_title(title, fontsize=20, pad=10, y=-0.25)

    # Remove all padding between subplots
    plt.subplots_adjust(
        left=0, right=1, top=0.90, bottom=0.05,
        wspace=0, hspace=0
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved comparison figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()



if __name__ == "__main__":
    make_side_by_side(SOURCE_DIR, CKPT_LABELS, save_path=COMPARE_PATH, show=False)
