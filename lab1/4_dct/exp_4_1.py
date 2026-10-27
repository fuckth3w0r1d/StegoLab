"""4.1：整幅图像的二维 DCT 变换。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
OUTPUT = BASE / "output/4_1"

image = np.array(Image.open(COVER).convert("L"))
coefficients = dctn(image.astype(float), norm="ortho")
OUTPUT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original grayscale")
plot = axes[1].imshow(np.log1p(np.abs(coefficients)), cmap="magma")
axes[1].set_title("Whole-image DCT: log(1+|C|)")
fig.colorbar(plot, ax=axes[1], fraction=.03)
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "whole_dct.png", dpi=150)
plt.close(fig)
print(f"4.1 结果：{OUTPUT}")
