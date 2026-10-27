"""4.2：8×8 分块 DCT/IDCT，并观察系数的数值分布。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
OUTPUT = BASE / "output/4_2"

image = np.array(Image.open(COVER).convert("L"))
h, w = image.shape[0] // 8 * 8, image.shape[1] // 8 * 8
if h == 0 or w == 0:
    raise ValueError("载体至少需要8×8像素。")
blocks = image[:h, :w].astype(float).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
coefficients = dctn(blocks - 128, axes=(-2, -1), norm="ortho")
reconstructed_blocks = idctn(coefficients, axes=(-2, -1), norm="ortho") + 128
reconstructed = image.copy()
reconstructed[:h, :w] = np.clip(np.rint(reconstructed_blocks.transpose(0, 2, 1, 3).reshape(h, w)), 0, 255).astype(np.uint8)
mosaic = np.log1p(np.abs(coefficients)).transpose(0, 2, 1, 3).reshape(h, w)
mean_abs = np.abs(coefficients).mean(axis=(0, 1))

OUTPUT.mkdir(parents=True, exist_ok=True)
Image.fromarray(reconstructed).save(OUTPUT / "reconstructed.png")
np.savetxt(OUTPUT / "mean_abs_coefficients.csv", mean_abs, delimiter=",", fmt="%.4f")
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Original")
axes[0, 1].imshow(mosaic, cmap="magma")
axes[0, 1].set_title("8x8 block DCT: log(1+|C|)")
axes[1, 0].imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title("Block IDCT reconstruction")
plot = axes[1, 1].imshow(np.log1p(mean_abs), cmap="magma")
axes[1, 1].set(title="Mean |C(u,v)|: log scale", xlabel="v", ylabel="u")
fig.colorbar(plot, ax=axes[1, 1], fraction=.04)
for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "block_dct.png", dpi=150)
plt.close(fig)
print("分布特点：能量通常集中于左上角低频区域，高频系数大多较小。")
print(f"4.2 结果：{OUTPUT}")
