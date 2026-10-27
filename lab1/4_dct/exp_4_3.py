"""4.3：删除分块 DCT 高频系数，逆变换并观察图像质量。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
OUTPUT = BASE / "output/4_3"
KEEP = [1, 3, 5, 8, 15]  # 保留零基频率坐标 u+v < keep 的系数。

image = np.array(Image.open(COVER).convert("L"))
h, w = image.shape[0] // 8 * 8, image.shape[1] // 8 * 8
if h == 0 or w == 0:
    raise ValueError("载体至少需要8×8像素。")
blocks = image[:h, :w].astype(float).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
coefficients = dctn(blocks - 128, axes=(-2, -1), norm="ortho")
OUTPUT.mkdir(parents=True, exist_ok=True)
items = [("Original", image)]
for keep in KEEP:
    mask = np.add.outer(np.arange(8), np.arange(8)) < keep
    restored_blocks = idctn(coefficients * mask, axes=(-2, -1), norm="ortho") + 128
    restored = image.copy()
    restored[:h, :w] = np.clip(np.rint(restored_blocks.transpose(0, 2, 1, 3).reshape(h, w)), 0, 255).astype(np.uint8)
    Image.fromarray(restored).save(OUTPUT / f"keep_{keep:02d}.png")
    items.append((f"Keep u+v < {keep}", restored))

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, (title, result) in zip(axes.flat, items):
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=150)
plt.close(fig)
print("保留的低频系数越少，图像通常越平滑，纹理和边缘细节损失越明显。")
print(f"4.3 结果：{OUTPUT}")
