"""3.1：观察 LSB/MSB 位平面，以及最低位、最高位清零的效果。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
OUTPUT = BASE / "output/3_1"

image = np.array(Image.open(COVER).convert("RGB"))
OUTPUT.mkdir(parents=True, exist_ok=True)
items = [
    ("Original", image),
    ("LSB cleared", image & 254),
    ("MSB cleared", image & 127),
    ("LSB plane", (image & 1) * 255),
    ("MSB plane", ((image >> 7) & 1) * 255),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax in axes.flat:
    ax.axis("off")
for ax, (title, result) in zip(axes.flat, items):
    ax.imshow(result)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(OUTPUT / "bit_planes.png", dpi=150)
plt.close(fig)
print(f"3.1 结果：{OUTPUT}")
