"""3.3：随机 LSB 嵌入/提取，绘制位置和直方图，观察值对效应。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
# 均匀随机字节用作秘密信息，以便清楚观察直方图的值对效应。
MESSAGE = BASE / "data/message_3_3.bin"
OUTPUT = BASE / "output/3_3"
SEED = 2026  # 提取时使用同一种子，不重复选择分量位置。


def embed(image, bits, seed):
    if len(bits) > image.size:
        raise ValueError("消息长度超过 LSB 容量。")
    positions = np.random.default_rng(seed).permutation(image.size)[:len(bits)]
    result = image.copy()
    flat = result.reshape(-1)
    flat[positions] = (flat[positions] & 254) | bits
    return result, positions


def extract(image, n, seed):
    positions = np.random.default_rng(seed).permutation(image.size)[:n]
    return image.reshape(-1)[positions] & 1


image = np.array(Image.open(COVER).convert("RGB"))
bits = np.unpackbits(np.frombuffer(MESSAGE.read_bytes(), dtype=np.uint8))
stego, positions = embed(image, bits, SEED)
OUTPUT.mkdir(parents=True, exist_ok=True)
Image.fromarray(stego).save(OUTPUT / "stego.png")
saved = np.array(Image.open(OUTPUT / "stego.png").convert("RGB"))
(OUTPUT / "extracted.bin").write_bytes(np.packbits(extract(saved, len(bits), SEED)).tobytes())

# RGB 通道位置先除以3，转换成对应的像素坐标。
pixels = positions[:2000] // 3
y, x = np.divmod(pixels, image.shape[1])
fig, ax = plt.subplots(figsize=(11, 4))
ax.imshow(image)
ax.scatter(x, y, s=5, color="red", alpha=.65, linewidths=0)
ax.set_title(f"First {len(pixels)} random positions / {len(bits)} embedded bits")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "positions.png", dpi=150)
plt.close(fig)

before = np.bincount(image.reshape(-1), minlength=256)
after = np.bincount(saved.reshape(-1), minlength=256)
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, title, counts in zip(axes[0], ["Before embedding", "After random LSB"], [before, after]):
    ax.bar(np.arange(256), counts, width=1)
    ax.set(title=title, xlabel="RGB sample value", ylabel="Count")
axes[1, 0].plot(np.arange(128), before[::2] - before[1::2], label="Before")
axes[1, 0].plot(np.arange(128), after[::2] - after[1::2], label="After")
axes[1, 0].set(title="Value pairs: H(2k) - H(2k+1)", xlabel="Pair index k", ylabel="Count difference")
axes[1, 0].legend()
values = np.arange(48, 64)
axes[1, 1].bar(values - .2, before[values], width=.4, label="Before")
axes[1, 1].bar(values + .2, after[values], width=.4, label="After")
axes[1, 1].set(title="Adjacent even/odd values", xlabel="RGB sample value", ylabel="Count")
axes[1, 1].legend()
fig.tight_layout()
fig.savefig(OUTPUT / "histograms.png", dpi=150)
plt.close(fig)
np.savetxt(OUTPUT / "histogram.csv", np.column_stack([np.arange(256), before, after]),
           fmt="%d", delimiter=",", header="value,before,after", comments="")
print("值对效应：随机消息替换最低位，使相邻偶数/奇数灰度值的频数趋于接近。")
print(f"3.3 嵌入并提取 {len(bits)} bit，结果：{OUTPUT}")
