"""4.4：DCT 两点法嵌入/提取，比较 alpha 和 JPEG 质量对误码率的影响。"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
MESSAGE = BASE / "data/message_4_4.bin"
OUTPUT = BASE / "output/4_4"
ALPHAS = [4, 8, 16, 32, 64]
QUALITIES = [10, 30, 50, 70, 90, 95, 100]
P1, P2 = (4, 1), (3, 2)  # 课件一基坐标 (5,2)、(4,3)。


def embed(image, bits, alpha):
    """每个完整8×8块嵌入1 bit，保证两系数之差至少为alpha。"""
    h, w = image.shape[0] // 8 * 8, image.shape[1] // 8 * 8
    if len(bits) > (h // 8) * (w // 8):
        raise ValueError("消息长度超过完整8×8块的数量。")
    if h == 0 or w == 0 or alpha <= 0:
        raise ValueError("需要至少8×8的载体和正数alpha。")
    blocks = image[:h, :w].astype(float).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    coefficients = dctn(blocks - 128, axes=(-2, -1), norm="ortho")
    flat = coefficients.reshape(-1, 8, 8)
    c1 = flat[:len(bits), P1[0], P1[1]].copy()
    c2 = flat[:len(bits), P2[0], P2[1]].copy()
    sign = bits.astype(float) * 2 - 1
    # bit=1: c1-c2>=alpha；bit=0: c2-c1>=alpha。
    change = sign * np.maximum(0, alpha - sign * (c1 - c2)) / 2
    flat[:len(bits), P1[0], P1[1]] = c1 + change
    flat[:len(bits), P2[0], P2[1]] = c2 - change
    restored = idctn(coefficients, axes=(-2, -1), norm="ortho") + 128
    result = image.copy()
    result[:h, :w] = np.clip(np.rint(restored.transpose(0, 2, 1, 3).reshape(h, w)), 0, 255).astype(np.uint8)
    return result


def extract(image, n):
    """比较对应块的两个中频系数，无需原图及alpha。"""
    h, w = image.shape[0] // 8 * 8, image.shape[1] // 8 * 8
    blocks = image[:h, :w].astype(float).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    coefficients = dctn(blocks - 128, axes=(-2, -1), norm="ortho").reshape(-1, 8, 8)[:n]
    return (coefficients[:, P1[0], P1[1]] > coefficients[:, P2[0], P2[1]]).astype(np.uint8)


image = np.array(Image.open(COVER).convert("L"))
bits = np.unpackbits(np.frombuffer(MESSAGE.read_bytes(), dtype=np.uint8))
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "jpeg").mkdir(exist_ok=True)
rows = []
items = [("Original", image)]
for alpha in ALPHAS:
    stego = embed(image, bits, alpha)
    Image.fromarray(stego).save(OUTPUT / f"alpha_{alpha:02d}.png")
    items.append((f"alpha={alpha}", stego))
    for quality in QUALITIES:
        path = OUTPUT / "jpeg" / f"alpha_{alpha:02d}_q_{quality:03d}.jpg"
        Image.fromarray(stego).save(path, quality=quality)
        compressed = np.array(Image.open(path).convert("L"))
        recovered = extract(compressed, len(bits))
        error_rate = float(np.mean(recovered != bits))
        rows.append({"alpha": alpha, "jpeg_quality": quality, "ber": error_rate})

# alpha=32的PNG用来演示消息提取；同一消息也用于所有参数组合。
saved = np.array(Image.open(OUTPUT / "alpha_32.png").convert("L"))
(OUTPUT / "extracted.bin").write_bytes(np.packbits(extract(saved, len(bits))).tobytes())
with (OUTPUT / "jpeg_ber.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["alpha", "jpeg_quality", "ber"])
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, (title, result) in zip(axes.flat, items):
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "alpha_comparison.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for alpha in ALPHAS:
    selected = [row for row in rows if row["alpha"] == alpha]
    ax.plot(QUALITIES, [row["ber"] for row in selected], marker="o", label=f"alpha={alpha}")
ax.set(xlabel="JPEG quality", ylabel="Bit error rate", title="DCT two-point method: JPEG robustness")
ax.legend()
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUTPUT / "jpeg_robustness.png", dpi=150)
plt.close(fig)
print("alpha越大通常越抗JPEG量化，同时对图像的修改也更明显。")
print(f"4.4 嵌入并提取 {len(bits)} bit，结果：{OUTPUT}")
