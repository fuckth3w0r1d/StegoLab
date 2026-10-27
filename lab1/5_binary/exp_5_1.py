"""5.1：二值图像边界隐写，分析R0、R1和lambda对随机加噪的鲁棒性。"""
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
MESSAGE = BASE / "data/message.txt"
OUTPUT = BASE / "output/5_1"
R0, R1, LAMBDA = .45, .55, .02
SEED = 2026  # 作为密钥，决定块的遍历顺序及块内随机取点。
THRESHOLDS = [(.49, .51), (.45, .55), (.40, .60)]
LAMBDAS = [.01, .02, .04]
NOISE = [0, .005, .01, .02, .05, .10]


def bounds(r0, r1, lam):
    """有效带对应的黑像素数量，10×10块每像素为1%。"""
    return (math.ceil((r0 - lam) * 100 - 1e-9), math.floor(r0 * 100 + 1e-9),
            math.ceil(r1 * 100 - 1e-9), math.floor((r1 + lam) * 100 + 1e-9))


def adjust(block, target, boundary, order):
    """只翻转原始八邻域边界上的点，避免新边界不断向外扩散。"""
    delta = target - int(block.sum())
    flat = block.reshape(-1)
    candidates = order[boundary.reshape(-1)[order] & (flat[order] == (delta < 0))]
    if len(candidates) < abs(delta):
        return None
    result = block.copy().reshape(-1)
    result[candidates[:abs(delta)]] = delta > 0
    return result.reshape(10, 10)


def embed(image, bits, r0, r1, lam, seed):
    black = image < 128
    rows, cols = black.shape[0] // 10, black.shape[1] // 10
    lo0, hi0, lo1, hi1 = bounds(r0, r1, lam)
    targets = ((lo0 + hi0) // 2, (lo1 + hi1 + 1) // 2)
    low = math.ceil((r0 - 3 * lam) * 100 - 1e-9)
    high = math.floor((r1 + 3 * lam) * 100 + 1e-9)
    rng = np.random.default_rng(seed)
    block_order = rng.permutation(rows * cols)
    k = 0
    for index in block_order:
        if k == len(bits):
            break
        row, col = divmod(int(index), cols)
        block = black[row*10:(row+1)*10, col*10:(col+1)*10]
        count = int(block.sum())
        bit = int(bits[k])
        # 采用块内部的八邻域，候选集合在修改前固定。
        boundary = maximum_filter(block, size=3, mode="nearest") != minimum_filter(block, size=3, mode="nearest")
        order = rng.permutation(100)
        usable = low <= count <= lo1 if bit == 0 else hi0 <= count <= high
        changed = adjust(block, targets[bit], boundary, order) if usable else None
        if changed is not None:
            block[:] = changed
            k += 1
        elif lo0 <= count <= hi0 or lo1 <= count <= hi1:
            # 原本在有效带的块若被跳过，须标为无效，以免提取时多读一位。
            invalid = sorted([max(0, low-1), min(100, high+1)], key=lambda x: abs(x-count))
            for target in invalid + [50, hi0+1, lo1-1]:
                changed = adjust(block, target, boundary, order)
                if changed is not None:
                    block[:] = changed
                    break
            else:
                raise ValueError("边界像素不足，无法将跳过的块标为无效。")
    if k < len(bits):
        raise ValueError(f"可用二值块不足：需要{len(bits)}块，只能嵌入{k}位。")
    return np.where(black, 0, 255).astype(np.uint8)


def extract(image, n, r0, r1, lam, seed):
    black = image < 128
    rows, cols = black.shape[0] // 10, black.shape[1] // 10
    counts = black[:rows*10, :cols*10].reshape(rows, 10, cols, 10).sum(axis=(1, 3)).reshape(-1)
    lo0, hi0, lo1, hi1 = bounds(r0, r1, lam)
    bits = []
    for index in np.random.default_rng(seed).permutation(rows * cols):
        count = int(counts[index])
        if lo0 <= count <= hi0:
            bits.append(0)
        elif lo1 <= count <= hi1:
            bits.append(1)
        if len(bits) == n:
            break
    return np.asarray(bits, dtype=np.uint8)


def add_noise(image, probability):
    flips = np.random.default_rng(SEED).random(image.shape) < probability
    return np.where(flips, 255 - image, image).astype(np.uint8)


image = np.array(Image.open(COVER).convert("L"))
image = np.where(image < 128, 0, 255).astype(np.uint8)
bits = np.unpackbits(np.frombuffer(MESSAGE.read_bytes(), dtype=np.uint8))
OUTPUT.mkdir(parents=True, exist_ok=True)
stego = embed(image, bits, R0, R1, LAMBDA, SEED)
Image.fromarray(stego).save(OUTPUT / "stego.png")
saved = np.array(Image.open(OUTPUT / "stego.png").convert("L"))
recovered = extract(saved, len(bits), R0, R1, LAMBDA, SEED)
(OUTPUT / "extracted.txt").write_bytes(np.packbits(recovered).tobytes())
noisy = add_noise(stego, .02)
Image.fromarray(noisy).save(OUTPUT / "noise_2pct.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, title, data in zip(axes, ["Original", "Binary stego", "2% random pixel flips"], [image, stego, noisy]):
    ax.imshow(data, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=150)
plt.close(fig)

results = []
for r0, r1 in THRESHOLDS:
    for lam in LAMBDAS:
        encoded = embed(image, bits, r0, r1, lam, SEED)
        for probability in NOISE:
            decoded = extract(add_noise(encoded, probability), len(bits), r0, r1, lam, SEED)
            # 按实际提取顺序比较；未提取到的位也计为错误。
            errors = int(np.count_nonzero(decoded != bits[:len(decoded)])) + len(bits) - len(decoded)
            results.append({"r0": r0, "r1": r1, "lambda": lam,
                            "noise_probability": probability, "error_rate": errors / len(bits)})
with (OUTPUT / "noise_results.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["r0", "r1", "lambda", "noise_probability", "error_rate"])
    writer.writeheader()
    writer.writerows(results)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (r0, r1) in zip(axes, THRESHOLDS):
    for lam in LAMBDAS:
        selected = [r for r in results if r["r0"] == r0 and r["lambda"] == lam]
        ax.plot(NOISE, [r["error_rate"] for r in selected], marker="o", label=f"lambda={lam:.0%}")
    ax.set(title=f"R0={r0:.0%}, R1={r1:.0%}", xlabel="Pixel flip probability", ylabel="Extraction error rate")
    ax.legend()
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUTPUT / "noise_robustness.png", dpi=150)
plt.close(fig)
print("噪声会改变块内黑像素比例，也可能改变块的有效性，使提取消息出现错位。")
print(f"5.1 嵌入并提取 {len(bits)} bit，结果：{OUTPUT}")
