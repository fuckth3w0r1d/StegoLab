"""3.2：按 RGB 分量的行优先顺序替换 LSB，再提取前 n 位。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
COVER = BASE / "data/cover.png"
MESSAGE = BASE / "data/message.txt"
OUTPUT = BASE / "output/3_2"


def embed(image, bits):
    if len(bits) > image.size:
        raise ValueError("消息长度超过 LSB 容量。")
    result = image.copy()
    flat = result.reshape(-1)
    flat[:len(bits)] = (flat[:len(bits)] & 254) | bits
    return result


def extract(image, n):
    return image.reshape(-1)[:n] & 1


image = np.array(Image.open(COVER).convert("RGB"))
message = MESSAGE.read_bytes()
bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
stego = embed(image, bits)
OUTPUT.mkdir(parents=True, exist_ok=True)
Image.fromarray(stego).save(OUTPUT / "stego.png")
saved = np.array(Image.open(OUTPUT / "stego.png").convert("RGB"))
recovered = np.packbits(extract(saved, len(bits))).tobytes()
(OUTPUT / "extracted.txt").write_bytes(recovered)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, title, data in zip(axes, ["Original", "Sequential LSB"], [image, saved]):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=150)
plt.close(fig)
print(f"嵌入 {len(bits)} bit，提取消息：{recovered.decode('utf-8')}")
print(f"3.2 结果：{OUTPUT}")
