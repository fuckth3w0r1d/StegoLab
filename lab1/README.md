# 实验1-2：基本图像信息隐藏方法

Python + uv 实现课件第三、四、五部分。每个实验小点对应一个独立 `.py`，直接运行即可，不需要其他脚本作为入口。

## 运行

在项目根目录安装依赖，然后运行所需实验：

```bash
uv sync --locked
uv run python 3_lsb/exp_3_1.py
```

Python版本为3.12。各脚本的输入路径、参数在文件顶部，输入输出均相对脚本位置定位。图片写入对应的 `output/实验编号/`；压缩包已包含运行结果。

## 目录与实验

| 目录 | 脚本 | 实验内容 |
| --- | --- | --- |
| `3_lsb/` | `exp_3_1.py` | LSB/MSB清零与位平面观察 |
| `3_lsb/` | `exp_3_2.py` | 顺序LSB嵌入与提取 |
| `3_lsb/` | `exp_3_3.py` | 随机LSB嵌入与提取、随机位置、直方图和值对效应 |
| `4_dct/` | `exp_4_1.py` | 整体二维DCT |
| `4_dct/` | `exp_4_2.py` | 8×8分块DCT/IDCT及系数分布 |
| `4_dct/` | `exp_4_3.py` | 高频系数删除与重建图像比较 |
| `4_dct/` | `exp_4_4.py` | 两点法嵌入与提取，α、JPEG质量与误码率 |
| `5_binary/` | `exp_5_1.py` | 二值图像嵌入与提取，R0、R1、λ与随机加噪 |

每个部分的 `data/` 放载体和秘密信息，`output/` 放对应实验的隐写图、提取结果、图表及所需CSV数据。根目录仅保留README和uv项目配置。

例如运行其余实验：

```bash
uv run python 3_lsb/exp_3_2.py
uv run python 3_lsb/exp_3_3.py
uv run python 4_dct/exp_4_1.py
uv run python 4_dct/exp_4_2.py
uv run python 4_dct/exp_4_3.py
uv run python 4_dct/exp_4_4.py
uv run python 5_binary/exp_5_1.py
```

## 数据与约定

- 替换各部分 `data/cover.png` 即可使用自己的图片。自然图来自所附课件的背景；二值图为生成的波纹图。
- 3.2、5.1使用 `message.txt`；3.3使用100KB均匀随机字节 `message_3_3.bin`，便于观察值对效应；4.4使用512字节 `message_4_4.bin`，同一消息用于各组参数。均可替换为自己的文件。
- 随机选点种子默认为2026，提取使用相同种子；消息长度由实验中的秘密文件确定，不额外添加消息头。
- DCT使用灰度图，只处理完整8×8块，右侧和下侧残余像素保留。4.4按课件允许的方案选择两点法。
- 5.1以10×10块的黑像素比例编码：低比例为0，高比例为1。课件第25页文字与第24页映射相反，此处统一采用第24页规则。只修改原始八邻域边界，提取时跳过无效块。

## 结果观察

- **3.3**：查看 `histograms.png`。LSB替换使相邻偶数/奇数值的频数趋于接近；`positions.png`显示前2000个随机位置。
- **4.2**：查看 `block_dct.png` 和8×8的 `mean_abs_coefficients.csv`，低频区域通常有较大的系数。
- **4.3**：保留 `u+v<keep` 的低频系数，`keep`越小，纹理和边缘损失通常越明显。
- **4.4**：查看 `alpha_comparison.png`、`jpeg_robustness.png`、`jpeg_ber.csv`。增大α通常提高抗JPEG压缩能力，同时增加图像改动。
- **5.1**：查看 `noise_robustness.png` 和 `noise_results.csv`。按实际提取顺序比较消息，未提取位计错；噪声改变块有效性时也会导致消息错位。曲线使用固定种子的随机黑白翻转噪声。
