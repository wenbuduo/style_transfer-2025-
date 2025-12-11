import os
from PIL import Image, ImageDraw, ImageFont

# ====== 配置 ======
content_dir = "data/content/test1"   # 原图目录
results_root = "results"            # 四个模型结果的根目录
output_path = "grid_4x5_models_with_content.png"

# 列顺序与标题
model_folders = [
    ("gatys", "Gatys"),
    ("fast_style_transfer", "Fast-ST"),
    ("adain", "AdaIN"),
    ("styleid", "StyleID"),
]

header_labels = [
    "Content(原图)",
    "Gatys",
    "Fast-ST",
    "AdaIN",
    "StyleID",
]

# ====== 1. 收集原图：content_id -> path ======
original_dict = {}
for fname in os.listdir(content_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue
    stem = os.path.splitext(fname)[0]  # 例如 000000000632
    original_dict[stem] = os.path.join(content_dir, fname)

# ====== 2. 收集每个模型的结果：content_id -> path ======
# overall_dict: content_id -> {"Content": path, "Gatys": ..., ...}
overall_dict = {}

# 先填原图
for cid, path in original_dict.items():
    if cid not in overall_dict:
        overall_dict[cid] = {}
    overall_dict[cid]["Content"] = path

# 再填各模型
for folder_name, label in model_folders:
    model_dir = os.path.join(results_root, folder_name)
    if not os.path.isdir(model_dir):
        print(f"警告：模型目录不存在: {model_dir}")
        continue

    for fname in os.listdir(model_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        # 例：000000000632_gatys.jpg
        stem = os.path.splitext(fname)[0]
        # 按最后一个下划线分割：000000000632_gatys -> (000000000632, gatys)
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        content_id = parts[0]

        if content_id not in overall_dict:
            overall_dict[content_id] = {}
        overall_dict[content_id][label] = os.path.join(model_dir, fname)

# ====== 3. 选取有原图的 content_id 并排序，取前 4 个 ======
content_ids = sorted(
    [cid for cid in overall_dict.keys() if "Content" in overall_dict[cid]]
)[:4]

if len(content_ids) < 4:
    print("警告：满足条件的内容图不足 4 个，实际只有:", len(content_ids))

if not content_ids:
    raise RuntimeError("没有找到任何 content_id，请检查目录和文件名。")

# ====== 4. 用一张图确定单元格尺寸 ======
sample_path = None
for cid in content_ids:
    # 优先找原图
    if "Content" in overall_dict[cid]:
        sample_path = overall_dict[cid]["Content"]
        break
    # 退而求其次从某个模型拿
    for _, label in model_folders:
        if label in overall_dict[cid]:
            sample_path = overall_dict[cid][label]
            break
    if sample_path:
        break

if sample_path is None:
    raise RuntimeError("没有找到任何可以作为样本的图片。")

sample_img = Image.open(sample_path)
cell_w, cell_h = sample_img.size

rows = len(content_ids)          # 4 行
cols = 1 + len(model_folders)    # 1 原图 + 4 模型 = 5 列
header_h = int(0.25 * cell_h)    # 标题栏高度

# ====== 5. 创建大画布 ======
grid_w = cols * cell_w
grid_h = header_h + rows * cell_h
grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
draw = ImageDraw.Draw(grid_img)

# 字体设置：如果有中文字体可以改成对应 ttf
try:
    font = ImageFont.truetype("arial.ttf", size=20)
except:
    font = ImageFont.load_default()

# ====== 6. 画标题栏 ======
for col_idx, label in enumerate(header_labels):
    x0 = col_idx * cell_w
    x1 = x0 + cell_w
    y0 = 0
    y1 = header_h

    # 可选：标题栏背景色
    # draw.rectangle([x0, y0, x1, y1], fill=(240, 240, 240))

    # 用 textbbox 计算文字宽高
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = x0 + (cell_w - text_w) // 2
    text_y = y0 + (header_h - text_h) // 2

    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

# ====== 7. 逐格粘贴每一行：原图 + 4 个模型 ======
for row_idx, cid in enumerate(content_ids):
    base_y = header_h + row_idx * cell_h

    # 7.1 原图（第 0 列）
    if "Content" in overall_dict[cid]:
        orig_img = Image.open(overall_dict[cid]["Content"]).resize((cell_w, cell_h))
        grid_img.paste(orig_img, (0, base_y))
    else:
        print(f"content_id={cid} 的原图缺失")

    # 7.2 四个模型结果（第 1~4 列）
    for col_idx, (folder_name, label) in enumerate(model_folders, start=1):
        path = overall_dict[cid].get(label, None)
        if path is None:
            print(f"缺失: content_id={cid}, model={label}")
            continue
        img = Image.open(path).resize((cell_w, cell_h))
        x = col_idx * cell_w
        y = base_y
        grid_img.paste(img, (x, y))

# ====== 8. 保存 ======
grid_img.save(output_path)
print("合成完成，保存到:", output_path)
