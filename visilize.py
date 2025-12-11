import os
from PIL import Image, ImageDraw, ImageFont

# ====== 配置 ======
styleid_dir = "styleid_output"  # StyleID 结果文件夹
content_dir = "data/test2"   # 原始内容图像所在文件夹
output_path = "styleid_grid_4x6_with_content.png"

# 列顺序：第 0 列是原图，后面是 6 种风格
style_order = [
    "Baroque",
    "Expressionism",
    "Impressionism",
    "Post-Impressionism",
    "Realism",
    "Romanticism",
]

# 标题栏文字（按列顺序）
header_labels = [
    "Content",
    "Baroque",
    "Expressionism",
    "Impressionism",
    "Post-Impressionism",
    "Realism",
    "Romanticism",
]

# ====== 1. 收集 styleid 输出文件，按 content_id & style 组织 ======
files = [f for f in os.listdir(styleid_dir) if f.lower().endswith(".png")]

# 映射: content_id -> {style_name: filepath}
stylized_dict = {}

for fname in files:
    # 例：000000000632_stylized_Baroque_0001.png
    parts = fname.split("_stylized_")
    if len(parts) != 2:
        continue
    content_id = parts[0]  # 000000000632
    style_part = parts[1]  # Baroque_0001.png
    style_name = style_part.rsplit("_", 1)[0]  # Baroque

    if content_id not in stylized_dict:
        stylized_dict[content_id] = {}
    stylized_dict[content_id][style_name] = os.path.join(styleid_dir, fname)

# ====== 2. 收集原图，按 content_id 匹配 ======
original_dict = {}
for fname in os.listdir(content_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue
    stem = os.path.splitext(fname)[0]  # 去掉扩展名
    original_dict[stem] = os.path.join(content_dir, fname)

# ====== 3. 选取 4 个 content_id（行） ======
content_ids = sorted(stylized_dict.keys())[:4]
if len(content_ids) < 4:
    print("警告：内容图不足 4 个，实际只有:", len(content_ids))

# ====== 4. 用一张图确定单元格尺寸 ======
sample_img_path = None
for cid in content_ids:
    # 优先用 stylized 图的尺寸
    for s in style_order:
        if s in stylized_dict[cid]:
            sample_img_path = stylized_dict[cid][s]
            break
    if sample_img_path:
        break

if sample_img_path is None:
    raise RuntimeError("没有找到任何风格化图像，请检查 styleid_output 目录。")

sample_img = Image.open(sample_img_path)
cell_w, cell_h = sample_img.size

rows = len(content_ids)          # 一般为 4 行
cols = 1 + len(style_order)      # 1 列原图 + 6 列风格 = 7 列
header_h = int(0.25 * cell_h)    # 标题栏高度，可以按需要调整

# ====== 5. 创建大画布 ======
grid_w = cols * cell_w
grid_h = header_h + rows * cell_h
grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
draw = ImageDraw.Draw(grid_img)

# 字体：如果你有中文字体，可以把路径改成具体的 ttf
try:
    # 示例：Windows 下的常见中文字体路径，可按实际修改
    font = ImageFont.truetype("arial.ttf", size=20)
except:
    font = ImageFont.load_default()

# ====== 6. 画标题栏 ======
for col_idx, label in enumerate(header_labels):
    x0 = col_idx * cell_w
    x1 = x0 + cell_w
    y0 = 0
    y1 = header_h

    # 可选：填充浅灰背景
    # draw.rectangle([x0, y0, x1, y1], fill=(240, 240, 240))

    # 用 textbbox 计算文字宽高
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = x0 + (cell_w - text_w) // 2
    text_y = y0 + (header_h - text_h) // 2
    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)


# ====== 7. 逐格粘贴原图 + 风格图 ======
for row_idx, cid in enumerate(content_ids):
    # 当前行左上角 y 坐标（跳过标题栏）
    base_y = header_h + row_idx * cell_h

    # 7.1 原图（第 0 列）
    orig_path = None
    if cid in original_dict:
        orig_path = original_dict[cid]
    else:
        # 有些情况下原图文件名可能带前缀/后缀，可以在这里打印调试
        print(f"原图缺失: content_id={cid}")

    if orig_path is not None:
        orig_img = Image.open(orig_path).resize((cell_w, cell_h))
        grid_img.paste(orig_img, (0, base_y))

    # 7.2 各种风格（第 1~6 列）
    for col_idx, style_name in enumerate(style_order, start=1):
        path = stylized_dict[cid].get(style_name, None)
        if path is None:
            print(f"缺失: content_id={cid}, style={style_name}")
            continue
        img = Image.open(path).resize((cell_w, cell_h))
        x = col_idx * cell_w
        y = base_y
        grid_img.paste(img, (x, y))

# ====== 8. 保存 ======
grid_img.save(output_path)
print("合成完成，保存到:", output_path)
