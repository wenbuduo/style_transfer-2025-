import os
import shutil
from collections import defaultdict
from tqdm import tqdm

# 设置风格和目标数量
style_count = {
    'Abstract Art': 3,
    'Art Nouveau (Modern)': 8,
    'Baroque': 150,
    'Cubism': 121,
    'High Renaissance': 79,
    'Mannerism': 129,
    'Neoclassicism': 5,
    'Realism': 150,
    'Symbolism': 108,
    'Ukiyo-e': 131,
    'Impressionism': 150,
    'Expressionism': 150,
    'Post-Impressionism': 150,
    'Romanticism': 150,
    'Very rare category': 7
}

# 目标文件夹
source_dir = "data/style1"
output_dir = "data/style"

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 初始化风格计数
style_images = defaultdict(list)

# 遍历所有图片并按照风格分类
for filename in tqdm(os.listdir(source_dir)):
    if filename.endswith(".jpg"):
        # 提取风格类别和编号，假设文件名格式为 "style_0000.jpg"
        # 这里用数字前缀作为风格类别（如 "3" 对应风格 "Baroque"）
        style = filename.split('_')[0]  # 获取数字前缀作为风格
        style_name = None

        # 根据数字前缀映射风格名称
        if style == '3':
            style_name = 'Baroque'
        elif style == '2':
            style_name = 'Art Nouveau (Modern)'
        elif style == '0':
            style_name = 'Abstract Art'
        elif style == '4':
            style_name = 'Cubism'
        elif style == '7':
            style_name = 'High Renaissance'
        elif style == '9':
            style_name = 'Mannerism'
        elif style == '10':
            style_name = 'Neoclassicism'
        elif style == '12':
            style_name = 'Realism'
        elif style == '15':
            style_name = 'Symbolism'
        elif style == '17':
            style_name = 'Ukiyo-e'
        elif style == '20':
            style_name = 'Impressionism'
        elif style == '21':
            style_name = 'Expressionism'
        elif style == '23':
            style_name = 'Post-Impressionism'
        elif style == '24':
            style_name = 'Romanticism'
        elif style == '25':
            style_name = 'Very rare category'

        if style_name and len(style_images[style_name]) < style_count.get(style_name, 0):
            style_images[style_name].append(filename)

# 确保每个风格至少150张，保存到文件夹中
for style, images in style_images.items():
    if len(images) == style_count.get(style, 0):
        # 创建风格子文件夹
        style_folder = os.path.join(output_dir, style)
        os.makedirs(style_folder, exist_ok=True)

        # 移动图片到对应的风格文件夹
        for i, img_filename in enumerate(images):
            src_path = os.path.join(source_dir, img_filename)
            dst_path = os.path.join(style_folder, f"{style}_{i + 1:04d}.jpg")
            shutil.copy(src_path, dst_path)

        print(f"风格 '{style}' 有 {len(images)} 张图片，已保存到 {style_folder}")
    else:
        print(f"风格 '{style}' 没有足够的图像，跳过该风格。")

print("分类完成！")
