import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # ⭐ 添加这一行即可

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

print("从 Hugging Face（国内镜像）下载 WikiArt（按风格分类）...")

# 加载数据集（自动走国内镜像）
dataset = load_dataset("huggan/wikiart", split="train", streaming=True)

os.makedirs('../data/style1', exist_ok=True)

style_count = defaultdict(int)
target_per_style = 150

print("开始下载并分类...")

for item in tqdm(dataset):
    try:
        style = item.get('style', 'unknown')

        if style_count[style] >= target_per_style:
            continue

        img = item['image']

        filename = f"data/style1/{style}_{style_count[style]:04d}.jpg"
        img.save(filename, 'JPEG')

        style_count[style] += 1

        if sum(style_count.values()) >= 1500:
            break

    except Exception:
        continue

print("\n✅ 完成！按风格分类下载:")
for style, count in sorted(style_count.items()):
    print(f"  {style}: {count} 张")

print(f"\n总计: {sum(style_count.values())} 张")
