import os
import shutil
import random

# 设置路径
content_dir = '../data/content'  # 内容图像所在目录
train_dir = '../data/content/train'  # 训练集保存目录
val_dir = '../data/content/val'  # 验证集保存目录
test_dir = '../data/content/test'  # 测试集保存目录

# 创建保存目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 打乱文件顺序
random.shuffle(image_files)

# 划分数据集：70% 训练集，15% 验证集，15% 测试集
total_images = len(image_files)
train_split = int(0.7 * total_images)
val_split = int(0.85 * total_images)

train_images = image_files[:train_split]
val_images = image_files[train_split:val_split]
test_images = image_files[val_split:]

# 拷贝文件到相应的目录
def copy_files(file_list, target_dir):
    for file_name in file_list:
        src = os.path.join(content_dir, file_name)
        dst = os.path.join(target_dir, file_name)
        shutil.copy(src, dst)

# 拷贝到训练集、验证集、测试集
copy_files(train_images, train_dir)
copy_files(val_images, val_dir)
copy_files(test_images, test_dir)

# 输出划分信息
print(f"总共有 {total_images} 张图像")
print(f"训练集: {len(train_images)} 张")
print(f"验证集: {len(val_images)} 张")
print(f"测试集: {len(test_images)} 张")

print("数据集划分完成！")
