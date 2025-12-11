"""
数据加载模块
处理COCO2017和WikiArt数据集的加载
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob


class ImageDataset(Dataset):
    """通用图像数据集"""
    
    def __init__(self, image_dir, transform=None, extensions=('.jpg', '.jpeg', '.png')):
        """
        Args:
            image_dir: 图像目录
            transform: 图像变换
            extensions: 支持的文件扩展名
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 获取所有图像文件
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, f'*{ext.upper()}')))
        
        self.image_paths = sorted(self.image_paths)
        print(f"找到 {len(self.image_paths)} 张图像在 {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


class StyleTransferDataset(Dataset):
    """风格迁移配对数据集"""
    
    def __init__(self, content_dir, style_dir, transform=None):
        """
        Args:
            content_dir: 内容图像目录
            style_dir: 风格图像目录
            transform: 图像变换
        """
        self.content_dataset = ImageDataset(content_dir, transform)
        self.style_dataset = ImageDataset(style_dir, transform)
    
    def __len__(self):
        # 返回较小数据集的大小
        return min(len(self.content_dataset), len(self.style_dataset))
    
    def __getitem__(self, idx):
        content_img, content_path = self.content_dataset[idx]
        # 对风格图像使用循环索引
        style_idx = idx % len(self.style_dataset)
        style_img, style_path = self.style_dataset[style_idx]
        
        return {
            'content': content_img,
            'style': style_img,
            'content_path': content_path,
            'style_path': style_path
        }


def get_transform(image_size=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    获取图像变换
    Args:
        image_size: 目标图像大小
        mean: 归一化均值
        std: 归一化标准差
    Returns:
        transform
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)  # 某些方法不需要归一化
    ])


def get_train_transform(image_size=256):
    """训练时的数据增强"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反归一化
    Args:
        tensor: 归一化后的tensor [C, H, W] 或 [B, C, H, W]
    Returns:
        反归一化后的tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean

def load_image(image_path, image_size=512, device='cpu', resize=True):
    """
    加载单张图像
    Args:
        image_path: 图像路径
        image_size: 目标尺寸
        device: 设备
    Returns:
        tensor图像
    """
    img = Image.open(image_path).convert('RGB')
    if resize and image_size is not None:
        img = img.resize((image_size, image_size))
    transform = transforms.ToTensor()
    img = transform(img).unsqueeze(0).to(device)

    return img

# def load_image(image_path, image_size=256, device='cpu'):
#     """
#     加载单张图像
#     Args:
#         image_path: 图像路径
#         image_size: 目标尺寸
#         device: 设备
#     Returns:
#         tensor图像
#     """
#     transform = get_transform(image_size)
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)
#     return image.to(device)


def save_image(tensor, save_path):
    """
    保存tensor图像
    Args:
        tensor: 图像tensor [C, H, W] 或 [B, C, H, W]
        save_path: 保存路径
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 确保值在[0, 1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像
    image = transforms.ToPILImage()(tensor.cpu())
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存
    image.save(save_path)


def create_dataloader(content_dir, style_dir, batch_size=4, image_size=256, 
                      shuffle=True, num_workers=4):
    """
    创建数据加载器
    Args:
        content_dir: 内容图像目录
        style_dir: 风格图像目录
        batch_size: 批大小
        image_size: 图像大小
        shuffle: 是否打乱
        num_workers: 工作进程数
    Returns:
        DataLoader
    """
    transform = get_train_transform(image_size)
    dataset = StyleTransferDataset(content_dir, style_dir, transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def test_dataloader():
    """测试数据加载器"""
    print("测试数据加载器...")
    
    # 创建测试目录和图像
    os.makedirs('../data/content', exist_ok=True)
    os.makedirs('../data/style1', exist_ok=True)
    
    # 创建测试图像
    for i in range(3):
        img = Image.new('RGB', (256, 256), color=(i*80, 100, 150))
        img.save(f'data/content/test_{i}.jpg')
        img.save(f'data/style/style_{i}.jpg')
    
    # 测试数据集
    dataset = ImageDataset('../data/content', get_transform())
    print(f"数据集大小: {len(dataset)}")
    
    # 测试配对数据集
    paired_dataset = StyleTransferDataset('../data/content', 'data/style', get_transform())
    print(f"配对数据集大小: {len(paired_dataset)}")
    
    # 测试DataLoader
    dataloader = create_dataloader('../data/content', 'data/style', batch_size=2)
    for batch in dataloader:
        print(f"内容图像形状: {batch['content'].shape}")
        print(f"风格图像形状: {batch['style'].shape}")
        break
    
    print("数据加载器测试通过!")


if __name__ == "__main__":
    test_dataloader()
