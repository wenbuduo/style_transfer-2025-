"""
Fast Style Transfer
论文: "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (2016)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 norm='instance', activation='relu'):
        super(ConvBlock, self).__init__()
        
        layers = [nn.ReflectionPad2d(padding)]
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0))
        
        # 归一化层
        if norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        
        # 激活函数
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1, norm='instance', activation='relu'),
            ConvBlock(channels, channels, 3, 1, 1, norm='instance', activation=None)
        )
    
    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """上采样块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(UpsampleBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, 
                             kernel_size//2, norm='instance', activation='relu')
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class TransformNet(nn.Module):
    """
    图像转换网络
    将内容图像转换为特定风格
    """
    
    def __init__(self):
        super(TransformNet, self).__init__()
        
        # 下采样层
        self.downsample = nn.Sequential(
            ConvBlock(3, 32, 9, 1, 4, norm='instance', activation='relu'),
            ConvBlock(32, 64, 3, 2, 1, norm='instance', activation='relu'),
            ConvBlock(64, 128, 3, 2, 1, norm='instance', activation='relu'),
        )
        
        # 残差块
        self.residual = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        # 上采样层
        self.upsample = nn.Sequential(
            UpsampleBlock(128, 64, 3, 1, 2),
            UpsampleBlock(64, 32, 3, 1, 2),
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, 1, 0),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            风格化图像 [B, 3, H, W]
        """
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        x = self.output(x)
        
        # 将输出从[-1, 1]映射到[0, 1]
        x = (x + 1) / 2
        
        return x


class FastStyleTransfer:
    """
    Fast Style Transfer模型封装
    每个风格需要训练一个单独的网络
    """
    
    def __init__(self, device='cuda'):
        """
        Args:
            device: 计算设备
        """
        self.device = device
        self.models = {}  # 存储不同风格的模型
    
    def create_model(self, style_name):
        """
        创建一个新的转换网络
        Args:
            style_name: 风格名称
        Returns:
            TransformNet模型
        """
        model = TransformNet().to(self.device)
        self.models[style_name] = model
        return model
    
    def load_model(self, style_name, checkpoint_path):
        """
        加载训练好的模型
        Args:
            style_name: 风格名称
            checkpoint_path: 模型权重路径
        """
        model = TransformNet().to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        self.models[style_name] = model
        return model
    
    def save_model(self, style_name, save_path):
        """
        保存模型
        Args:
            style_name: 风格名称
            save_path: 保存路径
        """
        if style_name in self.models:
            torch.save(self.models[style_name].state_dict(), save_path)
            print(f"模型已保存到: {save_path}")
        else:
            print(f"错误: 风格 '{style_name}' 的模型不存在")
    
    def transfer(self, content_img, style_name):
        """
        执行风格迁移
        Args:
            content_img: 内容图像 [B, 3, H, W]
            style_name: 风格名称
        Returns:
            风格化图像 [B, 3, H, W]
        """
        if style_name not in self.models:
            raise ValueError(f"风格 '{style_name}' 的模型不存在。请先训练或加载模型。")
        
        model = self.models[style_name]
        model.eval()
        
        content_img = content_img.to(self.device)
        
        with torch.no_grad():
            output = model(content_img)
        
        return output
    
    def get_trainable_model(self, style_name):
        """
        获取可训练的模型
        Args:
            style_name: 风格名称
        Returns:
            TransformNet模型
        """
        if style_name not in self.models:
            self.create_model(style_name)
        
        return self.models[style_name]


def test_fast_style_transfer():
    """测试Fast Style Transfer模型"""
    print("测试Fast Style Transfer模型...")
    
    # 创建测试输入
    content = torch.rand(2, 3, 256, 256)
    
    # 创建模型
    model = TransformNet()
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        output = model(content)
    
    print(f"输入形状: {content.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 测试封装类
    fst = FastStyleTransfer(device='cpu')
    fst.create_model('test_style')
    
    with torch.no_grad():
        output2 = fst.transfer(content, 'test_style')
    
    print(f"通过封装类的输出形状: {output2.shape}")
    
    print("Fast Style Transfer模型测试通过!")


if __name__ == "__main__":
    test_fast_style_transfer()
