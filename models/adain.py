"""
AdaIN风格迁移
论文: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (2017)
"""
import torch
import torch.nn as nn
from torchvision import models


class AdaIN(nn.Module):
    """自适应实例归一化层"""
    
    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps
    
    def forward(self, content_feat, style_feat):
        """
        Args:
            content_feat: 内容特征 [B, C, H, W]
            style_feat: 风格特征 [B, C, H, W]
        Returns:
            对齐后的特征 [B, C, H, W]
        """
        # 计算内容特征的均值和标准差
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_std = content_feat.std(dim=[2, 3], keepdim=True) + self.eps
        
        # 计算风格特征的均值和标准差
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        style_std = style_feat.std(dim=[2, 3], keepdim=True) + self.eps
        
        # 归一化内容特征并应用风格统计量
        normalized = (content_feat - content_mean) / content_std
        stylized = normalized * style_std + style_mean
        
        return stylized


class Encoder(nn.Module):
    """编码器 - 使用VGG19的前几层"""
    
    def __init__(self):
        super(Encoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # 使用到relu4_1
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        
        # 冻结参数
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """解码器 - 镜像编码器结构"""
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        # 使用反射填充来减少边界伪影
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, 1, 0),
        )
    
    def forward(self, x):
        return self.decoder(x)


class AdaINModel(nn.Module):
    """完整的AdaIN风格迁移模型"""
    
    def __init__(self):
        super(AdaINModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.adain = AdaIN()
        
        # 编码器不训练
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        """编码"""
        return self.encoder(x)
    
    def decode(self, t):
        """解码"""
        return self.decoder(t)
    
    def forward(self, content, style, alpha=1.0):
        """
        前向传播
        Args:
            content: 内容图像 [B, 3, H, W]
            style: 风格图像 [B, 3, H, W]
            alpha: 风格迁移强度 [0, 1]
        Returns:
            风格迁移后的图像 [B, 3, H, W]
        """
        # 编码
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        
        # AdaIN
        t = self.adain(content_feat, style_feat)
        
        # 插值控制风格强度
        t = alpha * t + (1 - alpha) * content_feat
        
        # 解码
        output = self.decode(t)
        
        return output
    
    def get_features_for_loss(self, x):
        """
        获取用于计算损失的多层特征
        """
        features = []
        for i, layer in enumerate(self.encoder.encoder):
            x = layer(x)
            if i in [1, 6, 11, 20]:  # relu1_1, relu2_1, relu3_1, relu4_1
                features.append(x)
        return features


def test_adain():
    """测试AdaIN模型"""
    print("测试AdaIN模型...")
    
    # 创建测试输入
    content = torch.rand(2, 3, 256, 256)
    style = torch.rand(2, 3, 256, 256)
    
    # 创建模型
    model = AdaINModel()
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        output = model(content, style, alpha=1.0)
    
    print(f"输入形状: {content.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试不同的alpha值
    with torch.no_grad():
        output_half = model(content, style, alpha=0.5)
        output_zero = model(content, style, alpha=0.0)
    
    print(f"alpha=1.0 输出范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"alpha=0.5 输出范围: [{output_half.min():.3f}, {output_half.max():.3f}]")
    print(f"alpha=0.0 输出范围: [{output_zero.min():.3f}, {output_zero.max():.3f}]")
    
    print("AdaIN模型测试通过!")


if __name__ == "__main__":
    test_adain()
