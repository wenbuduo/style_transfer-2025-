"""
损失函数模块
包含内容损失、风格损失和感知损失的实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """VGG19特征提取器，用于计算内容和风格损失"""
    
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        # 加载预训练的VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # 定义用于提取特征的层
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(4):  # relu1_2
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):  # relu2_2
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18):  # relu3_4
            self.slice3.add_module(str(x), vgg[x])
        for x in range(18, 27):  # relu4_4
            self.slice4.add_module(str(x), vgg[x])
        for x in range(27, 36):  # relu5_4
            self.slice5.add_module(str(x), vgg[x])
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """提取多层特征"""
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


class ContentLoss(nn.Module):
    """内容损失"""
    
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, input_features, target_features):
        """
        计算内容损失
        Args:
            input_features: 生成图像的特征
            target_features: 目标内容图像的特征
        Returns:
            内容损失值
        """
        return self.criterion(input_features, target_features)


class StyleLoss(nn.Module):
    """风格损失 (基于Gram矩阵)"""
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def gram_matrix(self, x):
        """
        计算Gram矩阵
        Args:
            x: 特征图 [B, C, H, W]
        Returns:
            Gram矩阵 [B, C, C]
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def forward(self, input_features, target_features):
        """
        计算风格损失
        Args:
            input_features: 生成图像的特征
            target_features: 目标风格图像的特征
        Returns:
            风格损失值
        """
        input_gram = self.gram_matrix(input_features)
        target_gram = self.gram_matrix(target_features)
        return self.criterion(input_gram, target_gram)


class TotalVariationLoss(nn.Module):
    """总变分损失，用于平滑生成的图像"""
    
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, x):
        """
        计算总变分损失
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            TV损失值
        """
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return (tv_h + tv_w) / (batch_size * c * h * w)


class PerceptualLoss(nn.Module):
    """
    感知损失 (结合内容损失和风格损失)
    用于训练Fast Style Transfer和AdaIN
    """
    
    def __init__(self, content_weight=1.0, style_weight=1e5, tv_weight=0):
        super(PerceptualLoss, self).__init__()
        
        self.feature_extractor = VGGFeatureExtractor()
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TotalVariationLoss()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # 定义哪些层用于内容和风格损失
        self.content_layers = [3]  # relu4_4
        self.style_layers = [0, 1, 2, 3]  # relu1_2, relu2_2, relu3_4, relu4_4
    
    def forward(self, output, content, style):
        """
        计算总损失
        Args:
            output: 生成的图像
            content: 内容图像
            style: 风格图像
        Returns:
            总损失和各项损失的字典
        """
        # 提取特征
        output_features = self.feature_extractor(output)
        content_features = self.feature_extractor(content)
        style_features = self.feature_extractor(style)
        
        # 计算内容损失
        content_loss = 0
        for i in self.content_layers:
            content_loss += self.content_loss(output_features[i], content_features[i])
        content_loss *= self.content_weight
        
        # 计算风格损失
        style_loss = 0
        for i in self.style_layers:
            style_loss += self.style_loss(output_features[i], style_features[i])
        style_loss *= self.style_weight
        
        # 计算TV损失
        tv_loss = self.tv_loss(output) * self.tv_weight if self.tv_weight > 0 else 0
        
        # 总损失
        total_loss = content_loss + style_loss + tv_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'content': content_loss.item(),
            'style': style_loss.item(),
            'tv': tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss
        }


def test_losses():
    """测试损失函数"""
    print("测试损失函数...")
    
    # 创建随机输入
    batch_size = 2
    img_size = 256
    output = torch.randn(batch_size, 3, img_size, img_size)
    content = torch.randn(batch_size, 3, img_size, img_size)
    style = torch.randn(batch_size, 3, img_size, img_size)
    
    # 测试感知损失
    perceptual_loss = PerceptualLoss(content_weight=1.0, style_weight=1e5)
    total_loss, loss_dict = perceptual_loss(output, content, style)
    
    print(f"总损失: {loss_dict['total']:.4f}")
    print(f"内容损失: {loss_dict['content']:.4f}")
    print(f"风格损失: {loss_dict['style']:.4f}")
    print(f"TV损失: {loss_dict['tv']:.4f}")
    print("损失函数测试通过!")


if __name__ == "__main__":
    test_losses()
