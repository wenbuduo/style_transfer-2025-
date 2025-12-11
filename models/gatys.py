"""
Gatys风格迁移 - 基于优化的方法
论文: "A Neural Algorithm of Artistic Style" (2015)
"""
import torch
import torch.optim as optim
import sys

sys.path.append('..')
from utils.losses import VGGFeatureExtractor, ContentLoss, StyleLoss, TotalVariationLoss


class GatysStyleTransfer:
    """
    Gatys风格迁移模型
    通过优化生成图像使其同时匹配内容图像的内容和风格图像的风格
    """

    def __init__(self, content_weight=1, style_weight=1e6, tv_weight=1e-3, device='cuda'):
        """
        Args:
            content_weight: 内容损失权重
            style_weight: 风格损失权重
            tv_weight: 总变分损失权重
            device: 计算设备
        """
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        # 创建特征提取器
        self.feature_extractor = VGGFeatureExtractor().to(device)
        self.feature_extractor.eval()

        # 损失函数
        self.content_loss_fn = ContentLoss()
        self.style_loss_fn = StyleLoss()
        self.tv_loss_fn = TotalVariationLoss()

        # 定义使用哪些层
        self.content_layers = [3]  # relu4_4
        self.style_layers = [0, 1, 2, 3, 4]  # relu1_2 到 relu5_4

    def transfer(self, content_img, style_img, num_steps=300, lr=0.01, init_random=False):
        """
        执行风格迁移
        Args:
            content_img: 内容图像 [1, 3, H, W]
            style_img: 风格图像 [1, 3, H, W]
            num_steps: 优化步数
            lr: 学习率
            init_random: 是否随机初始化（False则用内容图像初始化）
        Returns:
            生成的图像 [1, 3, H, W]
        """
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)

        # 初始化生成图像
        if init_random:
            generated = torch.randn_like(content_img).to(self.device)
        else:
            generated = content_img.clone()

        generated.requires_grad_(True)

        # 提取目标特征
        with torch.no_grad():
            content_features = self.feature_extractor(content_img)
            style_features = self.feature_extractor(style_img)

        # 优化器
        optimizer = optim.LBFGS([generated], lr=lr)

        # 存储损失历史
        loss_history = []

        print(f"开始Gatys风格迁移 (共{num_steps}步)...")

        step = [0]
        while step[0] < num_steps:

            def closure():
                optimizer.zero_grad()

                # 提取生成图像的特征
                generated_features = self.feature_extractor(generated)

                # 计算内容损失
                content_loss = 0
                for i in self.content_layers:
                    content_loss += self.content_loss_fn(
                        generated_features[i], content_features[i]
                    )
                content_loss *= self.content_weight

                # 计算风格损失
                style_loss = 0
                for i in self.style_layers:
                    style_loss += self.style_loss_fn(
                        generated_features[i], style_features[i]
                    )
                style_loss *= self.style_weight

                # 计算TV损失
                tv_loss = self.tv_loss_fn(generated) * self.tv_weight

                # 总损失
                total_loss = content_loss + style_loss + tv_loss
                total_loss.backward()

                # 记录
                if step[0] % 50 == 0:
                    print(f"Step {step[0]}: "
                          f"Total={total_loss.item():.2f}, "
                          f"Content={content_loss.item():.2f}, "
                          f"Style={style_loss.item():.2f}, "
                          f"TV={tv_loss.item():.6f}")

                loss_history.append({
                    'step': step[0],
                    'total': total_loss.item(),
                    'content': content_loss.item(),
                    'style': style_loss.item(),
                    'tv': tv_loss.item()
                })

                step[0] += 1

                return total_loss

            optimizer.step(closure)

            # 裁剪到[0, 1]
            with torch.no_grad():
                generated.clamp_(0, 1)

        print("Gatys风格迁移完成!")
        return generated.detach(), loss_history


def test_gatys():
    """测试Gatys模型"""
    print("测试Gatys风格迁移...")

    # 创建测试图像
    content = torch.rand(1, 3, 256, 256)
    style = torch.rand(1, 3, 256, 256)

    # 创建模型
    model = GatysStyleTransfer(
        content_weight=1,
        style_weight=1e6,
        tv_weight=1e-3,
        device='cpu'
    )

    # 执行风格迁移
    output, loss_history = model.transfer(
        content, style,
        num_steps=10,  # 测试用少量步数
        lr=0.01
    )

    print(f"输出图像形状: {output.shape}")
    print(f"损失历史记录数: {len(loss_history)}")
    print("Gatys模型测试通过!")


if __name__ == "__main__":
    test_gatys()