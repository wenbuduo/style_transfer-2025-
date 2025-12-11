"""
评估指标模块
包含PSNR、SSIM等图像质量评估指标
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class MetricsCalculator:
    """图像质量评估指标计算器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # 初始化LPIPS模型
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
    
    def calculate_psnr(self, img1, img2):
        """
        计算PSNR (Peak Signal-to-Noise Ratio)
        Args:
            img1, img2: 图像，可以是tensor或numpy array
        Returns:
            PSNR值 (dB)
        """
        # 转换为numpy array
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # 确保值在[0, 1]范围内
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # 如果是批量图像，逐个计算
        if img1.ndim == 4:  # [B, C, H, W]
            psnr_values = []
            for i in range(img1.shape[0]):
                # 转换为 [H, W, C] 格式
                im1 = np.transpose(img1[i], (1, 2, 0))
                im2 = np.transpose(img2[i], (1, 2, 0))
                psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
                psnr_values.append(psnr)
            return np.mean(psnr_values)
        else:
            # 单张图像
            if img1.ndim == 3 and img1.shape[0] == 3:  # [C, H, W]
                img1 = np.transpose(img1, (1, 2, 0))
                img2 = np.transpose(img2, (1, 2, 0))
            return peak_signal_noise_ratio(img1, img2, data_range=1.0)
    
    def calculate_ssim(self, img1, img2):
        """
        计算SSIM (Structural Similarity Index)
        Args:
            img1, img2: 图像，可以是tensor或numpy array
        Returns:
            SSIM值 [0, 1]
        """
        # 转换为numpy array
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # 确保值在[0, 1]范围内
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # 如果是批量图像，逐个计算
        if img1.ndim == 4:  # [B, C, H, W]
            ssim_values = []
            for i in range(img1.shape[0]):
                # 转换为 [H, W, C] 格式
                im1 = np.transpose(img1[i], (1, 2, 0))
                im2 = np.transpose(img2[i], (1, 2, 0))
                ssim = structural_similarity(im1, im2, multichannel=True, 
                                            data_range=1.0, channel_axis=2)
                ssim_values.append(ssim)
            return np.mean(ssim_values)
        else:
            # 单张图像
            if img1.ndim == 3 and img1.shape[0] == 3:  # [C, H, W]
                img1 = np.transpose(img1, (1, 2, 0))
                img2 = np.transpose(img2, (1, 2, 0))
            return structural_similarity(img1, img2, multichannel=True,
                                        data_range=1.0, channel_axis=2)
    
    def calculate_lpips(self, img1, img2):
        """
        计算LPIPS (Learned Perceptual Image Patch Similarity)
        越小表示越相似
        Args:
            img1, img2: tensor图像 [B, C, H, W] 或 [C, H, W]
        Returns:
            LPIPS距离
        """
        # 确保是tensor
        if not isinstance(img1, torch.Tensor):
            img1 = torch.from_numpy(img1)
        if not isinstance(img2, torch.Tensor):
            img2 = torch.from_numpy(img2)
        
        # 添加batch维度
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # 转到设备
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # 归一化到[-1, 1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        
        with torch.no_grad():
            distance = self.lpips_model(img1, img2)
        
        return distance.mean().item()
    
    def calculate_all_metrics(self, generated, content=None, style=None):
        """
        计算所有评估指标
        Args:
            generated: 生成的图像
            content: 内容图像 (用于计算与内容的相似度)
            style: 风格图像 (用于计算与风格的相似度)
        Returns:
            指标字典
        """
        metrics = {}
        
        if content is not None:
            metrics['psnr_content'] = self.calculate_psnr(generated, content)
            metrics['ssim_content'] = self.calculate_ssim(generated, content)
            metrics['lpips_content'] = self.calculate_lpips(generated, content)
        
        if style is not None:
            metrics['psnr_style'] = self.calculate_psnr(generated, style)
            metrics['ssim_style'] = self.calculate_ssim(generated, style)
            metrics['lpips_style'] = self.calculate_lpips(generated, style)
        
        return metrics


def test_metrics():
    """测试评估指标"""
    print("测试评估指标...")
    
    # 创建测试图像
    img1 = torch.rand(1, 3, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.1  # 添加噪声
    
    # 创建计算器
    calculator = MetricsCalculator(device='cpu')
    
    # 计算指标
    psnr = calculator.calculate_psnr(img1, img2)
    ssim = calculator.calculate_ssim(img1, img2)
    lpips_score = calculator.calculate_lpips(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"LPIPS: {lpips_score:.4f}")
    
    # 测试所有指标
    metrics = calculator.calculate_all_metrics(img1, content=img2, style=img2)
    print("\n所有指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("评估指标测试通过!")


if __name__ == "__main__":
    test_metrics()
