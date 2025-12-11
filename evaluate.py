"""
评估脚本
计算所有模型的评估指标
"""
import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import glob

from models.gatys import GatysStyleTransfer
from models.fast_style_transfer import FastStyleTransfer
from models.adain import AdaINModel
from utils.losses import PerceptualLoss, VGGFeatureExtractor
from utils.metrics import MetricsCalculator
from utils.data_loader import load_image, save_image


class StyleTransferEvaluator:
    """风格迁移评估器"""

    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_calc = MetricsCalculator(device)
        self.feature_extractor = VGGFeatureExtractor().to(device)
        self.feature_extractor.eval()

    def calculate_content_loss(self, generated, content):
        """计算内容损失"""
        with torch.no_grad():
            gen_features = self.feature_extractor(generated)
            content_features = self.feature_extractor(content)

            # 使用relu4_4层
            content_loss = torch.nn.functional.mse_loss(
                gen_features[3], content_features[3]
            )

        return content_loss.item()

    def calculate_style_loss(self, generated, style):
        """计算风格损失"""

        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram.div(c * h * w)

        with torch.no_grad():
            gen_features = self.feature_extractor(generated)
            style_features = self.feature_extractor(style)

            # 使用多层计算风格损失
            style_loss = 0
            for i in range(5):
                gen_gram = gram_matrix(gen_features[i])
                style_gram = gram_matrix(style_features[i])
                style_loss += torch.nn.functional.mse_loss(gen_gram, style_gram)

        return style_loss.item()

    def evaluate_single_pair(self, generated, content, style):
        """评估单个图像对"""
        results = {}

        # 内容损失和风格损失
        results['content_loss'] = self.calculate_content_loss(generated, content)
        results['style_loss'] = self.calculate_style_loss(generated, style)

        # PSNR和SSIM (与内容图像)
        results['psnr_content'] = self.metrics_calc.calculate_psnr(generated, content)
        results['ssim_content'] = self.metrics_calc.calculate_ssim(generated, content)

        # LPIPS
        results['lpips_content'] = self.metrics_calc.calculate_lpips(generated, content)
        results['lpips_style'] = self.metrics_calc.calculate_lpips(generated, style)

        return results

    def evaluate_directory(self, generated_dir, content_dir, style_dir):

        generated_files = sorted(glob.glob(os.path.join(generated_dir, '*.jpg')) +
                                 glob.glob(os.path.join(generated_dir, '*.png')))

        if len(generated_files) == 0:
            print(f"警告: 在 {generated_dir} 中没有找到图像")
            return None

        # 建立内容图索引表：{ "00000000632": "path/to/content.jpg" }
        content_index = {}
        for c in glob.glob(os.path.join(content_dir, "*")):
            name = os.path.splitext(os.path.basename(c))[0]
            content_index[name] = c

        # 风格图同理：只取目录中的第一张
        style_files = glob.glob(os.path.join(style_dir, "*"))
        if len(style_files) == 0:
            print("风格图目录为空！")
            return None
        style_path = style_files[0]

        all_results = []

        print(f"评估 {len(generated_files)} 张图像...")
        for gen_path in tqdm(generated_files):

            gen_name = os.path.basename(gen_path).split("_")[0]  # 取出前缀
            if gen_name not in content_index:
                print(f"跳过 {gen_path}: 找不到对应内容图 {gen_name}.jpg")
                continue

            content_path = content_index[gen_name]

            # 加载图像
            generated = load_image(gen_path, device=self.device)
            content = load_image(content_path, device=self.device)
            # 不进行预缩放，保持风格图原始纹理
            style = load_image(style_path, device=self.device, resize=False)

            # 最后统一调整成 generated 的大小，避免 VGG 尺寸不一致
            style = torch.nn.functional.interpolate(
                style,
                size=generated.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            # 评估
            results = self.evaluate_single_pair(generated, content, style)
            results['filename'] = os.path.basename(gen_path)
            all_results.append(results)

        if not all_results:
            print("没有有效的图像被评估。")
            return None

        # 计算平均值
        avg_results = {}
        for key in all_results[0].keys():
            if key != 'filename':
                values = [r[key] for r in all_results]
                avg_results[f'avg_{key}'] = np.mean(values)
                avg_results[f'std_{key}'] = np.std(values)

        return {'individual': all_results, 'average': avg_results}

def convert_to_jsonable(obj):
    """
    递归地将:
    - numpy.float32 / numpy.float64
    - torch.Tensor
    - list / dict 内部的 float32
    全部转换为 Python float 或 int
    """
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: convert_to_jsonable(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_to_jsonable(v) for v in obj]

    elif isinstance(obj, np.generic):
        return obj.item()

    elif torch.is_tensor(obj):
        return obj.item()

    elif isinstance(obj, float) or isinstance(obj, int) or obj is None:
        return obj

    else:
        try:
            return float(obj)
        except:
            return str(obj)


def evaluate_model(args, model_name):
    """评估单个模型"""
    print(f"\n{'='*50}")
    print(f"评估 {model_name.upper()}")
    print(f"{'='*50}")

    evaluator = StyleTransferEvaluator(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 确定生成图像的目录
    generated_dir = os.path.join(args.results_dir, model_name)

    if not os.path.exists(generated_dir):
        print(f"警告: 目录 {generated_dir} 不存在")
        return None

    # 评估
    results = evaluator.evaluate_directory(
        generated_dir,
        args.content_dir,
        args.style_dir
    )

    if results:
        # 打印平均结果
        print("\n平均评估指标:")
        print("-" * 50)
        for key, value in results['average'].items():
            print(f"{key}: {value:.4f}")

        # 保存结果
        output_file = os.path.join(args.results_dir, f'{model_name}_metrics.json')
        with open(output_file, 'w') as f:
            json.dump(convert_to_jsonable(results), f, indent=2)
        print(f"\n结果已保存到: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='评估风格迁移模型')

    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果目录')
    parser.add_argument('--content_dir', type=str, default='data/content/test1',
                       help='内容图像目录')
    parser.add_argument('--style_dir', type=str, default='data/baroque',
                       help='风格图像目录')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['gatys', 'fast_style_transfer', 'adain', 'styleid'],
                       help='要评估的模型列表')

    args = parser.parse_args()

    # 评估所有模型
    all_results = {}
    for model_name in args.models:
        results = evaluate_model(args, model_name)
        if results:
            all_results[model_name] = results['average']

    # 保存汇总结果
    if all_results:
        summary_file = os.path.join(args.results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(convert_to_jsonable(all_results), f, indent=2)


        print("\n" + "="*50)
        print("评估完成!")
        print("="*50)
        print(f"汇总结果已保存到: {summary_file}")

        # 打印对比表格
        print("\n模型对比:")
        print("-" * 80)
        print(f"{'模型':<20} {'内容损失':<15} {'风格损失':<15} {'PSNR':<10} {'SSIM':<10}")
        print("-" * 80)
        for model, metrics in all_results.items():
            print(f"{model:<20} "
                  f"{metrics.get('avg_content_loss', 0):<15.4f} "
                  f"{metrics.get('avg_style_loss', 0):<15.4f} "
                  f"{metrics.get('avg_psnr_content', 0):<10.2f} "
                  f"{metrics.get('avg_ssim_content', 0):<10.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
