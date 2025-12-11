"""
风格迁移推理脚本
支持所有模型的推理
"""
import argparse
import os
import time
import torch

from models.gatys import GatysStyleTransfer
from models.fast_style_transfer import FastStyleTransfer
from models.adain import AdaINModel
from utils.data_loader import load_image, save_image


def inference_gatys(args):
    """Gatys风格迁移推理"""
    print("=" * 50)
    print("Gatys风格迁移")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载图像
    print("加载图像...")
    content = load_image(args.content, args.image_size, device)
    style = load_image(args.style, args.image_size, device)
    
    # 创建模型
    model = GatysStyleTransfer(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        device=device
    )
    
    # 执行风格迁移
    start_time = time.time()
    output, loss_history = model.transfer(
        content, style,
        num_steps=args.num_steps,
        lr=args.lr
    )
    elapsed_time = time.time() - start_time
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    base_c = os.path.basename(args.content)
    name_c = os.path.splitext(base_c)[0]
    output_path = os.path.join(args.output_dir, f'{name_c}_gatys.jpg')
    save_image(output, output_path)
    
    print(f"\n推理完成!")
    print(f"耗时: {elapsed_time:.2f} 秒")
    print(f"结果已保存到: {output_path}")


def inference_fast_style_transfer(args):
    """Fast Style Transfer推理"""
    print("=" * 50)
    print("Fast Style Transfer")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载图像
    print("加载图像...")
    content = load_image(args.content, args.image_size, device)
    
    # 创建模型并加载权重
    fst = FastStyleTransfer(device=device)
    
    if args.checkpoint is None:
        print("错误: Fast Style Transfer需要提供训练好的模型权重 (--checkpoint)")
        return
    
    print(f"加载模型: {args.checkpoint}")
    fst.load_model(args.style_name, args.checkpoint)
    
    # 执行风格迁移
    start_time = time.time()
    output = fst.transfer(content, args.style_name)
    elapsed_time = time.time() - start_time
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.basename(args.content)
    name = os.path.splitext(base)[0]
    output_path = os.path.join(args.output_dir, f'{name}_fast.jpg')

    save_image(output, output_path)
    
    print(f"\n推理完成!")
    print(f"耗时: {elapsed_time:.4f} 秒")
    print(f"结果已保存到: {output_path}")


def inference_adain(args):
    """AdaIN推理"""
    print("=" * 50)
    print("AdaIN风格迁移")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载图像
    print("加载图像...")
    content = load_image(args.content, args.image_size, device)
    style = load_image(args.style, args.image_size, device)
    
    # 创建模型
    model = AdaINModel().to(device)
    
    # 加载训练好的解码器权重(如果提供)
    if args.checkpoint:
        print(f"加载模型: {args.checkpoint}")
        model.decoder.load_state_dict(
            torch.load(args.checkpoint, map_location=device)
        )
    else:
        print("警告: 未提供训练好的权重,使用随机初始化的解码器")
    
    model.eval()
    
    # 执行风格迁移
    start_time = time.time()
    with torch.no_grad():
        output = model(content, style, alpha=args.alpha)
    elapsed_time = time.time() - start_time
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    base_c = os.path.basename(args.content)
    name_c = os.path.splitext(base_c)[0]
    output_path = os.path.join(args.output_dir, f'{name_c}_adain.jpg')
    save_image(output, output_path)

    print(f"\n推理完成!")
    print(f"耗时: {elapsed_time:.4f} 秒")
    print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='风格迁移推理')
    
    # 模型选择
    parser.add_argument('--model', type=str, required=True,
                       choices=['gatys', 'fast_style_transfer', 'adain'],
                       help='使用的模型')
    
    # 输入输出
    parser.add_argument('--content', type=str, required=True,
                       help='内容图像路径')
    parser.add_argument('--style', type=str,
                       help='风格图像路径 (Gatys和AdaIN需要)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--checkpoint', type=str,
                       help='模型权重路径 (Fast Style Transfer和AdaIN需要)')
    
    # 图像参数
    parser.add_argument('--image_size', type=int, default=512,
                       help='图像大小')
    
    # Gatys特定参数
    parser.add_argument('--num_steps', type=int, default=300,
                       help='Gatys优化步数')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Gatys学习率')
    parser.add_argument('--content_weight', type=float, default=1,
                       help='内容损失权重')
    parser.add_argument('--style_weight', type=float, default=1e6,
                       help='风格损失权重')
    parser.add_argument('--tv_weight', type=float, default=1e-3,
                       help='TV损失权重')
    
    # Fast Style Transfer特定参数
    parser.add_argument('--style_name', type=str, default='default',
                       help='风格名称 (Fast Style Transfer)')
    
    # AdaIN特定参数
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='AdaIN风格强度 [0, 1]')
    
    args = parser.parse_args()
    
    # 验证输入
    if args.model in ['gatys', 'adain'] and not args.style:
        parser.error(f"{args.model}需要提供风格图像 (--style)")
    
    # 执行推理
    if args.model == 'gatys':
        inference_gatys(args)
    elif args.model == 'fast_style_transfer':
        inference_fast_style_transfer(args)
    elif args.model == 'adain':
        inference_adain(args)


if __name__ == "__main__":
    main()
