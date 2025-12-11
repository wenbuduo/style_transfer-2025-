"""
结果对比和可视化脚本
生成对比图表和可视化
"""
import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


def plot_metrics_comparison(results_dict, save_dir):
    """绘制指标对比图"""
    
    models = list(results_dict.keys())
    
    # 定义要对比的指标
    metrics_to_plot = [
        ('avg_content_loss', '内容损失', 'lower_better'),
        ('avg_style_loss', '风格损失', 'lower_better'),
        ('avg_psnr_content', 'PSNR (dB)', 'higher_better'),
        ('avg_ssim_content', 'SSIM', 'higher_better'),
        ('avg_lpips_content', 'LPIPS (内容)', 'lower_better'),
    ]
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name, direction) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # 提取数据
        values = []
        model_names = []
        for model in models:
            if metric_key in results_dict[model]:
                values.append(results_dict[model][metric_key])
                model_names.append(model)
        
        if not values:
            continue
        
        # 绘制柱状图
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(range(len(model_names)), values, color=colors[:len(model_names)])
        
        # 设置标签
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}' if value < 1 else f'{value:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        # 添加网格
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 标注最佳模型
        if direction == 'lower_better':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    # 删除多余的子图
    for idx in range(len(metrics_to_plot), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"指标对比图已保存: {save_path}")
    plt.close()


def create_visual_comparison(results_dir, models, save_dir):
    """创建视觉对比图"""
    
    # 找到所有模型的输出图像
    sample_images = {}
    for model in models:
        model_dir = os.path.join(results_dir, model)
        if os.path.exists(model_dir):
            images = glob.glob(os.path.join(model_dir, '*.jpg')) + \
                    glob.glob(os.path.join(model_dir, '*.png'))
            if images:
                sample_images[model] = images[0]  # 取第一张作为示例
    
    if not sample_images:
        print("警告: 没有找到可视化的图像")
        return
    
    # 创建对比图
    n_models = len(sample_images)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(5*(n_models+2), 5))
    
    # 加载内容和风格图像
    content_dir = 'data/content'
    style_dir = 'data/style1'
    
    content_images = glob.glob(os.path.join(content_dir, '*'))
    style_images = glob.glob(os.path.join(style_dir, '*'))
    
    # 显示内容图像
    if content_images:
        content_img = Image.open(content_images[0])
        axes[0].imshow(content_img)
        axes[0].set_title('内容图像', fontsize=14, fontweight='bold')
        axes[0].axis('off')
    
    # 显示风格图像
    if style_images:
        style_img = Image.open(style_images[0])
        axes[1].imshow(style_img)
        axes[1].set_title('风格图像', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    # 显示各模型的输出
    for idx, (model, img_path) in enumerate(sample_images.items(), start=2):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(model.upper(), fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'visual_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"视觉对比图已保存: {save_path}")
    plt.close()


def generate_report(results_dict, save_dir):
    """生成文本报告"""
    
    report_path = os.path.join(save_dir, 'comparison_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("神经风格迁移模型对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 模型列表
        f.write("评估模型:\n")
        for i, model in enumerate(results_dict.keys(), 1):
            f.write(f"  {i}. {model.upper()}\n")
        f.write("\n")
        
        # 详细指标
        f.write("详细评估指标:\n")
        f.write("-" * 80 + "\n")
        
        for model, metrics in results_dict.items():
            f.write(f"\n{model.upper()}:\n")
            f.write("  " + "-" * 76 + "\n")
            
            metric_display = [
                ('avg_content_loss', '内容损失'),
                ('avg_style_loss', '风格损失'),
                ('avg_psnr_content', 'PSNR'),
                ('avg_ssim_content', 'SSIM'),
                ('avg_lpips_content', 'LPIPS (内容)'),
                ('avg_lpips_style', 'LPIPS (风格)'),
            ]
            
            for key, name in metric_display:
                if key in metrics:
                    f.write(f"  {name:<20}: {metrics[key]:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # 最佳模型分析
        f.write("\n最佳模型分析:\n")
        f.write("-" * 80 + "\n")
        
        analyses = {
            'avg_content_loss': ('内容保持最佳', 'lower'),
            'avg_style_loss': ('风格迁移最佳', 'lower'),
            'avg_psnr_content': '图像质量最佳 (PSNR)',
            'avg_ssim_content': '结构相似度最佳',
        }
        
        for metric_key, description in analyses.items():
            if isinstance(description, tuple):
                desc, direction = description
            else:
                desc = description
                direction = 'higher'
            
            values = {}
            for model, metrics in results_dict.items():
                if metric_key in metrics:
                    values[model] = metrics[metric_key]
            
            if values:
                if direction == 'lower':
                    best_model = min(values, key=values.get)
                else:
                    best_model = max(values, key=values.get)
                
                f.write(f"\n{desc}: {best_model.upper()}\n")
                f.write(f"  值: {values[best_model]:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"对比报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='对比风格迁移结果')
    
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果目录')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['gatys', 'fast_style_transfer', 'adain', 'styleid'],
                       help='要对比的模型')
    
    args = parser.parse_args()
    
    # 读取评估结果
    summary_file = os.path.join(args.results_dir, 'evaluation_summary.json')
    
    if not os.path.exists(summary_file):
        print(f"错误: 找不到评估结果文件 {summary_file}")
        print("请先运行 evaluate.py")
        return
    
    with open(summary_file, 'r') as f:
        results_dict = json.load(f)
    
    # 创建输出目录
    comparison_dir = os.path.join(args.results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    print("生成对比分析...")
    
    # 生成图表
    plot_metrics_comparison(results_dict, comparison_dir)
    
    # 创建视觉对比
    create_visual_comparison(args.results_dir, args.models, comparison_dir)
    
    # 生成文本报告
    generate_report(results_dict, comparison_dir)
    
    print("\n" + "="*50)
    print("对比分析完成!")
    print("="*50)
    print(f"所有结果保存在: {comparison_dir}")


if __name__ == "__main__":
    main()
