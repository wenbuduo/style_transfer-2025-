"""
训练baseline模型 (Fast Style Transfer 和 AdaIN)
Gatys不需要训练,直接优化即可
"""
import argparse
import os
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.fast_style_transfer import FastStyleTransfer
from models.adain import AdaINModel
from utils.losses import PerceptualLoss
from utils.data_loader import create_dataloader, save_image


def train_fast_style_transfer(args):
    """训练Fast Style Transfer模型"""
    print("=" * 50)
    print("训练Fast Style Transfer")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    dataloader = create_dataloader(
        args.content_dir,
        args.style_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # 创建模型
    fst = FastStyleTransfer(device=device)
    model = fst.create_model(args.style_name)
    model.train()
    
    # 损失函数
    criterion = PerceptualLoss(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, 'fast_style_transfer'))
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            content = batch['content'].to(device)
            style = batch['style'].to(device)
            
            # 前向传播
            output = model(content)
            
            # 计算损失
            loss, loss_dict = criterion(output, content, style)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录
            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.2f}",
                'content': f"{loss_dict['content']:.2f}",
                'style': f"{loss_dict['style']:.2f}"
            })
            
            # TensorBoard记录
            if global_step % args.log_interval == 0:
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/content', loss_dict['content'], global_step)
                writer.add_scalar('Loss/style', loss_dict['style'], global_step)
                writer.add_scalar('Loss/tv', loss_dict['tv'], global_step)
            
            global_step += 1
        
        # Epoch结束
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.2f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                args.checkpoint_dir,
                'fast_style_transfer',
                f'{args.style_name}_epoch_{epoch+1}.pth'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fst.save_model(args.style_name, save_path)
            print(f"模型已保存: {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(
        args.checkpoint_dir,
        'fast_style_transfer',
        f'{args.style_name}_final.pth'
    )
    fst.save_model(args.style_name, final_path)
    print(f"训练完成! 最终模型: {final_path}")
    
    writer.close()


def train_adain(args):
    """训练AdaIN模型"""
    print("=" * 50)
    print("训练AdaIN")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    dataloader = create_dataloader(
        args.content_dir,
        args.style_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # 创建模型 (只训练解码器)
    model = AdaINModel().to(device)
    model.decoder.train()
    
    # 损失函数
    criterion = PerceptualLoss(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=0  # AdaIN通常不使用TV loss
    ).to(device)
    
    # 优化器 (只优化解码器)
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, 'adain'))
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            content = batch['content'].to(device)
            style = batch['style'].to(device)
            
            # 前向传播
            output = model(content, style, alpha=1.0)
            
            # 计算损失
            loss, loss_dict = criterion(output, content, style)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录
            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.2f}",
                'content': f"{loss_dict['content']:.2f}",
                'style': f"{loss_dict['style']:.2f}"
            })
            
            # TensorBoard记录
            if global_step % args.log_interval == 0:
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/content', loss_dict['content'], global_step)
                writer.add_scalar('Loss/style', loss_dict['style'], global_step)
            
            global_step += 1
        
        # Epoch结束
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.2f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                args.checkpoint_dir,
                'adain',
                f'decoder_epoch_{epoch+1}.pth'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.decoder.state_dict(), save_path)
            print(f"模型已保存: {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, 'adain', 'decoder_final.pth')
    torch.save(model.decoder.state_dict(), final_path)
    print(f"训练完成! 最终模型: {final_path}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='训练Baseline模型')
    
    # 模型选择
    parser.add_argument('--model', type=str, required=True,
                       choices=['fast_style_transfer', 'adain'],
                       help='要训练的模型')
    
    # 数据路径
    parser.add_argument('--content_dir', type=str, default='data/content',
                       help='内容图像目录')
    parser.add_argument('--style_dir', type=str, default='data/style',
                       help='风格图像目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=2,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批大小')
    parser.add_argument('--image_size', type=int, default=256,
                       help='图像大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    # 损失权重
    parser.add_argument('--content_weight', type=float, default=1.0,
                       help='内容损失权重')
    parser.add_argument('--style_weight', type=float, default=1e5,
                       help='风格损失权重')
    parser.add_argument('--tv_weight', type=float, default=1e-6,
                       help='TV损失权重')
    
    # 保存和日志
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志目录')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='保存模型的epoch间隔')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志记录间隔')
    
    # Fast Style Transfer特定参数
    parser.add_argument('--style_name', type=str, default='default',
                       help='风格名称 (仅用于Fast Style Transfer)')
    
    args = parser.parse_args()
    
    # 训练对应的模型
    if args.model == 'fast_style_transfer':
        train_fast_style_transfer(args)
    elif args.model == 'adain':
        train_adain(args)


if __name__ == "__main__":
    main()
