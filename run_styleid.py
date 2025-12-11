"""
StyleID 风格迁移 - 独立运行脚本
基于 Stable Diffusion 的风格迁移方法
使用 DDIM Inversion + Attention Feature Injection
"""
import argparse
import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
import pickle
import time

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision.transforms as transforms

# 全局变量存储特征（在 DDIM inversion 的 callback 里写）
feat_maps = []


def load_img(path, size=256):
    """加载并预处理图像"""
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"  加载图像: {os.path.basename(path)}, 原始尺寸 ({x}, {y})")

    # 中心裁剪并调整大小
    image = transforms.CenterCrop(min(x, y))(image)
    image = image.resize((size, size), resample=Image.Resampling.LANCZOS)

    # 转换为tensor
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def adain(cnt_feat, sty_feat):
    """AdaIN (Adaptive Instance Normalization)"""
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
    return output


def load_model_from_config(config, ckpt):
    """从配置和权重文件加载模型"""
    print(f"正在加载模型: {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"  全局步数: {pl_sd['global_step']}")

    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    # 不在这里 .cuda()，统一在 main 里根据设备和精度处理
    model.eval()
    print("模型加载完成")
    return model


def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    """合并内容和风格特征

    注意：这里不再写死 50 步，改为使用 opt.ddim_steps
    """
    merged_feats = [
        {
            'config': {
                'gamma': opt.gamma,
                'T': opt.T,
                'timestep': i,
            }
        }
        for i in range(opt.ddim_steps)
    ]

    for i in range(len(merged_feats)):
        # 只在 start_step 之后的步数注入特征
        if i < (opt.ddim_steps - start_step):
            continue

        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            # Query 来自内容图像
            if ori_key.endswith('q'):
                merged_feats[i][ori_key] = cnt_feat[ori_key]
            # Key 和 Value 来自风格图像
            if ori_key.endswith('k') or ori_key.endswith('v'):
                merged_feats[i][ori_key] = sty_feat[ori_key]

    return merged_feats


def extract_features(model, sampler, image, uc, opt, unet_model,
                     self_attn_indices, idx_time_dict, time_idx_dict,
                     cache_path=None):
    """提取图像特征（支持缓存）"""
    global feat_maps

    device = image.device

    # 如果有缓存，直接加载（缓存里是 CPU 上的特征）
    if cache_path and os.path.isfile(cache_path):
        print(f"  加载缓存特征: {os.path.basename(cache_path)}")
        with open(cache_path, 'rb') as f:
            cached_feat = pickle.load(f)
            # z_enc 是保存在 features[0]['z_enc'] 里的 CPU tensor，这里移动回当前设备
            z_enc = cached_feat[0]['z_enc'].to(device=device, dtype=image.dtype)
        return cached_feat, z_enc

    # ---------- 关键修改 1：在 callback 里把特征搬到 CPU，避免显存爆炸 ----------

    def save_feature_map(feature_map, filename, time):
        """
        将当前步骤的特征保存到全局 feat_maps 中。
        这里立刻 .detach().to('cpu', dtype=torch.float16) 以释放 GPU 显存。
        """
        global feat_maps
        cur_idx = idx_time_dict[time]
        # 只要是合法 index 就保存
        if cur_idx < 0 or cur_idx >= len(feat_maps):
            return
        feat_maps[cur_idx][f"{filename}"] = (
            feature_map.detach().to('cpu', dtype=torch.float16)
        )

    def save_feature_maps(blocks, i, feature_type="output_block"):
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_indices:
                    attn_block = block[1].transformer_blocks[0].attn1
                    q = attn_block.q
                    k = attn_block.k
                    v = attn_block.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)

    def ddim_callback(pred_x0, xt, i):
        # 在每个时间步的 DDIM inversion 回调中存关键特征
        save_feature_maps(unet_model.output_blocks, i, "output_block")
        save_feature_map(xt, 'z_enc', i)

    # 编码图像到 latent
    print("  编码图像...")
    # 使用 autocast 降低显存（仅在 GPU 上生效）
    amp_ctx = autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        with amp_ctx:
            image_encoded = model.get_first_stage_encoding(
                model.encode_first_stage(image)
            )

            # DDIM Inversion
            z_enc, _ = sampler.encode_ddim(
                image_encoded.clone(),
                num_steps=opt.ddim_steps,
                unconditional_conditioning=uc,
                end_step=time_idx_dict[opt.ddim_steps - 1 - opt.start_step],
                callback_ddim_timesteps=opt.ddim_steps,
                img_callback=ddim_callback
            )

    # features 全是 CPU 上的（float16），避免长期占用 GPU 显存
    features = copy.deepcopy(feat_maps)

    # 取出 DDIM 反推开始时的 latent 编码（CPU → 当前 device）
    z_enc_cpu = features[0]['z_enc']
    z_enc = z_enc_cpu.to(device=device, dtype=image.dtype)

    # 清理中间变量和显存碎片
    del image_encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 保存缓存
    if cache_path:
        print(f"  保存特征缓存: {os.path.basename(cache_path)}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)

    return features, z_enc


def stylize(model, sampler, cnt_img, sty_img, uc, opt, unet_model,
            self_attn_indices, idx_time_dict, time_idx_dict,
            cnt_cache=None, sty_cache=None):
    """执行风格迁移"""
    global feat_maps
    # feat_maps 的长度与 ddim_steps 一致
    feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T}} for _ in range(opt.ddim_steps)]

    device = cnt_img.device
    shape = [4, opt.image_size // 8, opt.image_size // 8]

    # 提取风格特征
    print("提取风格特征...")
    sty_feat, sty_z_enc = extract_features(
        model, sampler, sty_img, uc, opt, unet_model,
        self_attn_indices, idx_time_dict, time_idx_dict, sty_cache
    )

    # 提取内容特征
    print("提取内容特征...")
    cnt_feat, cnt_z_enc = extract_features(
        model, sampler, cnt_img, uc, opt, unet_model,
        self_attn_indices, idx_time_dict, time_idx_dict, cnt_cache
    )

    # 执行风格迁移
    print("生成风格迁移图像...")
    with torch.no_grad():
        amp_ctx = autocast("cuda") if device.type == "cuda" else nullcontext()
        with amp_ctx:
            with model.ema_scope():
                # AdaIN 初始化（latent 上做）
                adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                # 特征融合（在 CPU 上完成 merge，之后整体搬到 GPU）
                merged_feats = feat_merge(opt, cnt_feat, sty_feat, start_step=opt.start_step)
                # ---------- 关键修改 2：在采样前统一把特征从 CPU → GPU，且用 half ----------
                for step_feat in merged_feats:
                    for k, v in list(step_feat.items()):
                        if k == "config":
                            continue
                        if isinstance(v, torch.Tensor):
                            step_feat[k] = v.to(device=device, dtype=cnt_img.dtype)
                # DDIM 采样
                samples_ddim, _ = sampler.sample(
                    S=opt.ddim_steps,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    x_T=adain_z_enc,
                    injected_features=merged_feats,
                    start_step=opt.start_step
                )
                # 解码图像
                x_samples = model.decode_first_stage(samples_ddim)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
                x_image = torch.from_numpy(x_samples).permute(0, 3, 1, 2)
                x_sample = 255. * rearrange(x_image[0].cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
    # 每次生成后尝试清理一下显存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return img


def main():
    parser = argparse.ArgumentParser(description='StyleID 风格迁移')

    # 数据路径
    parser.add_argument('--content_dir', default='data/test2', help='内容图像目录')
    parser.add_argument('--style_dir', default='data/style2', help='风格图像目录')
    parser.add_argument('--output_dir', default='styleid_output', help='输出目录')
    parser.add_argument('--cache_dir', default='cache_features', help='特征缓存目录')

    # 模型配置
    parser.add_argument('--ckpt', default='models_weights/sd-v1-4.ckpt', help='模型权重路径')
    parser.add_argument('--config', default='models_weights/stable-diffusion-v1/v1-inference.yaml', help='模型配置')

    # 风格迁移参数
    parser.add_argument('--image_size', type=int, default=256, help='图像大小')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM 步数')
    parser.add_argument('--start_step', type=int, default=45, help='开始注入特征的步数')
    parser.add_argument('--gamma', type=float, default=0.75, help='Query 保留比例')
    parser.add_argument('--T', type=float, default=1.5, help='Attention 温度系数')
    parser.add_argument('--attn_layers', type=str, default='6,7,8,9,10,11', help='注入特征的层')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_images', type=int, default=-1, help='最多处理图像数（-1表示全部）')

    opt = parser.parse_args()

    # 安全性：start_step 至少要 < ddim_steps
    if opt.start_step >= opt.ddim_steps:
        print(f"警告: start_step({opt.start_step}) >= ddim_steps({opt.ddim_steps})，自动调整 start_step = {opt.ddim_steps - 1}")
        opt.start_step = opt.ddim_steps - 1

    # 设置随机种子
    seed_everything(opt.seed)

    # 创建输出目录
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.cache_dir, exist_ok=True)

    print("=" * 70)
    print("StyleID 风格迁移")
    print("=" * 70)

    # 加载模型
    print("\n[1/4] 加载模型...")
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")

    # ---------- 关键修改 3：在 GPU 上使用 half precision 以减少显存 ----------
    if device.type == "cuda":
        model = model.to(device).half()
    else:
        model = model.to(device)

    # 初始化采样器
    print("\n[2/4] 初始化采样器...")
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=0.0, verbose=False)

    unet_model = model.model.diffusion_model
    self_attn_indices = list(map(int, opt.attn_layers.split(',')))

    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {t: i for i, t in enumerate(time_range)}
    time_idx_dict = {i: t for i, t in enumerate(time_range)}

    # 无条件嵌入
    print("\n[2.5/4] 计算无条件文本嵌入...")
    uc = model.get_learned_conditioning([""])
    uc = uc.to(device)
    if device.type == "cuda":
        uc = uc.half()

    # 获取图像列表（支持递归查找）
    def find_images(directory):
        """递归查找目录下的所有图像文件"""
        images = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.relpath(os.path.join(root, file), directory)
                    images.append(rel_path)
        return sorted(images)

    print("\n[3/4] 扫描图像...")
    content_imgs = find_images(opt.content_dir)
    style_imgs = find_images(opt.style_dir)

    if opt.max_images > 0:
        content_imgs = content_imgs[:opt.max_images]
        style_imgs = style_imgs[:opt.max_images]

    print(f"找到 {len(content_imgs)} 张内容图像")
    print(f"找到 {len(style_imgs)} 张风格图像")
    print(f"总共需要生成 {len(content_imgs) * len(style_imgs)} 张图像")

    # 执行风格迁移
    print("\n[4/4] 开始风格迁移...")
    print("=" * 70)

    start_time = time.time()
    total_count = 0

    for sty_idx, sty_name in enumerate(style_imgs, 1):
        print(f"\n风格图像 [{sty_idx}/{len(style_imgs)}]: {sty_name}")
        print("-" * 70)

        sty_path = os.path.join(opt.style_dir, sty_name)
        sty_img = load_img(sty_path, opt.image_size).to(device)
        if device.type == "cuda":
            sty_img = sty_img.half()
        sty_cache = os.path.join(opt.cache_dir, f"{os.path.splitext(sty_name)[0]}_sty.pkl")

        for cnt_idx, cnt_name in enumerate(content_imgs, 1):
            print(f"\n  内容图像 [{cnt_idx}/{len(content_imgs)}]: {cnt_name}")

            cnt_path = os.path.join(opt.content_dir, cnt_name)
            cnt_img = load_img(cnt_path, opt.image_size).to(device)
            if device.type == "cuda":
                cnt_img = cnt_img.half()
            cnt_cache = os.path.join(opt.cache_dir, f"{os.path.splitext(cnt_name)[0]}_cnt.pkl")

            # 执行风格迁移
            result_img = stylize(
                model, sampler, cnt_img, sty_img, uc, opt, unet_model,
                self_attn_indices, idx_time_dict, time_idx_dict,
                cnt_cache, sty_cache
            )

            # 保存结果
            output_name = f"{os.path.splitext(cnt_name)[0]}_stylized_{os.path.splitext(sty_name)[0]}.png"
            output_path = os.path.join(opt.output_dir, output_name)
            result_img.save(output_path)
            print(f"  保存: {output_name}")

            total_count += 1

            # 再清一下缓存，尽量防止长循环中显存碎片问题
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"完成! 总共生成 {total_count} 张图像")
    if total_count > 0:
        print(f"总耗时: {elapsed_time:.2f} 秒 (平均 {elapsed_time/total_count:.2f} 秒/张)")
    print(f"输出目录: {opt.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
