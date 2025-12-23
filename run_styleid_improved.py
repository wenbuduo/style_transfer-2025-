"""
StyleID 风格迁移 - 改进版（可直接运行）

在原始脚本的基础上加入两类改进：

1) cross-attention + self-attention 组合注入
   - 既注入 self-attn(attn1) 的 K/V，也可选注入 cross-attn(attn2) 的 K/V
   - Q 默认保留内容侧（并支持用 gamma / gamma_ca 做混合）

2) 多层注意力特征组合（跨层/跨尺度聚合）
   - neighbor：对 (l-r ... l ... l+r) 的 K/V 做加权聚合
   - pyramid：在 neighbor 的基础上加入“隔层采样”的跨尺度候选集合

并新增一个“数据体现”的评估模式：
  --eval_compare  生成 baseline（仅 self-attn、无聚合）vs improved（按你打开的开关）
                 对每个样本对输出：运行时间、SSIM(内容保留)、风格相似度(颜色+纹理) 等
                 并写入 CSV + 打印均值提升。

注意：
- 风格相似度这里用无需额外权重的轻量指标（颜色直方图相似度 + Laplacian 纹理直方图相似度）。
- SSIM 优先使用 skimage，如果你的环境没有 skimage，会自动退化到 1 - MSE 的近似。
"""

import argparse
import os
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
import math
import csv

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

    image = transforms.CenterCrop(min(x, y))(image)
    image = image.resize((size, size), resample=Image.Resampling.LANCZOS)

    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def tensor_to_uint8_img(x: torch.Tensor) -> Image.Image:
    """x: [1,3,H,W] in [0,1]"""
    x = x.detach().cpu().clamp(0, 1)
    x = x[0].permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)


def img_to_np01(img: Image.Image) -> np.ndarray:
    x = np.asarray(img).astype(np.float32) / 255.0
    return x


def _try_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """a,b: HWC float [0,1]"""
    try:
        from skimage.metrics import structural_similarity as ssim
        # multichannel=True 在新版本叫 channel_axis
        return float(ssim(a, b, channel_axis=2, data_range=1.0))
    except Exception:
        # fallback: 近似内容相似度
        mse = float(np.mean((a - b) ** 2))
        return float(max(0.0, 1.0 - mse))


def _hist_cosine(h1: np.ndarray, h2: np.ndarray) -> float:
    h1 = h1.astype(np.float32)
    h2 = h2.astype(np.float32)
    n1 = np.linalg.norm(h1) + 1e-8
    n2 = np.linalg.norm(h2) + 1e-8
    return float(np.dot(h1, h2) / (n1 * n2))


def color_hist_similarity(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """颜色直方图相似度（按通道）"""
    sims = []
    for c in range(3):
        h1, _ = np.histogram(a[..., c], bins=bins, range=(0.0, 1.0), density=True)
        h2, _ = np.histogram(b[..., c], bins=bins, range=(0.0, 1.0), density=True)
        sims.append(_hist_cosine(h1, h2))
    return float(np.mean(sims))


def texture_hist_similarity(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Laplacian 纹理直方图相似度（灰度）"""
    def lap_mag(x):
        g = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)
        # 3x3 Laplacian
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        # 简单卷积（边界用 reflect）
        pad = np.pad(g, ((1, 1), (1, 1)), mode='reflect')
        out = (
            k[0, 0] * pad[:-2, :-2] + k[0, 1] * pad[:-2, 1:-1] + k[0, 2] * pad[:-2, 2:] +
            k[1, 0] * pad[1:-1, :-2] + k[1, 1] * pad[1:-1, 1:-1] + k[1, 2] * pad[1:-1, 2:] +
            k[2, 0] * pad[2:, :-2] + k[2, 1] * pad[2:, 1:-1] + k[2, 2] * pad[2:, 2:]
        )
        return np.abs(out)

    la = lap_mag(a)
    lb = lap_mag(b)
    # 归一化到 [0,1] 再做直方图
    la = la / (la.max() + 1e-8)
    lb = lb / (lb.max() + 1e-8)
    h1, _ = np.histogram(la, bins=bins, range=(0.0, 1.0), density=True)
    h2, _ = np.histogram(lb, bins=bins, range=(0.0, 1.0), density=True)
    return _hist_cosine(h1, h2)


def style_similarity(style_img: Image.Image, out_img: Image.Image) -> float:
    a = img_to_np01(style_img)
    b = img_to_np01(out_img)
    c = color_hist_similarity(a, b)
    t = texture_hist_similarity(a, b)
    return float(0.5 * c + 0.5 * t)


def adain(cnt_feat, sty_feat):
    """AdaIN (Adaptive Instance Normalization)"""
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((cnt_feat - cnt_mean) / (cnt_std + 1e-8)) * sty_std + sty_mean
    return output


def load_model_from_config(config, ckpt):
    """从配置和权重文件加载模型"""
    print(f"正在加载模型: {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"  全局步数: {pl_sd['global_step']}")

    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("模型加载完成")
    return model


def _parse_int_list(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_weights(s: str):
    ws = [float(x.strip()) for x in s.split(',') if x.strip()]
    sm = sum(ws)
    if sm <= 0:
        return ws
    return [w / sm for w in ws]


def _infer_hw_from_tokens(tokens: torch.Tensor):
    n = tokens.shape[1]
    h = int(round(math.sqrt(n)))
    w = n // h
    if h * w != n:
        h = int(math.floor(math.sqrt(n)))
        w = n // h
    return h, w


def _tokens_to_spatial(tokens: torch.Tensor, h: int, w: int):
    b, n, c = tokens.shape
    x = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x


def _spatial_to_tokens(x: torch.Tensor):
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)


def _resize_tokens(tokens: torch.Tensor, target_n: int):
    if tokens is None:
        return None
    if tokens.shape[1] == target_n:
        return tokens
    h1, w1 = _infer_hw_from_tokens(tokens)
    x = _tokens_to_spatial(tokens, h1, w1)
    h2 = int(round(math.sqrt(target_n)))
    w2 = target_n // h2
    if h2 * w2 != target_n:
        h2 = int(math.floor(math.sqrt(target_n)))
        w2 = target_n // h2
    x2 = torch.nn.functional.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=False)
    return _spatial_to_tokens(x2)


def _collect_feat(step_feat: dict, block_idx: int, attn_kind: str, qkv: str):
    key = f"output_block_{block_idx}_{attn_kind}_attn_{qkv}"
    return step_feat.get(key, None)


def aggregate_kv(style_feats, t: int, block_idx: int, attn_kind: str,
                 target_n: int, mode: str, radius: int, weights):
    """聚合风格侧 K/V，并对齐到 target_n。

    注意：不同 output_block 的注意力通道维度（C）可能不同。
    因此跨层聚合时必须确保候选层与目标层的 C 一致，否则会在加权求和时报维度不匹配。
    """

    # 目标层的通道维度（用本层的 K/V 作为参照）
    ref_k = _collect_feat(style_feats[t], block_idx, attn_kind, 'k')
    ref_v = _collect_feat(style_feats[t], block_idx, attn_kind, 'v')
    ref_c = None
    if ref_k is not None:
        ref_c = ref_k.shape[-1]
    elif ref_v is not None:
        ref_c = ref_v.shape[-1]

    if mode == 'none':
        k = _resize_tokens(ref_k, target_n)
        v = _resize_tokens(ref_v, target_n)
        return k, v

    candidates = []

    if mode == 'neighbor':
        offsets = list(range(-radius, radius + 1))
        if len(weights) != len(offsets):
            ws = [1.0 / len(offsets)] * len(offsets)
        else:
            ws = weights

        for off, w0 in zip(offsets, ws):
            bidx = block_idx + off
            k_raw = _collect_feat(style_feats[t], bidx, attn_kind, 'k')
            v_raw = _collect_feat(style_feats[t], bidx, attn_kind, 'v')
            if k_raw is None or v_raw is None:
                continue
            # 过滤掉通道维度不一致的层（否则无法求和）
            if ref_c is not None and (k_raw.shape[-1] != ref_c or v_raw.shape[-1] != ref_c):
                continue
            k = _resize_tokens(k_raw, target_n)
            v = _resize_tokens(v_raw, target_n)
            if k is None or v is None:
                continue
            candidates.append((w0, k, v))

    elif mode == 'pyramid':
        idxs = {block_idx}
        for off in range(-radius, radius + 1):
            idxs.add(block_idx + off)
        for off in range(-2 * radius, 2 * radius + 1, 2):
            idxs.add(block_idx + off)
        idxs = sorted(list(idxs))
        w0 = 1.0 / max(1, len(idxs))
        for bidx in idxs:
            k_raw = _collect_feat(style_feats[t], bidx, attn_kind, 'k')
            v_raw = _collect_feat(style_feats[t], bidx, attn_kind, 'v')
            if k_raw is None or v_raw is None:
                continue
            if ref_c is not None and (k_raw.shape[-1] != ref_c or v_raw.shape[-1] != ref_c):
                continue
            k = _resize_tokens(k_raw, target_n)
            v = _resize_tokens(v_raw, target_n)
            if k is None or v is None:
                continue
            candidates.append((w0, k, v))

    if not candidates:
        return None, None

    k_sum = None
    v_sum = None
    w_sum = 0.0
    for w, k, v in candidates:
        if k_sum is None:
            k_sum = w * k
            v_sum = w * v
        else:
            k_sum = k_sum + w * k
            v_sum = v_sum + w * v
        w_sum += w
    if w_sum > 0:
        k_sum = k_sum / w_sum
        v_sum = v_sum / w_sum
    return k_sum, v_sum


def expanded_indices(base: list[int], mode: str, radius: int) -> list[int]:
    if mode == 'none':
        return sorted(list(set(base)))
    idxs = set(base)
    if mode in ('neighbor', 'pyramid'):
        for i in base:
            for off in range(-radius, radius + 1):
                idxs.add(i + off)
    if mode == 'pyramid':
        for i in base:
            for off in range(-2 * radius, 2 * radius + 1, 2):
                idxs.add(i + off)
    return sorted(list(idxs))


def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    """合并内容与风格特征（支持 self+cross + 聚合）"""
    merged_feats = [
        {'config': {'gamma': opt.gamma, 'T': opt.T, 'timestep': i}}
        for i in range(opt.ddim_steps)
    ]

    self_layers = _parse_int_list(opt.attn_layers)
    cross_layers = _parse_int_list(opt.cross_attn_layers) if opt.use_cross_attn else []

    weights = _parse_weights(opt.agg_weights)
    radius = opt.agg_radius
    agg_mode = opt.layer_agg

    for t in range(opt.ddim_steps):
        if t < (opt.ddim_steps - start_step):
            continue

        cnt_step = cnt_feats[t]
        # self-attn
        for block_idx in self_layers:
            q_cnt = cnt_step.get(f"output_block_{block_idx}_self_attn_q", None)
            if q_cnt is None:
                continue
            # Q：内容侧保留（也支持混合）
            prev_q = merged_feats[t].get(f"output_block_{block_idx}_self_attn_q", None)
            merged_feats[t][f"output_block_{block_idx}_self_attn_q"] = (
                q_cnt if prev_q is None else opt.gamma * q_cnt + (1.0 - opt.gamma) * prev_q
            )

            target_n = q_cnt.shape[1]
            k_agg, v_agg = aggregate_kv(sty_feats, t, block_idx, 'self', target_n, agg_mode, radius, weights)
            if k_agg is not None and v_agg is not None:
                merged_feats[t][f"output_block_{block_idx}_self_attn_k"] = k_agg
                merged_feats[t][f"output_block_{block_idx}_self_attn_v"] = v_agg

        # cross-attn（组合注入）
        if opt.use_cross_attn:
            for block_idx in cross_layers:
                q_cnt = cnt_step.get(f"output_block_{block_idx}_cross_attn_q", None)
                if q_cnt is None:
                    continue
                prev_q = merged_feats[t].get(f"output_block_{block_idx}_cross_attn_q", None)
                merged_feats[t][f"output_block_{block_idx}_cross_attn_q"] = (
                    q_cnt if prev_q is None else opt.gamma_ca * q_cnt + (1.0 - opt.gamma_ca) * prev_q
                )

                target_n = q_cnt.shape[1]
                k_agg, v_agg = aggregate_kv(sty_feats, t, block_idx, 'cross', target_n, agg_mode, radius, weights)
                if k_agg is not None and v_agg is not None:
                    merged_feats[t][f"output_block_{block_idx}_cross_attn_k"] = k_agg
                    merged_feats[t][f"output_block_{block_idx}_cross_attn_v"] = v_agg

    return merged_feats


def extract_features(model, sampler, image, uc, opt, unet_model,
                     save_self_indices, save_cross_indices,
                     idx_time_dict, time_idx_dict,
                     cache_path=None):
    """提取图像特征（支持缓存）"""
    global feat_maps

    device = image.device

    if cache_path and os.path.isfile(cache_path):
        print(f"  加载缓存特征: {os.path.basename(cache_path)}")
        with open(cache_path, 'rb') as f:
            cached_feat = pickle.load(f)
            z_enc = cached_feat[0]['z_enc'].to(device=device, dtype=image.dtype)
        return cached_feat, z_enc

    def save_feature_map(feature_map, filename, time_step):
        global feat_maps
        cur_idx = idx_time_dict[time_step]
        if cur_idx < 0 or cur_idx >= len(feat_maps):
            return
        feat_maps[cur_idx][filename] = feature_map.detach().to('cpu', dtype=torch.float16)

    def save_feature_maps(blocks, time_step, feature_type="output_block"):
        for block_idx, block in enumerate(blocks):
            if len(block) <= 1 or "SpatialTransformer" not in str(type(block[1])):
                continue

            # self-attn
            if block_idx in save_self_indices:
                attn1 = block[1].transformer_blocks[0].attn1
                save_feature_map(attn1.q, f"{feature_type}_{block_idx}_self_attn_q", time_step)
                save_feature_map(attn1.k, f"{feature_type}_{block_idx}_self_attn_k", time_step)
                save_feature_map(attn1.v, f"{feature_type}_{block_idx}_self_attn_v", time_step)

            # cross-attn（可选）
            if opt.use_cross_attn and block_idx in save_cross_indices:
                attn2 = block[1].transformer_blocks[0].attn2
                if hasattr(attn2, 'q') and hasattr(attn2, 'k') and hasattr(attn2, 'v'):
                    save_feature_map(attn2.q, f"{feature_type}_{block_idx}_cross_attn_q", time_step)
                    save_feature_map(attn2.k, f"{feature_type}_{block_idx}_cross_attn_k", time_step)
                    save_feature_map(attn2.v, f"{feature_type}_{block_idx}_cross_attn_v", time_step)

    def ddim_callback(pred_x0, xt, time_step):
        save_feature_maps(unet_model.output_blocks, time_step, "output_block")
        save_feature_map(xt, 'z_enc', time_step)

    print("  编码图像...")
    amp_ctx = autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        with amp_ctx:
            image_encoded = model.get_first_stage_encoding(model.encode_first_stage(image))
            z_enc, _ = sampler.encode_ddim(
                image_encoded.clone(),
                num_steps=opt.ddim_steps,
                unconditional_conditioning=uc,
                end_step=time_idx_dict[opt.ddim_steps - 1 - opt.start_step],
                callback_ddim_timesteps=opt.ddim_steps,
                img_callback=ddim_callback
            )

    features = copy.deepcopy(feat_maps)
    z_enc = features[0]['z_enc'].to(device=device, dtype=image.dtype)

    del image_encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if cache_path:
        print(f"  保存特征缓存: {os.path.basename(cache_path)}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)

    return features, z_enc


def stylize(model, sampler, cnt_img, sty_img, uc, opt, unet_model,
            self_layers, cross_layers,
            idx_time_dict, time_idx_dict,
            cnt_cache=None, sty_cache=None):
    """执行风格迁移"""
    global feat_maps
    feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T}} for _ in range(opt.ddim_steps)]

    device = cnt_img.device
    shape = [4, opt.image_size // 8, opt.image_size // 8]

    weights = _parse_weights(opt.agg_weights)
    radius = opt.agg_radius
    agg_mode = opt.layer_agg

    # 为了支持“跨层/跨尺度聚合”，我们需要在 inversion 阶段把候选层的特征都保存下来
    save_self = expanded_indices(self_layers, agg_mode, radius)
    save_cross = expanded_indices(cross_layers, agg_mode, radius)

    print("提取风格特征...")
    sty_feat, sty_z_enc = extract_features(
        model, sampler, sty_img, uc, opt, unet_model,
        save_self, save_cross,
        idx_time_dict, time_idx_dict,
        cache_path=sty_cache
    )

    print("提取内容特征...")
    cnt_feat, cnt_z_enc = extract_features(
        model, sampler, cnt_img, uc, opt, unet_model,
        save_self, save_cross,
        idx_time_dict, time_idx_dict,
        cache_path=cnt_cache
    )

    print("生成风格迁移图像...")
    with torch.no_grad():
        amp_ctx = autocast("cuda") if device.type == "cuda" else nullcontext()
        with amp_ctx:
            with model.ema_scope():
                adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                merged_feats = feat_merge(opt, cnt_feat, sty_feat, start_step=opt.start_step)

                # 统一把特征从 CPU -> GPU
                for step_feat in merged_feats:
                    for k, v in list(step_feat.items()):
                        if k == 'config':
                            continue
                        if isinstance(v, torch.Tensor):
                            step_feat[k] = v.to(device=device, dtype=cnt_img.dtype)

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

                x_samples = model.decode_first_stage(samples_ddim)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                img = tensor_to_uint8_img(x_samples)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return img


def find_images(directory):
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                images.append(rel_path)
    return sorted(images)


def evaluate_pair(content_img: Image.Image, style_img: Image.Image, out_img: Image.Image):
    c = img_to_np01(content_img)
    o = img_to_np01(out_img)
    content_ssim = _try_ssim(c, o)
    sty_sim = style_similarity(style_img, out_img)
    return content_ssim, sty_sim


def main():
    parser = argparse.ArgumentParser(description='StyleID 风格迁移（改进版）')

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
    parser.add_argument('--gamma', type=float, default=0.75, help='self-attn Query 保留比例')
    parser.add_argument('--T', type=float, default=1.5, help='Attention 温度系数')
    parser.add_argument('--attn_layers', type=str, default='6,7,8,9,10,11', help='self-attn 注入层')

    # ====== 改进 1：cross-attn + self-attn 组合注入 ======
    parser.add_argument('--use_cross_attn', action='store_true', help='启用 cross-attn (attn2) 注入')
    parser.add_argument('--cross_attn_layers', type=str, default='6,7,8,9,10,11', help='cross-attn 注入层')
    parser.add_argument('--gamma_ca', type=float, default=None, help='cross-attn Query 保留比例（默认=gamma）')

    # ====== 改进 2：多层/跨尺度聚合 ======
    parser.add_argument('--layer_agg', type=str, default='neighbor',
                        choices=['none', 'neighbor', 'pyramid'],
                        help='K/V 聚合：none=不聚合，neighbor=跨层邻域，pyramid=跨尺度集合')
    parser.add_argument('--agg_radius', type=int, default=1, help='neighbor/pyramid 聚合半径')
    parser.add_argument('--agg_weights', type=str, default='0.25,0.5,0.25', help='neighbor 聚合权重')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_images', type=int, default=-1, help='最多处理图像数（-1表示全部）')

    # 评估与对比
    parser.add_argument('--eval_compare', action='store_true',
                        help='同时跑 baseline vs improved，并输出指标提升（会更慢）')
    parser.add_argument('--compare_pairs', type=int, default=6,
                        help='eval_compare 时最多对比多少个 (content, style) 对')
    parser.add_argument('--metrics_csv', type=str, default='metrics.csv',
                        help='指标输出 CSV 文件名（在 output_dir 下）')

    opt = parser.parse_args()

    if opt.gamma_ca is None:
        opt.gamma_ca = opt.gamma

    if opt.start_step >= opt.ddim_steps:
        print(f"警告: start_step({opt.start_step}) >= ddim_steps({opt.ddim_steps})，自动调整 start_step = {opt.ddim_steps - 1}")
        opt.start_step = opt.ddim_steps - 1

    seed_everything(opt.seed)
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.cache_dir, exist_ok=True)

    print("=" * 70)
    print("StyleID 风格迁移（改进版）")
    print("=" * 70)

    print("\n[1/4] 加载模型...")
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        model = model.to(device).half()
    else:
        model = model.to(device)

    print("\n[2/4] 初始化采样器...")
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=0.0, verbose=False)

    unet_model = model.model.diffusion_model
    self_layers = _parse_int_list(opt.attn_layers)
    cross_layers = _parse_int_list(opt.cross_attn_layers) if opt.use_cross_attn else []

    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {t: i for i, t in enumerate(time_range)}
    time_idx_dict = {i: t for i, t in enumerate(time_range)}

    print("\n[2.5/4] 计算无条件文本嵌入...")
    uc = model.get_learned_conditioning([""]).to(device)
    if device.type == "cuda":
        uc = uc.half()

    print("\n[3/4] 扫描图像...")
    content_imgs = find_images(opt.content_dir)
    style_imgs = find_images(opt.style_dir)

    if opt.max_images > 0:
        content_imgs = content_imgs[:opt.max_images]
        style_imgs = style_imgs[:opt.max_images]

    print(f"找到 {len(content_imgs)} 张内容图像")
    print(f"找到 {len(style_imgs)} 张风格图像")
    print(f"总共需要生成 {len(content_imgs) * len(style_imgs)} 张图像")

    # baseline 参数（用于 eval_compare）
    baseline_opt = copy.deepcopy(opt)
    baseline_opt.use_cross_attn = False
    baseline_opt.layer_agg = 'none'
    baseline_opt.agg_radius = 0
    baseline_opt.agg_weights = '1.0'

    print("\n[4/4] 开始风格迁移...")
    print("=" * 70)

    metrics_rows = []
    total_count = 0
    t0_all = time.time()

    # 为 eval_compare 限制对比数量
    pair_budget = opt.compare_pairs if opt.eval_compare else None

    for sty_idx, sty_name in enumerate(style_imgs, 1):
        print(f"\n风格图像 [{sty_idx}/{len(style_imgs)}]: {sty_name}")
        print("-" * 70)

        sty_path = os.path.join(opt.style_dir, sty_name)
        sty_tensor = load_img(sty_path, opt.image_size).to(device)
        if device.type == "cuda":
            sty_tensor = sty_tensor.half()
        sty_pil = Image.open(sty_path).convert('RGB').resize((opt.image_size, opt.image_size), Image.Resampling.LANCZOS)
        sty_cache = os.path.join(opt.cache_dir, f"{os.path.splitext(sty_name)[0]}_sty.pkl")

        for cnt_idx, cnt_name in enumerate(content_imgs, 1):
            print(f"\n  内容图像 [{cnt_idx}/{len(content_imgs)}]: {cnt_name}")
            cnt_path = os.path.join(opt.content_dir, cnt_name)
            cnt_tensor = load_img(cnt_path, opt.image_size).to(device)
            if device.type == "cuda":
                cnt_tensor = cnt_tensor.half()
            cnt_pil = Image.open(cnt_path).convert('RGB').resize((opt.image_size, opt.image_size), Image.Resampling.LANCZOS)
            cnt_cache = os.path.join(opt.cache_dir, f"{os.path.splitext(cnt_name)[0]}_cnt.pkl")

            # -------- baseline --------
            if opt.eval_compare:
                t0 = time.time()
                out_base = stylize(
                    model, sampler, cnt_tensor, sty_tensor, uc, baseline_opt, unet_model,
                    self_layers=_parse_int_list(baseline_opt.attn_layers),
                    cross_layers=[],
                    idx_time_dict=idx_time_dict,
                    time_idx_dict=time_idx_dict,
                    cnt_cache=cnt_cache,
                    sty_cache=sty_cache,
                )
                base_time = time.time() - t0
                base_content, base_style = evaluate_pair(cnt_pil, sty_pil, out_base)

                base_name = f"{os.path.splitext(cnt_name)[0]}__{os.path.splitext(sty_name)[0]}__baseline.png"
                out_base.save(os.path.join(opt.output_dir, base_name))

            # -------- improved --------
            t1 = time.time()
            out_imp = stylize(
                model, sampler, cnt_tensor, sty_tensor, uc, opt, unet_model,
                self_layers=self_layers,
                cross_layers=cross_layers,
                idx_time_dict=idx_time_dict,
                time_idx_dict=time_idx_dict,
                cnt_cache=cnt_cache,
                sty_cache=sty_cache,
            )
            imp_time = time.time() - t1
            imp_content, imp_style = evaluate_pair(cnt_pil, sty_pil, out_imp)

            imp_name = f"{os.path.splitext(cnt_name)[0]}__{os.path.splitext(sty_name)[0]}__improved.png"
            out_imp.save(os.path.join(opt.output_dir, imp_name))

            # 记录指标
            row = {
                'content': cnt_name,
                'style': sty_name,
                'improved_time_s': round(imp_time, 4),
                'improved_ssim': round(imp_content, 6),
                'improved_style_sim': round(imp_style, 6),
            }

            if opt.eval_compare:
                row.update({
                    'baseline_time_s': round(base_time, 4),
                    'baseline_ssim': round(base_content, 6),
                    'baseline_style_sim': round(base_style, 6),
                    'delta_time_s': round(imp_time - base_time, 4),
                    'delta_ssim': round(imp_content - base_content, 6),
                    'delta_style_sim': round(imp_style - base_style, 6),
                })

            metrics_rows.append(row)
            total_count += 1

            print(f"  保存: {imp_name}")
            if opt.eval_compare:
                print(f"  baseline: time={base_time:.2f}s SSIM={base_content:.3f} styleSim={base_style:.3f}")
            print(f"  improved: time={imp_time:.2f}s SSIM={imp_content:.3f} styleSim={imp_style:.3f}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if pair_budget is not None:
                pair_budget -= 1
                if pair_budget <= 0:
                    break

        if pair_budget is not None and pair_budget <= 0:
            break

    # 写 CSV
    csv_path = os.path.join(opt.output_dir, opt.metrics_csv)
    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in metrics_rows:
                w.writerow(r)

    elapsed_all = time.time() - t0_all

    print("\n" + "=" * 70)
    print(f"完成! 生成 {total_count} 个样本（{'对比模式' if opt.eval_compare else '单模式'}）")
    print(f"总耗时: {elapsed_all:.2f} 秒")
    print(f"输出目录: {opt.output_dir}")
    if metrics_rows:
        print(f"指标 CSV: {csv_path}")

    # 打印均值提升
    if opt.eval_compare and metrics_rows:
        d_ssim = [r['delta_ssim'] for r in metrics_rows]
        d_style = [r['delta_style_sim'] for r in metrics_rows]
        d_time = [r['delta_time_s'] for r in metrics_rows]
        print("\n对比统计（improved - baseline）:")
        print(f"  平均 ΔSSIM: {float(np.mean(d_ssim)):.6f}")
        print(f"  平均 ΔStyleSim: {float(np.mean(d_style)):.6f}")
        print(f"  平均 ΔTime(s): {float(np.mean(d_time)):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
