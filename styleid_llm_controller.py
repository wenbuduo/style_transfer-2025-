import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import copy
import argparse
from types import SimpleNamespace
from PIL import Image
import torch
import numpy as np
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from run_styleid import load_img, load_model_from_config, stylize
from openai import OpenAI      # 火山兼容 OpenAI 格式

# ---------- 1. 豆包 vision 客户端 ----------
client = OpenAI(
    api_key="fea90c0d-1e81-47f7-9bbb-db75abb824fb",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def image_to_url(pil_img: Image.Image) -> str:
    import base64, io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def llm_generate_style_card(style_path: str) -> dict:
    img = Image.open(style_path).convert("RGB")
    sys = ("你是一位风格分析师。根据图片输出一段严格 JSON，字段："
           "color_palette,brushstroke,texture,contrast,lighting,detail_level,"
           "style_strength(0~1),notes。不要有任何额外解释。")
    rsp = client.chat.completions.create(
        model="Doubao-Seed-1.6-vision",
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": image_to_url(img)}}
                  ]}],
        temperature=0.3,
        max_tokens=400
    )
    return json.loads(rsp.choices[0].message.content.strip())

def llm_judge_and_suggest(result_path: str, content_path: str, style_path: str) -> dict:
    imgs = [Image.open(p).convert("RGB") for p in (content_path, style_path, result_path)]
    sys = ("你是风格迁移评委。给定内容图、风格图、结果图，判断结果是否成功结合内容结构与风格样式。"
           "输出严格 JSON：{\"pass\":bool,\"delta\":{\"gamma\":float,\"T\":float,\"start_step\":int}} "
           "gamma∈[0,1] 越小越偏风格；T≥1 越大越偏风格；start_step 越早越偏风格。")
    rsp = client.chat.completions.create(
        model="doubao-seed-1-6-vision-250815",
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": image_to_url(imgs[0])}},
                      {"type": "image_url", "image_url": {"url": image_to_url(imgs[1])}},
                      {"type": "image_url", "image_url": {"url": image_to_url(imgs[2])}}
                  ]}],
        temperature=0.3,
        max_tokens=400
    )
    return json.loads(rsp.choices[0].message.content.strip())

# ---------- 2. Style Card → StyleID 参数 ----------
def style_card_to_styleid_opt(card, base):
    opt = copy.deepcopy(base)
    s = card["style_strength"]
    opt.gamma  = max(0.3, 1.0 - 0.7 * s)
    opt.T      = 1.0 + 1.2 * s
    opt.start_step = int(opt.ddim_steps * (0.8 - 0.5 * s))
    opt.start_step = max(1, min(opt.start_step, opt.ddim_steps - 1))
    opt.attn_layers = "8,9,10,11"
    return opt

# ---------- 3. 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", default="test/content/000000000139.jpg")
    parser.add_argument("--style", default="test/style/Baroque_0001.jpg")
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--ckpt",  default="models_weights/sd-v1-4.ckpt")
    parser.add_argument("--config", default="models_weights/stable-diffusion-v1/v1-inference.yaml")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_opt = SimpleNamespace(image_size=512, ddim_steps=50, start_step=45,
                               gamma=0.75, T=1.5, attn_layers="6,7,8,9,10,11")

    # 加载模型
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    model = model.to(device).half() if device.type == "cuda" else model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=base_opt.ddim_steps, ddim_eta=0.0, verbose=False)
    unet_model = model.model.diffusion_model
    self_attn_indices = list(map(int, base_opt.attn_layers.split(",")))
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {t: i for i, t in enumerate(time_range)}
    time_idx_dict = {i: t for i, t in enumerate(time_range)}
    uc = model.get_learned_conditioning([""]).to(device)
    if device.type == "cuda":
        uc = uc.half()

    # 加载图像
    cnt_img = load_img(args.content, base_opt.image_size).to(device)
    sty_img = load_img(args.style, base_opt.image_size).to(device)
    if device.type == "cuda":
        cnt_img = cnt_img.half()
        sty_img = sty_img.half()

    # LLM 初始风格卡片
    style_card = llm_generate_style_card(args.style)
    print(">>> Style Card:", json.dumps(style_card, indent=2, ensure_ascii=False))
    opt_styleid = style_card_to_styleid_opt(style_card, base_opt)

    # 闭环生成
    for r in range(args.rounds):
        print(f"\n=== Round {r+1} ===")
        out_path = os.path.join(args.outdir, f"round_{r+1}.png")
        result_img = stylize(model, sampler, cnt_img, sty_img, uc, opt_styleid,
                             unet_model, self_attn_indices, idx_time_dict, time_idx_dict, None, None)
        result_img.save(out_path)
        print("Saved:", out_path)

        judge = llm_judge_and_suggest(out_path, args.content, args.style)
        print(">>> Judge:", json.dumps(judge, indent=2, ensure_ascii=False))
        if judge["pass"]:
            print("LLM judge: PASS → finish")
            break

        delta = judge["delta"]
        opt_styleid.gamma += delta.get("gamma", 0.0)
        opt_styleid.T += delta.get("T", 0.0)
        opt_styleid.start_step = max(1, opt_styleid.start_step + delta.get("start_step", 0))
        print("Updated opt:", vars(opt_styleid))

    print("\nAll done! 结果在", args.outdir)

if __name__ == "__main__":
    main()