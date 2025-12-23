import os
# 全局配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import copy
import argparse
import warnings
from types import SimpleNamespace
from PIL import Image
import torch
import numpy as np
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from run_styleid import load_img, load_model_from_config, stylize
from openai import OpenAI
from openai import NotFoundError, APIError, RateLimitError, AuthenticationError


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ---------- 1. 豆包 vision 客户端配置 & 工具函数 ----------
def init_doubao_client(api_key: str) -> OpenAI:
    """初始化豆包客户端（使用可正常调用的配置）"""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"  # 保持和测试代码一致的base_url
        )
        # 移除模型列表验证（避免额外报错）
        return client
    except AuthenticationError as e:
        raise RuntimeError(
            f"API Key认证失败: {str(e)}\n"
            "请检查：1.API Key格式是否为 fea90c0d-1e81-47f7-9bbb-db75abb824fb 这种标准UUID格式\n"
            "2.Key是否在火山引擎控制台正确生成并授权"
        )
    except Exception as e:
        raise RuntimeError(f"初始化豆包客户端失败: {str(e)}")


# 使用你提供的可正常调用的API Key和Model名
API_KEY = "fea90c0d-1e81-47f7-9bbb-db75abb824fb"  # 修正后的合法UUID格式Key
MODEL_NAME = "doubao-seed-1-6-vision-250815"  # 你提供的有效模型ID
client = init_doubao_client(API_KEY)


def image_to_base64(pil_img: Image.Image, format: str = "JPEG") -> str:
    """图片转base64（适配豆包API要求）"""
    import base64
    import io
    try:
        # 压缩图片到合理尺寸，避免API超限
        if pil_img.size[0] > 1024 or pil_img.size[1] > 1024:
            pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format=format, quality=85)
        base64_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{base64_str}"
    except Exception as e:
        raise RuntimeError(f"图片转Base64失败: {str(e)}")


def safe_json_parse(text: str) -> dict:
    """安全解析JSON，兼容LLM返回格式"""
    try:
        # 清理LLM返回的多余字符
        text = text.strip().strip("```").strip("json").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        # 容错处理
        text = text.replace("'", "\"").replace("\n", "").replace("\t", "")
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"JSON解析失败，原始文本: {text[:200]} | 错误: {e}")


# ---------- 2. LLM 调用函数（使用正确的模型参数） ----------
def llm_generate_style_card(style_path: str, retry_times: int = 3) -> dict:
    """生成风格卡片（整合可正常调用的API参数）"""
    for attempt in range(retry_times):
        try:
            img = Image.open(style_path).convert("RGB")
            sys_prompt = (
                "你是一位专业的艺术风格分析师。请严格按照以下要求输出JSON格式内容，不要有任何额外解释、说明或换行：\n"
                "JSON字段要求：\n"
                "- color_palette: 字符串，描述主色调和配色方案\n"
                "- brushstroke: 字符串，描述笔触风格（如粗犷、细腻、写意等）\n"
                "- texture: 字符串，描述纹理特征（如磨砂、光滑、斑驳等）\n"
                "- contrast: 字符串，描述对比度（如高对比、低对比、柔和等）\n"
                "- lighting: 字符串，描述光影风格（如明暗强烈、柔和、逆光等）\n"
                "- detail_level: 字符串，描述细节丰富度（如高细节、简约、抽象等）\n"
                "- style_strength: 浮点数，0~1之间，代表风格强度\n"
                "- notes: 字符串，补充风格特征说明"
            )

            # 完全复用你提供的可正常调用的参数结构
            rsp = client.chat.completions.create(
                model=MODEL_NAME,  # 使用正确的模型ID
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_to_base64(img)}},
                        {"type": "text", "text": "分析这张图片的艺术风格，并按要求输出JSON"}  # 补充文本指令
                    ]}
                ],
                temperature=0.3,
                max_tokens=400
            )

            content = rsp.choices[0].message.content.strip()
            style_card = safe_json_parse(content)

            # 验证必要字段
            required_fields = ["style_strength"]
            for field in required_fields:
                if field not in style_card:
                    raise ValueError(f"风格卡片缺少必要字段: {field}")

            return style_card

        except NotFoundError as e:
            if attempt == retry_times - 1:
                raise RuntimeError(
                    f"模型不存在或无访问权限（重试{retry_times}次失败）: {str(e)}\n"
                    f"请确认模型ID '{MODEL_NAME}' 是否正确，以及API Key是否有权限访问该模型"
                )
            warnings.warn(f"第{attempt + 1}次调用失败（模型未找到），重试中...")
            continue
        except (APIError, RateLimitError) as e:
            if attempt == retry_times - 1:
                raise RuntimeError(f"API调用失败（重试{retry_times}次）: {str(e)}")
            warnings.warn(f"第{attempt + 1}次调用失败，重试中...")
            continue
        except Exception as e:
            if attempt == retry_times - 1:
                raise RuntimeError(f"生成风格卡片失败: {str(e)}")
            warnings.warn(f"第{attempt + 1}次调用失败，重试中...")
            continue


def llm_judge_and_suggest(result_path: str, content_path: str, style_path: str, retry_times: int = 3) -> dict:
    """LLM评估结果（使用正确的模型参数）"""
    for attempt in range(retry_times):
        try:
            imgs = [Image.open(p).convert("RGB") for p in (content_path, style_path, result_path)]
            sys_prompt = (
                "你是专业的风格迁移评委。请严格按照以下要求输出JSON格式内容，不要有任何额外解释：\n"
                "输入：内容图、风格图、迁移结果图\n"
                "输出JSON字段要求：\n"
                "- pass: 布尔值，true表示风格迁移成功，false表示需要调整\n"
                "- delta: 对象，包含调整参数：\n"
                "  - gamma: 浮点数，调整范围[-0.2, 0.2]，越小越偏风格\n"
                "  - T: 浮点数，调整范围[-0.5, 0.5]，越大越偏风格\n"
                "  - start_step: 整数，调整范围[-5, 5]，越小（越早）越偏风格"
            )

            # 构造和测试代码一致的多图片调用结构
            user_content = []
            # 内容图
            user_content.append({"type": "image_url", "image_url": {"url": image_to_base64(imgs[0])}})
            user_content.append({"type": "text", "text": "这是内容图"})
            # 风格图
            user_content.append({"type": "image_url", "image_url": {"url": image_to_base64(imgs[1])}})
            user_content.append({"type": "text", "text": "这是风格图"})
            # 结果图
            user_content.append({"type": "image_url", "image_url": {"url": image_to_base64(imgs[2])}})
            user_content.append({"type": "text", "text": "这是风格迁移结果图，请按要求评估并输出JSON"})

            rsp = client.chat.completions.create(
                model=MODEL_NAME,  # 使用正确的模型ID
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                max_tokens=400
            )

            content = rsp.choices[0].message.content.strip()
            judge_result = safe_json_parse(content)

            # 验证和修正参数范围
            judge_result = validate_judge_params(judge_result)
            return judge_result

        except Exception as e:
            if attempt == retry_times - 1:
                raise RuntimeError(f"评估风格迁移结果失败: {str(e)}")
            warnings.warn(f"第{attempt + 1}次评估失败，重试中...")
            continue


def validate_judge_params(judge_result: dict) -> dict:
    """验证并修正评估参数的范围"""
    # 确保pass字段存在
    if "pass" not in judge_result:
        judge_result["pass"] = False

    # 确保delta字段存在且参数在合理范围
    delta = judge_result.get("delta", {})
    # gamma范围限制
    gamma = delta.get("gamma", 0.0)
    delta["gamma"] = max(-0.2, min(0.2, gamma))
    # T范围限制
    T = delta.get("T", 0.0)
    delta["T"] = max(-0.5, min(0.5, T))
    # start_step范围限制
    start_step = delta.get("start_step", 0)
    delta["start_step"] = max(-5, min(5, int(start_step)))

    judge_result["delta"] = delta
    return judge_result


# ---------- 3. Style Card → StyleID 参数转换 ----------
def style_card_to_styleid_opt(card: dict, base_opt: SimpleNamespace) -> SimpleNamespace:
    """风格卡片转StyleID参数，增加参数范围限制"""
    opt = copy.deepcopy(base_opt)
    s = card["style_strength"]

    # 计算参数并限制范围
    opt.gamma = max(0.1, min(0.9, 1.0 - 0.7 * s))  # 限制gamma在0.1-0.9之间
    opt.T = max(1.0, min(3.0, 1.0 + 1.2 * s))  # 限制T在1.0-3.0之间
    opt.start_step = int(opt.ddim_steps * (0.8 - 0.5 * s))
    opt.start_step = max(1, min(opt.ddim_steps - 1, opt.start_step))  # 确保在有效范围
    opt.attn_layers = "8,9,10,11"

    return opt


# ---------- 4. 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="StyleID + LLM 闭环风格迁移")
    parser.add_argument("--content", default="test/content/000000000139.jpg")
    parser.add_argument("--style", default="test/style/Baroque_0001.jpg")
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--ckpt", default="models_weights/sd-v1-4.ckpt")
    parser.add_argument("--config", default="models_weights/stable-diffusion-v1/v1-inference.yaml")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--api-key", default="fea90c0d-1e81-47f7-9bbb-db75abb824fb", help="豆包API Key")
    args = parser.parse_args()

    # 更新全局API Key（支持命令行传入）
    global API_KEY, client
    if args.api_key != API_KEY:
        API_KEY = args.api_key
        client = init_doubao_client(API_KEY)

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    # 设置随机种子
    seed_everything(SEED)

    # 初始化基础参数
    base_opt = SimpleNamespace(
        image_size=512,
        ddim_steps=50,
        start_step=45,
        gamma=0.75,
        T=1.5,
        attn_layers="6,7,8,9,10,11"
    )

    try:
        # 加载模型（优化设备和精度）
        print("加载模型中...")
        config = OmegaConf.load(args.config)
        model = load_model_from_config(config, args.ckpt)

        # 根据设备调整精度
        if DEVICE.type == "cuda":
            model = model.to(DEVICE).half()
            torch.cuda.empty_cache()  # 清理显存
        else:
            model = model.to(DEVICE)

        # 初始化采样器
        sampler = DDIMSampler(model)
        sampler.make_schedule(
            ddim_num_steps=base_opt.ddim_steps,
            ddim_eta=0.0,
            verbose=False
        )

        # 准备StyleID相关参数
        unet_model = model.model.diffusion_model
        self_attn_indices = list(map(int, base_opt.attn_layers.split(",")))
        time_range = np.flip(sampler.ddim_timesteps)
        idx_time_dict = {t: i for i, t in enumerate(time_range)}
        time_idx_dict = {i: t for i, t in enumerate(time_range)}

        # 加载条件向量
        uc = model.get_learned_conditioning([""]).to(DEVICE)
        if DEVICE.type == "cuda":
            uc = uc.half()

        # 加载图像（优化精度）
        print("加载图像中...")
        cnt_img = load_img(args.content, base_opt.image_size).to(DEVICE)
        sty_img = load_img(args.style, base_opt.image_size).to(DEVICE)

        if DEVICE.type == "cuda":
            cnt_img = cnt_img.half()
            sty_img = sty_img.half()

        # LLM生成初始风格卡片
        print("生成初始风格卡片...")
        style_card = llm_generate_style_card(args.style)
        print(">>> 初始风格卡片:", json.dumps(style_card, indent=2, ensure_ascii=False))
        opt_styleid = style_card_to_styleid_opt(style_card, base_opt)

        # 闭环生成
        success = False
        for r in range(args.rounds):
            print(f"\n=== 第 {r + 1} 轮生成 ===")
            out_path = os.path.join(args.outdir, f"round_{r + 1}.png")

            # 风格迁移生成
            try:
                result_img = stylize(
                    model=model,
                    sampler=sampler,
                    cnt_img=cnt_img,
                    sty_img=sty_img,
                    uc=uc,
                    opt=opt_styleid,
                    unet_model=unet_model,
                    self_attn_indices=self_attn_indices,
                    idx_time_dict=idx_time_dict,
                    time_idx_dict=time_idx_dict
                )
                result_img.save(out_path)
                print(f"生成结果已保存: {out_path}")
            except Exception as e:
                print(f"风格迁移生成失败: {str(e)}")
                continue

            # LLM评估结果
            print("LLM评估结果中...")
            try:
                judge = llm_judge_and_suggest(out_path, args.content, args.style)
                print(">>> 评估结果:", json.dumps(judge, indent=2, ensure_ascii=False))

                if judge["pass"]:
                    print("✅ LLM评估通过，结束闭环")
                    success = True
                    break

                # 更新参数
                delta = judge["delta"]
                opt_styleid.gamma += delta.get("gamma", 0.0)
                opt_styleid.T += delta.get("T", 0.0)
                opt_styleid.start_step = max(
                    1,
                    min(opt_styleid.ddim_steps - 1,
                        opt_styleid.start_step + delta.get("start_step", 0))
                )

                # 再次限制参数范围
                opt_styleid.gamma = max(0.1, min(0.9, opt_styleid.gamma))
                opt_styleid.T = max(1.0, min(3.0, opt_styleid.T))

                print(f">>> 更新后的参数: {vars(opt_styleid)}")

            except Exception as e:
                print(f"评估失败，使用原参数继续: {str(e)}")
                continue

        if not success:
            print(f"\n⚠️  达到最大轮数({args.rounds})，未通过评估")

        print(f"\n🎉 任务完成！结果保存在: {args.outdir}")

    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()