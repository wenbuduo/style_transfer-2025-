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


# ---------- 2. LLM 调用函数（增强智能判断） ----------
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


def llm_judge_and_suggest(result_path: str, content_path: str, style_path: str,
                          prev_problem: str = "", retry_times: int = 3) -> dict:
    """LLM评估结果（增加历史问题参考，避免反复横跳）"""
    for attempt in range(retry_times):
        try:
            imgs = [Image.open(p).convert("RGB") for p in (content_path, style_path, result_path)]
            # 增强版Prompt：加入历史问题参考，要求优先解决未修复的核心问题
            sys_prompt = (
                "你是专业的风格迁移评委，精通StyleID风格迁移参数的调整逻辑。请严格按照以下步骤和要求输出JSON格式内容，不要有任何额外解释：\n"
                "步骤1：参考历史问题（若有），对比分析三张图\n"
                "  - 历史问题：{prev_problem}\n"
                "  - 内容图：保留主体内容和结构\n"
                "  - 风格图：提取色彩、笔触、纹理、对比度、光影等风格特征\n"
                "  - 结果图：判断是否同时保留了内容图的主体结构，且融合了风格图的核心风格特征\n"
                "步骤2：判断是否通过（pass字段）\n"
                "  - true：结果图同时满足“保留内容主体”和“充分融合风格特征”两个条件\n"
                "  - false：结果图存在问题（风格不足/内容丢失过多/风格融合不自然），优先标记历史未修复的核心问题\n"
                "步骤3：若未通过，按以下规则给出调整delta\n"
                "  核心原则：优先修复历史核心问题，避免参数反向大幅回调；根据问题严重程度调整delta幅度\n"
                "  参数与风格/内容的对应关系：\n"
                "  - gamma：取值范围[0.1,0.9]，gamma越小→风格越浓（内容越弱）；gamma越大→内容越浓（风格越弱）\n"
                "  - T：取值范围[1.0,3.0]，T越大→风格越浓；T越小→内容越浓\n"
                "  - start_step：取值范围[1,49]，start_step越小（越早）→风格越浓；start_step越大（越晚）→内容越浓\n"
                "  问题严重程度与delta幅度对应：\n"
                "  - 轻微问题：delta幅度减半（gamma±0.05~0.075，T±0.1~0.15，start_step±1）\n"
                "  - 中度问题：正常幅度（gamma±0.075~0.125，T±0.15~0.25，start_step±1~2）\n"
                "  - 严重问题：正常幅度的1.5倍（gamma±0.125~0.15，T±0.25~0.3，start_step±2~3）\n"
                "  回调限制规则：\n"
                "  - 若上一轮因“内容丢失”调大了gamma/调小了T/调大了start_step，本轮即使风格不足，也不能将这些参数回调超过上一轮调整幅度的50%\n"
                "  - 若上一轮因“风格不足”调小了gamma/调大了T/调小了start_step，本轮即使内容丢失，也不能将这些参数回调超过上一轮调整幅度的50%\n"
                "输出JSON字段要求：\n"
                "- pass: 布尔值\n"
                "- problem: 字符串，描述结果图的具体问题，标注问题严重程度（轻微/中度/严重）\n"
                "- problem_type: 字符串，分类（content_loss/style_loss/blend_unnatural）\n"
                "- delta: 对象，包含gamma（浮点数）、T（浮点数）、start_step（整数），无需调整的参数设为0"
            ).format(prev_problem=prev_problem if prev_problem else "无")

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
                max_tokens=500  # 增加token数，容纳更多判断逻辑
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
    """验证并修正评估参数的范围，过滤无效调整"""
    # 确保必要字段存在
    if "pass" not in judge_result:
        judge_result["pass"] = False
    if "problem" not in judge_result:
        judge_result["problem"] = "未明确说明问题"
    if "problem_type" not in judge_result:
        judge_result["problem_type"] = "blend_unnatural"
    delta = judge_result.get("delta", {})

    # 初始化未指定的参数为0
    delta.setdefault("gamma", 0.0)
    delta.setdefault("T", 0.0)
    delta.setdefault("start_step", 0)

    # 限制delta的调整幅度（避免单次调整过大）
    delta["gamma"] = max(-0.15, min(0.15, delta["gamma"]))  # 单次最大调整±0.15
    delta["T"] = max(-0.3, min(0.3, delta["T"]))  # 单次最大调整±0.3
    delta["start_step"] = max(-3, min(3, int(delta["start_step"])))  # 单次最大调整±3

    # 限制参数最终范围
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


# ---------- 4. 智能参数更新（带历史记忆和回调限制） ----------
def smart_update_params(opt_styleid, delta, prev_adj, prev_problem_type):
    """
    智能更新参数，避免反向大幅回调
    :param opt_styleid: 当前StyleID参数
    :param delta: LLM给出的调整幅度
    :param prev_adj: 上一轮的调整记录
    :param prev_problem_type: 上一轮的问题类型
    :return: 更新后的参数、本轮调整记录
    """
    current_adj = {"gamma": 0.0, "T": 0.0, "start_step": 0.0}

    # 1. Gamma 更新（带回调限制）
    if prev_problem_type == "content_loss" and delta["gamma"] < 0:
        # 上一轮因内容丢失调大了gamma，本轮不能反向回调超过上一轮调整幅度的50%
        max_back_adj = prev_adj["gamma"] * 0.5
        delta["gamma"] = max(-max_back_adj, delta["gamma"])
        print(f">>> 限制gamma回调：上一轮调整+{prev_adj['gamma']}，本轮反向调整不超过-{max_back_adj}")
    elif prev_problem_type == "style_loss" and delta["gamma"] > 0:
        # 上一轮因风格不足调小了gamma，本轮不能反向回调超过上一轮调整幅度的50%
        max_back_adj = abs(prev_adj["gamma"]) * 0.5
        delta["gamma"] = min(max_back_adj, delta["gamma"])
        print(f">>> 限制gamma回调：上一轮调整{prev_adj['gamma']}，本轮反向调整不超过+{max_back_adj}")

    if not (delta["gamma"] < 0 and opt_styleid.gamma <= 0.1) and not (delta["gamma"] > 0 and opt_styleid.gamma >= 0.9):
        opt_styleid.gamma += delta["gamma"]
        current_adj["gamma"] = delta["gamma"]
    else:
        print(f">>> gamma已触达边界({opt_styleid.gamma})，本次不调整")

    # 2. T 更新（带回调限制）
    if prev_problem_type == "content_loss" and delta["T"] > 0:
        # 上一轮因内容丢失调小了T，本轮不能反向回调超过上一轮调整幅度的50%
        max_back_adj = abs(prev_adj["T"]) * 0.5
        delta["T"] = min(max_back_adj, delta["T"])
        print(f">>> 限制T回调：上一轮调整{prev_adj['T']}，本轮反向调整不超过+{max_back_adj}")
    elif prev_problem_type == "style_loss" and delta["T"] < 0:
        # 上一轮因风格不足调大了T，本轮不能反向回调超过上一轮调整幅度的50%
        max_back_adj = prev_adj["T"] * 0.5
        delta["T"] = max(-max_back_adj, delta["T"])
        print(f">>> 限制T回调：上一轮调整+{prev_adj['T']}，本轮反向调整不超过-{max_back_adj}")

    if not (delta["T"] < 0 and opt_styleid.T <= 1.0) and not (delta["T"] > 0 and opt_styleid.T >= 3.0):
        opt_styleid.T += delta["T"]
        current_adj["T"] = delta["T"]
    else:
        print(f">>> T已触达边界({opt_styleid.T})，本次不调整")

    # 3. start_step 更新（带回调限制）
    new_start_step = opt_styleid.start_step + delta["start_step"]
    if prev_problem_type == "content_loss" and delta["start_step"] < 0:
        # 上一轮因内容丢失调大了start_step，本轮不能反向回调超过上一轮调整幅度的50%
        max_back_adj = prev_adj["start_step"] * 0.5
        delta["start_step"] = max(-max_back_adj, delta["start_step"])
        new_start_step = opt_styleid.start_step + delta["start_step"]
        print(f">>> 限制start_step回调：上一轮调整+{prev_adj['start_step']}，本轮反向调整不超过-{max_back_adj}")
    elif prev_problem_type == "style_loss" and delta["start_step"] > 0:
        # 上一轮因风格不足调小了start_step，本轮不能反向回调超过上一轮调整幅度的50%
        max_back_adj = abs(prev_adj["start_step"]) * 0.5
        delta["start_step"] = min(max_back_adj, delta["start_step"])
        new_start_step = opt_styleid.start_step + delta["start_step"]
        print(f">>> 限制start_step回调：上一轮调整{prev_adj['start_step']}，本轮反向调整不超过+{max_back_adj}")

    if new_start_step >= 1 and new_start_step <= opt_styleid.ddim_steps - 1:
        opt_styleid.start_step = new_start_step
        current_adj["start_step"] = delta["start_step"]
    else:
        print(f">>> start_step已触达边界({opt_styleid.start_step})，本次不调整")

    # 兜底限制参数范围
    opt_styleid.gamma = max(0.1, min(0.9, opt_styleid.gamma))
    opt_styleid.T = max(1.0, min(3.0, opt_styleid.T))
    opt_styleid.start_step = max(1, min(opt_styleid.ddim_steps - 1, opt_styleid.start_step))

    return opt_styleid, current_adj


# ---------- 4. 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="StyleID + LLM 闭环风格迁移（智能防震荡版）")
    parser.add_argument("--content", default="test/content/000000000632.jpg")
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

        # 初始化历史记录
        prev_problem = ""
        prev_problem_type = ""
        prev_adj = {"gamma": 0.0, "T": 0.0, "start_step": 0.0}  # 上一轮调整记录

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

            # LLM评估结果（传入历史问题）
            print("LLM评估结果中...")
            try:
                judge = llm_judge_and_suggest(out_path, args.content, args.style, prev_problem)
                print(">>> 评估结果:", json.dumps(judge, indent=2, ensure_ascii=False))

                if judge["pass"]:
                    print("✅ LLM评估通过，结束闭环")
                    success = True
                    break

                # 记录本轮问题
                current_problem = judge["problem"]
                current_problem_type = judge["problem_type"]
                delta = judge["delta"]

                # 打印本轮信息
                print(f">>> 本轮问题：{current_problem}")
                print(f">>> 问题类型：{current_problem_type}")
                print(f">>> 调整幅度：gamma={delta['gamma']}、T={delta['T']}、start_step={delta['start_step']}")

                # 智能更新参数（带回调限制）
                opt_styleid, current_adj = smart_update_params(
                    opt_styleid, delta, prev_adj, prev_problem_type
                )

                # 更新历史记录
                prev_problem = current_problem
                prev_problem_type = current_problem_type
                prev_adj = current_adj

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