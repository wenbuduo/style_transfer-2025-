import os
import re
import argparse
from PIL import Image, ImageOps

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def is_img(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)


def safe_open(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def pad_resize(img: Image.Image, size: int) -> Image.Image:
    # 等比缩放 + padding 到 size x size
    return ImageOps.pad(img, (size, size), method=Image.Resampling.LANCZOS, color=(255, 255, 255), centering=(0.5, 0.5))


def parse_output_name(filename: str):
    """
    解析输出文件名：content__style__tag.png
    返回 (content, style, tag) 或 None
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    content = parts[0]
    style = parts[1]
    tag = "__".join(parts[2:])  # 防止 tag 里还有 __
    return content, style, tag


def scan_outputs(output_dir: str):
    """
    扫描 output_dir 下所有图片，按 (content, style) 分组，并记录不同 tag 的路径
    groups[(content, style)][tag] = path
    """
    groups = {}
    for fn in os.listdir(output_dir):
        if not is_img(fn):
            continue
        parsed = parse_output_name(fn)
        if parsed is None:
            continue
        content, style, tag = parsed
        key = (content, style)
        groups.setdefault(key, {})
        groups[key][tag] = os.path.join(output_dir, fn)
    return groups


def list_images_in_dir(dir_path: str):
    """
    返回 dict: {stem: path}
    stem 是不含扩展名的文件名
    """
    m = {}
    if not dir_path or not os.path.isdir(dir_path):
        return m
    for root, _, files in os.walk(dir_path):
        for fn in files:
            if not is_img(fn):
                continue
            path = os.path.join(root, fn)
            stem = os.path.splitext(fn)[0]
            m[stem] = path
    return m


def make_canvas(cols, tile, gap, bg=(255, 255, 255)):
    """
    cols: List[List[PIL.Image]]  每个内层 list 是该列的多行图片
    会自动把每张图 pad_resize 到 tile x tile
    输出一张拼接后的大图
    """
    # 每列行数对齐（用空白图补齐）
    max_rows = max((len(c) for c in cols), default=0)
    blank = Image.new("RGB", (tile, tile), bg)
    norm_cols = []
    for c in cols:
        c2 = [pad_resize(img, tile) for img in c]
        if len(c2) < max_rows:
            c2.extend([blank] * (max_rows - len(c2)))
        norm_cols.append(c2)

    ncols = len(norm_cols)
    if ncols == 0 or max_rows == 0:
        return None

    W = ncols * tile + (ncols - 1) * gap
    H = max_rows * tile + (max_rows - 1) * gap
    canvas = Image.new("RGB", (W, H), bg)

    for ci, col in enumerate(norm_cols):
        for ri, img in enumerate(col):
            x = ci * (tile + gap)
            y = ri * (tile + gap)
            canvas.paste(img, (x, y))
    return canvas


def merge_pair_mode(groups, content_map, style_map, out_dir, tile, gap, include_style_ref):
    """
    每个 (content, style) 生成一张图：
    左列：content 原图（可选 style 参考图放在下面）
    右列：该对下的所有 tag 输出（按 tag 排序）逐行往下排
    """
    os.makedirs(out_dir, exist_ok=True)

    for (content, style), tag2path in sorted(groups.items()):
        if content not in content_map:
            print(f"[SKIP] 找不到 content 原图: {content}")
            continue

        left_imgs = [safe_open(content_map[content])]
        if include_style_ref:
            if style in style_map:
                left_imgs.append(safe_open(style_map[style]))
            else:
                print(f"[WARN] 找不到 style 参考图: {style}（继续）")

        # 右列：各种输出（baseline / improved / ...）
        right_imgs = []
        for tag in sorted(tag2path.keys()):
            right_imgs.append(safe_open(tag2path[tag]))

        canvas = make_canvas([left_imgs, right_imgs], tile=tile, gap=gap)
        if canvas is None:
            continue

        out_name = f"{content}__{style}__merged.png"
        out_path = os.path.join(out_dir, out_name)
        canvas.save(out_path)
        print(f"[OK] {out_path}")


def merge_content_mode(groups, content_map, out_dir, tile, gap, variant, max_styles):
    """
    每个 content 生成一张图：
    第一列：content 原图
    后续每列：不同 style 的某个 variant（默认 improved）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 反向索引：content -> [(style, tag2path)]
    by_content = {}
    for (content, style), tag2path in groups.items():
        by_content.setdefault(content, [])
        by_content[content].append((style, tag2path))

    for content, items in sorted(by_content.items()):
        if content not in content_map:
            print(f"[SKIP] 找不到 content 原图: {content}")
            continue

        cols = []
        cols.append([safe_open(content_map[content])])  # 第一列：原图

        # 其他列：每个 style 一列
        styles_added = 0
        for style, tag2path in sorted(items, key=lambda x: x[0]):
            if variant not in tag2path:
                continue
            cols.append([safe_open(tag2path[variant])])
            styles_added += 1
            if max_styles > 0 and styles_added >= max_styles:
                break

        canvas = make_canvas(cols, tile=tile, gap=gap)
        if canvas is None:
            continue
        out_name = f"{content}__{variant}__improved.png"
        out_path = os.path.join(out_dir, out_name)
        canvas.save(out_path)
        print(f"[OK] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True, help="内容原图目录（用于左侧原图列）")
    ap.add_argument("--style_dir", default="", help="风格原图目录（pair 模式可选放左列第二行）")
    ap.add_argument("--output_dir", required=True, help="你生成的结果图目录（含 content__style__tag.png）")
    ap.add_argument("--save_dir", default="merged", help="合成图输出目录")

    ap.add_argument("--mode", choices=["pair", "content"], default="pair",
                    help="pair: 每个 content__style 一张（左原图列 + 右各种输出列）；content: 每个 content 一张（原图列 + 各style列）")
    ap.add_argument("--variant", default="improved", help="content 模式时选择用哪个 tag（比如 improved/baseline）")
    ap.add_argument("--max_styles", type=int, default=10, help="content 模式最多拼多少个 style（<=0 不限制）")

    ap.add_argument("--tile", type=int, default=256, help="每格大小（正方形）")
    ap.add_argument("--gap", type=int, default=12, help="格子间距")
    ap.add_argument("--include_style_ref", action="store_true", help="pair 模式时左列额外附上 style 参考图")

    args = ap.parse_args()

    groups = scan_outputs(args.output_dir)
    if not groups:
        print("[ERR] 没在 output_dir 里找到符合 content__style__tag.* 的图片")
        return

    content_map = list_images_in_dir(args.content_dir)
    style_map = list_images_in_dir(args.style_dir) if args.style_dir else {}

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "pair":
        merge_pair_mode(groups, content_map, style_map, args.save_dir, args.tile, args.gap, args.include_style_ref)
    else:
        merge_content_mode(groups, content_map, args.save_dir, args.tile, args.gap, args.variant, args.max_styles)


if __name__ == "__main__":
    main()
