import matplotlib
matplotlib.use("Agg")  # ★ 必须放在 import pyplot 之前
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === 让 matplotlib 支持中文显示 ===
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体 (Windows 常见)
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题


results = {
    "Gatys": {
        "psnr": 11.56,
        "lpips_c": 0.6412,
        "lpips_s": 1.8201
    },
    "Fast-ST": {
        "psnr": 9.25,
        "lpips_c": 0.7572,
        "lpips_s": 1.6245
    },
    "AdaIN": {
        "psnr": 11.56,
        "lpips_c": 0.6412,
        "lpips_s": 1.2201
    },
    "StyleID": {
        "psnr": 27.19,
        "lpips_c": 0.2987,
        "lpips_s": 0.8368
    },
}

plt.figure(figsize=(6, 5))
for name, m in results.items():
    plt.scatter(m["lpips_s"], m["psnr"], label=name, s=100)
    plt.text(m["lpips_s"] + 0.005, m["psnr"] + 0.2, name, fontsize=9)

plt.xlabel("LPIPS_style (越低越好)")
plt.ylabel("PSNR_content (越高越好)")
plt.title("内容保真度与风格相似性关系 (PSNR vs LPIPS_s)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("tradeoff_psnr_lpips_s.png", dpi=300)
# 注意：用 Agg 后不要 plt.show() 也没关系，重点是保存
