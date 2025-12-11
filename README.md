# 神经风格迁移课程设计项目

## 项目概述
实现基于StyleID (CVPR 2024)的图像风格迁移，并与三个baseline模型进行对比。

## 模型列表
1. **StyleID** (主模型) - CVPR 2024 Highlight，基于扩散模型的免训练风格迁移
2. **Gatys Style Transfer** (Baseline 1) - 经典优化方法
3. **Fast Style Transfer** (Baseline 2) - 前馈网络快速风格迁移
4. **AdaIN** (Baseline 3) - 自适应实例归一化实时风格迁移

## 数据集

### 轻量级方案 (推荐) ⭐
- **自动生成**: 0MB, 立即可用 (方案1)
- **COCO样本**: 50张, 约50MB (方案2)
- **详见**: `MINI_DATASET_GUIDE.md`

### 完整数据集 (可选)
- **内容图**: COCO2017 (~20GB)
- **风格图**: WikiArt (~30GB)

**💡 建议**: 使用轻量级方案足够完成课程设计!

## 评估指标
- 内容损失 (Content Loss)
- 风格损失 (Style Loss)
- PSNR (峰值信噪比)
- SSIM (结构相似度)

## 项目结构
```
neural_style_transfer/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── setup.sh                     # 环境设置脚本
├── download_data.sh             # 数据下载脚本
├── models/                      # 模型实现
│   ├── gatys.py                # Gatys模型
│   ├── fast_style_transfer.py  # Fast Style Transfer
│   ├── adain.py                # AdaIN模型
│   └── styleid_wrapper.py      # StyleID封装
├── utils/                       # 工具函数
│   ├── data_loader.py          # 数据加载
│   ├── losses.py               # 损失函数
│   └── metrics.py              # 评估指标
├── train_baselines.py          # 训练baseline模型
├── inference.py                # 推理脚本
├── evaluate.py                 # 评估脚本
└── compare_results.py          # 结果对比

## 运行步骤

### 1. 环境配置
```bash
bash setup.sh
```

### 2. 下载数据
```bash
bash download_data.sh
```

### 3. 训练baseline模型
```bash
# 训练Gatys (无需训练，直接优化)
# 训练Fast Style Transfer
python train_baselines.py --model fast_style_transfer --epochs 2

# 训练AdaIN
python train_baselines.py --model adain --epochs 5
```

### 4. 运行风格迁移
```bash
# 使用Gatys
python inference.py --model gatys --content data/content/sample.jpg --style data/style/sample.jpg

# 使用Fast Style Transfer
python inference.py --model fast_style_transfer --content data/content/sample.jpg --style data/style/sample.jpg

# 使用AdaIN
python inference.py --model adain --content data/content/sample.jpg --style data/style/sample.jpg

# 使用StyleID (需要先设置StyleID环境)
cd StyleID
python run_styleid.py --cnt ../data/content --sty ../data/style
```

### 5. 评估和对比
```bash
python evaluate.py --results_dir results/
python compare_results.py --results_dir results/
```

## 系统要求
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.1+ (推荐GPU: 20GB+ VRAM for StyleID)
- 至少50GB磁盘空间

## 注意事项
1. StyleID需要下载Stable Diffusion权重 (~4GB)
2. COCO2017数据集较大 (~20GB)
3. WikiArt数据集 (~30GB)
4. 建议先用小数据集测试

## 参考文献
1. StyleID: Chung et al., CVPR 2024
2. Gatys: "A Neural Algorithm of Artistic Style", 2015
3. Fast Style Transfer: Johnson et al., ECCV 2016
4. AdaIN: Huang et al., ICCV 2017
