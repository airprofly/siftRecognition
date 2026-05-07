<div align="center">

# 🔧 Harris 角点检测 · SIFT 特征 · 场景识别
### Harris Corner Detection, SIFT Feature Extraction & Scene Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9+-orange.svg)](https://pytorch.org/) [![NumPy 2.3](https://img.shields.io/badge/NumPy-2.3+-green.svg)](https://numpy.org/)

计算机视觉 · Harris 角点检测 · SIFT 特征 · 场景识别 · PyTorch

</div>

---

## 📖 项目简介

基于 PyTorch 的计算机视觉课程项目，实现三大核心视觉管线：

1. **Harris 角点检测** — 完全基于 PyTorch 自定义层实现
2. **SIFT 特征提取** — 从零实现 SIFT 描述子计算的全流程
3. **场景识别** — 三种分类管线（Tiny Image + k-NN / Bag of SIFT + k-NN / Bag of SIFT + SVM）

## 📌 功能特性

- ✅ **HarrisNet** — 纯 PyTorch `nn.Sequential` 构建 Harris 角点检测器
- ✅ **SIFTNet** — PyTorch 自定义层实现 SIFT 特征描述子
- ✅ **特征匹配** — 基于欧氏距离的最近邻匹配与 Lowe 比率测试
- ✅ **k-means 聚类** — 手动实现 k-means++ 初始化的聚类算法
- ✅ **场景识别** — 三条完整管线（Tiny Image / BoF + k-NN / BoF + SVM）
- ✅ **特征缓存** — 自动缓存特征与词表到本地，避免重复计算
- ✅ **混淆矩阵** — 自动生成分类结果可视化

## 📁 项目结构

<details>
<summary><b>查看目录结构</b></summary>

```text
project/
├── configs/                  # 🔧 配置系统
│   ├── app_config.py         # AppConfig / SceneConfig / SceneRecConfig
│   ├── app_config.yml        # YAML 配置文件
│   ├── logger_config.py      # loguru 日志配置
│   ├── plt_config.py         # matplotlib 交互配置
│   └── __init__.py           # 全局 APP_CONFIG 单例
├── datasets/
│   └── scenedataset.py       # SceneDataset（按类别子目录组织）
├── models/                   # 🧠 模型实现
│   ├── harrisNet.py          # HarrisNet — Harris 角点检测器
│   ├── siftNet.py            # SIFTNet — SIFT 描述子计算
│   ├── feature_match.py      # 特征匹配（距离计算 + 比率测试）
│   ├── kmeans.py             # k-means 聚类与视觉词表构建
│   ├── recognition.py        # 场景识别核心（Tiny Image / BoF / k-NN）
│   └── svm.py                # LinearSVM 分类器
├── utils/
│   ├── utils.py              # 图像 I/O、可视化、评估
│   └── dataloader.py         # Tiny Image 数据加载器
├── tests/                    # 📝 单元测试（pytest）
│   ├── test_harris.py        # HarrisNet 各层测试
│   ├── test_sift.py          # SIFT 组件测试
│   ├── test_feature_match.py # 特征匹配测试
│   ├── test_recognition.py   # 场景识别完整测试
│   └── conftest.py           # sys.path 配置
├── schemas/
│   └── appConfig.schema.json # YAML 配置 JSON Schema
├── run_scene_recognition.py  # 🚀 场景识别主入口
├── environment.yml           # 💾 Conda 环境配置
└── requirements.txt          # 💾 pip 依赖
```

</details>

## 🔧 环境配置

<details>
<summary><b>查看环境配置</b></summary>

### 前置要求
- 安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 创建虚拟环境

```bash
# 使用 Conda（推荐）
conda env create -f environment.yml
conda activate pytorch

# 或使用 pip
pip install -r requirements.txt
```

</details>

## 🚀 快速开始

### 场景识别

```bash
conda activate pytorch
python run_scene_recognition.py
```

按顺序执行三条识别管线，并输出混淆矩阵可视化结果。

### 运行测试

```bash
# 运行全部测试
python -m pytest tests/

# 测试单个模块
python -m pytest tests/test_harris.py -v
python -m pytest tests/test_sift.py -v
python -m pytest tests/test_recognition.py -v
```

### 配置文件

项目配置通过 `configs/app_config.yml` 管理，支持运行时修改：

```yaml
scene_rec:
  vocab_size: 200        # 视觉词表大小
  k_tiny_images: 3       # Tiny Image 管线 k-NN k 值
  k_bag_of_sifts: 15     # BoF 管线 k-NN k 值
  svm_c: 0.1             # SVM 正则化参数
  num_per_cat: 100       # 每类训练样本数
  stride: 20             # 密集 SIFT 采样步长
```

### 特征缓存

场景识别管线自动缓存中间结果（视觉词表、特征矩阵）到 `data/` 目录，首次计算后自动复用。

## 🧠 核心算法

### Harris 角点检测

$$
R = \det(M) - \alpha \cdot \text{trace}(M)^2
$$

其中 $M = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix}$，$S$ 为高斯加权后的梯度二阶矩矩阵。

- **实现**：基于 PyTorch `nn.Sequential` 堆叠自定义层（`ImageGradientsLayer` → `ChannelProductLayer` → `GaussianSmoothingLayer` → `HarrisResponseLayer`）
- **输入**：任意尺寸灰度图像（N, 1, H, W）
- **输出**：角点响应图 + 角点坐标

### SIFT 特征

从零实现 SIFT 描述子计算，包含：

- **主方向估计**：梯度方向直方图加权投票
- **描述子生成**：$4 \times 4$ 子区域 × 8 方向 = 128 维向量
- **三线性插值**：相邻直方图 bin 的平滑分配

### 场景识别

三条对比管线：

| 管线 | 特征 | 分类器 | 特点 |
|------|------|--------|------|
| **Tiny Image** | 16×16 灰度缩略图（256维） | k-NN (k=3) | 快速基线 |
| **BoF + k-NN** | 密集 SIFT + k-means 词袋直方图 | k-NN (k=15) | 中层特征 |
| **BoF + SVM** | 密集 SIFT + k-means 词袋直方图 | LinearSVM (C=0.1) | 最强分类器 |

数据集：[15 类场景数据集](http://people.csail.mit.edu/torralba/code/spatialenvelope/)，包含 Bedroom、Coast、Forest、Highway、Office 等常见室内外场景。

### k-means 聚类

- **初始化**：k-means++（加权距离采样）
- **迭代**：基于广播的欧氏距离计算，支持收敛自动停止
- **应用**：视觉词表构建 + SIFT 描述子量化

## 📄 许可证

本项目采用 [MIT 许可证](https://opensource.org/licenses/MIT)。

## 📚 参考文献

1. C. Harris and M. Stephens — [*A Combined Corner and Edge Detector*](https://doi.org/10.5244/C.2.23), In *Proc. of Fourth Alvey Vision Conference*, 1988.
2. D. G. Lowe — [*Distinctive Image Features from Scale-Invariant Keypoints*](https://doi.org/10.1023/B:VISI.0000029664.99615.94), *International Journal of Computer Vision*, 60(2):91–110, 2004.
3. A. Torralba, R. Fergus, W. T. Freeman — [*80 Million Tiny Images: A Large Dataset for Non-Parametric Object and Scene Recognition*](https://doi.org/10.1109/TPAMI.2008.128), *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(11):1958–1970, 2008.
4. J. Sivic and A. Zisserman — [*Video Google: A Text Retrieval Approach to Object Matching in Videos*](https://doi.org/10.1109/ICCV.2003.1238663), In *Proc. ICCV*, 2003.
5. [15 Scene Category Dataset](https://people.csail.mit.edu/torralba/code/spatialenvelope/) — MIT CSAIL Spatial Envelope
