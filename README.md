# cvforge

> 基于特征的图像分类系统 · 作者：**晨星** · v1.0.0
> 复用 scikit-image(OpenCV)/scikit-learn 等领先开源，零自研模型，干净环境一键复现。

cvforge 是一套**可实际运行**的「特征提取 + 分类」图像识别系统：内置合成形状基准数据，组合 4 个特征提取器 × 5 个分类器，按 M4 式分层评测输出准确率/F1，并附 KMeans 无监督对齐。**可选后端缺失自动降级，clone 后零下载即可跑通 demo**。

## 特性

- ✅ **复用优先**：HOG（scikit-image）、ORB/SIFT（OpenCV）、RF/SVM/KNN（scikit-learn），无自研算法。
- ✅ **严格评测**：accuracy / macro-F1 / balanced-accuracy + KMeans 聚类对齐(NMI/ARI)。
- ✅ **离线降级**：scikit-image / OpenCV 缺失 → `available()=False` 自动跳过，纯 numpy/sklearn 像素路径保证 demo 可跑。
- ✅ **可独立验证**：每模块单测 + 最小可运行示例；32 个 pytest 用例全绿。
- ✅ **依赖锁定**：`requirements.lock.txt`（pip freeze）保证可复现。
- ✅ **一键复现**：`make` / `Dockerfile` 双通道。

## 快速开始

```bash
# 1) 创建隔离环境并安装锁定依赖
python -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.lock.txt

# 2) 跑端到端基准（默认合成数据）→ 生成 benchmark.json
.venv/bin/python -m cvforge.cli demo

# 3) 查看版本
.venv/bin/python -m cvforge.cli version
```

### 对指定特征+分类器做评测

```bash
# HOG 特征 + 随机森林
.venv/bin/python -m cvforge.cli evaluate --feature hog --model rf
# 可选: --feature pixel/hog/orb/sift ; --model rf/svm/knn/majority/random
```

## 项目结构

```
cvforge/
├── cvforge/
│   ├── core/            # 类型 / 配置 / 错误 / 接口契约
│   ├── data/            # 合成形状生成 + 图像目录/CSV 载入
│   ├── features/        # PixelFlatten / HOG / ORB / SIFT
│   ├── models/          # sklearn 分类器 + 弱基线
│   ├── eval/            # 分类/聚类指标 + 基准编排
│   ├── pipeline/        # 端到端 CVForgePipeline
│   ├── examples/        # run_demo.py 最小可运行示例
│   └── cli.py           # 命令行入口
├── tests/               # 32 个 pytest 用例
├── docs/architecture.md # 架构图 + 选型依据 + 基线表
├── requirements.txt     # 语义化依赖范围
├── requirements.lock.txt# pip freeze 全量锁定
├── Dockerfile / Makefile / .gitignore
```

## 特征与分类器一览

| 特征 | 后端 | 说明 |
|------|------|------|
| PixelFlatten | numpy | 像素展平（必选弱基线） |
| HOG | scikit-image | 梯度方向直方图（**综合最优**） |
| ORB | OpenCV | 关键点均值描述子 |
| SIFT | OpenCV | 关键点均值描述子 |

| 分类器 | 后端 | 说明 |
|--------|------|------|
| RandomForest | scikit-learn | 集成树（SOTA） |
| SVM(RBF) | scikit-learn | RBF 核支持向量机（SOTA） |
| KNN | scikit-learn | K 近邻 |
| Majority / Random | — | 弱基线（量化增益） |

## 评测与验收

- **指标**：accuracy / macro-F1 / balanced-accuracy（有监督）+ NMI/ARI（无监督对齐）。
- **切分**：分层抽样 `test_size=0.30`，固定 `random_state` 可复现。
- **默认基线**（本机实测）：HOG+RF/SVM **ACC 0.979**，SIFT **0.917**，ORB+SVM **0.854**，均远高于随机(0.19)/多数类(0.25)；像素+RF 仅 0.69（印证 SOTA 特征增益）。
- **验收（DoD）**：`git clone` → 一键脚本 → demo 跑通零干预；pytest 全过；依赖锁定可复现；文档覆盖架构/部署/使用三部分。

> 注：ORB 在小尺寸平滑合成图上默认 FAST 阈值过高会漏掉关键点，已放宽 `fastThreshold/edgeThreshold` 修复；在真实自然图像上表现更佳。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q -W ignore::UserWarning
# 32 passed
```

## 许可证

代码以 MIT 许可发布；所复用上游（scikit-image BSD-3 / OpenCV Apache-2 / scikit-learn BSD-3 / Pillow MIT）遵循各自许可证。
