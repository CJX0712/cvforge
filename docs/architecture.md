# cvforge 架构文档

> 基于特征的图像分类系统 · 作者：晨星 · 版本：1.0.0
> 设计原则：**复用业界领先开源**（scikit-image / OpenCV / scikit-learn），**禁止从零自研**；按单一职责分模块，每模块可独立验证。

## 1. 系统定位

cvforge 是一套**可实际运行**的「特征提取 + 分类」图像识别系统：

- 内置 4 类可区分合成形状（circle / triangle / square / cross）作为基准数据。
- 4 个特征提取器：PixelFlatten（必选基线）+ HOG(scikit-image) + ORB/SIFT(OpenCV)。
- 5 个分类器：RandomForest / SVM / KNN（SOTA 监督）+ Majority / Random（弱基线）。
- 评测：有监督 accuracy / macro-F1 / balanced-accuracy + 无监督 KMeans 聚类对齐(NMI/ARI)。
- 可选后端（scikit-image / OpenCV）缺失时通过 `available()` 自动降级，**clone 后零下载即可跑通 demo**（纯 numpy + sklearn 的像素路径）。

## 2. 架构图

```mermaid
flowchart TD
    subgraph 数据层 data
        SYN[synthetic.make_synthetic<br/>纯 numpy 形状生成]
        LOAD[loader.ImageFolderSource / CSVDataSource<br/>PIL 载入(可选)]
        DS[(ImageDataset)]
    end

    subgraph 特征层 features
        PIX[PixelFlatten<br/>numpy 必选]
        HOG[HOGExtractor<br/>scikit-image]
        ORB[ORB/SIFT<br/>OpenCV]
    end

    subgraph 模型层 models
        SK[sklearn_classifiers<br/>RF/SVM/KNN]
        BASE[baselines<br/>Majority/Random]
    end

    subgraph 评测层 eval
        MET[metrics<br/>acc/F1/bal + NMI/ARI]
        BENCH[benchmark<br/>分层评估 + 聚合]
    end

    subgraph 编排 pipeline
        PIPE[cvforge_pipeline.CVForgePipeline<br/>default_extractors/default_classifiers/run]
        CLI[cli<br/>demo / evaluate / version]
    end

    SYN --> DS
    LOAD --> DS
    DS --> PIX & HOG & ORB
    PIX & HOG & ORB --> SK & BASE
    SK & BASE --> MET
    MET --> BENCH
    BENCH --> PIPE --> CLI
```

## 3. 模块划分与单一职责

| 模块 | 文件 | 职责 | 对外接口 |
|------|------|------|----------|
| 核心类型 | `core/types.py` | `ImageDataset`/`Features`/`Prediction`/`BenchmarkResult` | dataclass |
| 配置 | `core/config.py` | 全局超参集中管理，`CVFORGE_*` 环境变量覆盖 | `Config` / `from_env()` |
| 错误 | `core/errors.py` | 分层错误码 E1xx | 异常类 |
| 接口 | `core/interfaces.py` | `FeatureExtractor`/`Classifier`/`DataSource`/`Metric`/`Evaluator` Protocol | Protocol |
| 数据 | `data/synthetic.py` | 合成形状图像（纯 numpy，固定随机种子） | `make_synthetic()` |
| 数据 | `data/loader.py` | 图像目录 / CSV 载入（PIL 可选） | `load_image_folder()` / `load_csv()` |
| 特征 | `features/pixel.py` | 像素展平（必选基线） | `PixelFlattenExtractor` |
| 特征 | `features/hog.py` | HOG 纹理特征 | `HOGExtractor` |
| 特征 | `features/keypoint.py` | ORB / SIFT 关键点均值描述子 | `ORBExtractor` / `SIFTExtractor` |
| 模型 | `models/sklearn_classifiers.py` | 复用 sklearn 分类器 | `make_random_forest/svm/knn` |
| 模型 | `models/baselines.py` | Majority / Random 弱基线 | `MajorityBaseline` / `RandomBaseline` |
| 评测 | `eval/metrics.py` | 分类/聚类指标 | `accuracy/macro_f1/...` |
| 评测 | `eval/benchmark.py` | 分层评估 + 聚合 + 聚类对齐 | `benchmark()` / `evaluate_clustering()` |
| 编排 | `pipeline/cvforge_pipeline.py` | 端到端串联 | `CVForgePipeline.run()` |
| 入口 | `cli.py` | 命令行 | `demo/evaluate/version` |

## 4. 接口契约（Protocol，解耦关键）

所有特征提取器实现统一 `FeatureExtractor` 协议，分类器实现 `Classifier` 协议，便于独立测试与替换：

```python
@runtime_checkable
class FeatureExtractor(Protocol):
    name: str
    def available(self) -> bool: ...            # 可选后端离线降级
    def extract(self, images: ImageDataset) -> Features: ...

@runtime_checkable
class Classifier(Protocol):
    name: str
    def available(self) -> bool: ...
    def fit(self, X, y) -> "Classifier": ...
    def predict(self, X) -> Prediction: ...
```

## 5. 选型依据（性能 / 生态 / 许可证 / 维护活跃度）

| 复用对象 | 用途 | 许可证 | 维护 | 不选替代理由 |
|----------|------|--------|------|--------------|
| scikit-image（HOG） | 梯度方向直方图纹理特征 | BSD-3 | 活跃 | 纯 numpy/scipy，pip wheel，无权重；工业标准 |
| OpenCV（ORB/SIFT） | 关键点局部特征 | Apache-2 | 活跃 | 业界标准 CV 库，headless wheel，无权重 |
| scikit-learn（RF/SVM/KNN） | 监督分类 | BSD-3 | 活跃 | 已为基石，零额外体积，稳定 |
| Pillow | 图像 IO（可选） | MIT | 活跃 | 标准图像库 |
| 自研 | — | — | — | **禁止**：特征与分类均复用 |

> 关键纪律：**所有"模型/特征"均为复用或教科书基线**，无任何从零自研算法；指标公式为评测基础设施，符合"不重复造轮子"。

## 6. 评测方法论

- **有监督**：分层 train/test 切分（`test_size=0.30`），accuracy / macro-F1 / balanced-accuracy。
- **无监督补充**：KMeans 聚类对齐真实标签，报告 NMI / ARI / AMI（衡量特征本身的可分性）。
- **弱基线对标**：Majority（训练集众数）/ Random（随机），量化 SOTA 特征相对增益。

## 7. 默认基准基线（本机实测，复现一致）

合成数据（4 类形状，每类 40，32×32，固定 random_state=42）：

| model | ACC | F1 | BAL_ACC |
|-------|-----|-----|---------|
| **HOG(skimage)@RandomForest** | **0.9792** | **0.9791** | **0.9792** |
| **HOG(skimage)@SVM(RBF)** | **0.9792** | **0.9791** | **0.9792** |
| HOG(skimage)@KNN | 0.9375 | 0.9386 | 0.9375 |
| **SIFT(cv2)@RandomForest** | **0.9167** | **0.9177** | **0.9167** |
| SIFT(cv2)@SVM(RBF) | 0.9167 | 0.9181 | 0.9167 |
| **ORB(cv2)@SVM(RBF)** | **0.8542** | **0.8523** | **0.8542** |
| ORB(cv2)@RandomForest | 0.8125 | 0.8102 | 0.8125 |
| PixelFlatten@RandomForest | 0.6875 | 0.6721 | 0.6875 |
| PixelFlatten@SVM(RBF) | 0.6250 | 0.6304 | 0.6250 |
| PixelFlatten@KNN | 0.5625 | 0.5574 | 0.5625 |
| Majority（基线） | 0.2500 | 0.1000 | 0.2500 |
| Random（基线） | 0.1875 | 0.1871 | 0.1875 |

无监督对齐：**HOG 特征 KMeans vs 真实标签 → NMI≈0.8 / ARI≈0.8**（像素特征仅 ≈0.25，印证 HOG 编码了结构）。

**结论**：HOG/SIFT 显著超越原始像素与弱基线（4 类随机下限 ≈0.19、多数类 ≈0.25），HOG+RF/SVM 达 ≈0.98，验证「复用 SOTA 特征 → 高性能」。

## 8. 复现与验证

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
.venv/bin/python -m pytest tests/ -q -W ignore::UserWarning   # 32 passed
.venv/bin/python -m cvforge.cli demo                          # -> benchmark.json
```

详见 `README.md`。
