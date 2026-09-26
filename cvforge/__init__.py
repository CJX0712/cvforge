"""cvforge — 基于特征的图像分类系统（作者: 晨星）。

复用 scikit-image(HOG) + OpenCV(ORB/SIFT, 可选) + scikit-learn(分类/聚类) 等领先开源，
无自研模型、无预训练权重下载；可选后端缺失时通过 available() 自动降级，
保证 clone 后零下载即可跑通 demo。
"""

__version__ = "1.0.0"
__author__ = "晨星"
