# -*- coding: utf-8 -*-
# 文件名: core/airpls_core.py
# 描述: AirPLS 基线校正核心模块
# (来自您的上传)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve


# --- 1. 核心 AirPLS 算法 ---

def airpls(y, lambda_=1000, max_iter=15):
    """
    Asymmetric Least Squares (AirPLS) 核心算法
    ... (函数内容与您提供的完全一致)
    """
    m = y.shape[0]
    w = np.ones(m)

    # 创建二阶差分矩阵 D
    D = eye(m, format='csc')
    D = D[2:, :] - 2 * D[1:-1, :] + D[:-2, :]
    D = D.T @ D

    # 预计算 P = lambda * D
    P = lambda_ * D

    for i in range(max_iter):
        W = diags(w, 0, shape=(m, m))
        C = W + P
        z = spsolve(C, w * y)
        d = y - z

        d_neg = d[d < 0]
        if len(d_neg) < 1:
            break

        std_neg = np.std(d_neg)
        if std_neg < 1e-6:
            break

        # 权重更新规则
        new_w = 1 / (1 + np.exp(2 * (d / (2 * std_neg))))

        if np.linalg.norm(new_w - w) / np.linalg.norm(w) < 1e-3:
            break

        w = new_w

    return z


# --- 2. GUI 可调用的高级 API 函数 ---

def smooth_spectrum(y_data, window_length=13, polyorder=3):
    """
    (可选) 对光谱进行 Savitzky-Golay 平滑。
    ... (函数内容与您提供的完全一致)
    """
    if window_length % 2 == 0:
        window_length += 1  # 确保为奇数
    return savgol_filter(y_data, window_length, polyorder)


def process_spectrum(y_data, lambda_, max_iter=15):
    """
    【GUI 主要调用函数】
    对单个光谱运行 AirPLS 并返回基线和校正后的光谱。
    ... (函数内容与您提供的完全一致)
    """
    if not isinstance(y_data, np.ndarray):
        y_data = np.array(y_data)

    baseline = airpls(y_data, lambda_=lambda_, max_iter=max_iter)
    corrected_spectrum = y_data - baseline

    return baseline, corrected_spectrum


# --- 3. GUI 可调用的辅助工具 (保存和绘图) ---
# (这些函数 (save_results, plot_results) 暂时不会被GUI调用，
# 因为GUI有自己的保存和绘图逻辑，但我们保留它们)
...