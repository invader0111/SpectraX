# -*- coding: utf-8 -*-
# 文件名: core/msc_core.py
# 描述: Multiplicative Scatter Correction (MSC) 核心模块 (已修改 API)

import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_msc(spectra_data, reference_spectrum=None):
    """
    对一组光谱数据执行 MSC 校正。
    (此核心函数不变)
    """
    input_is_1d = False
    if spectra_data.ndim == 1:
        input_is_1d = True
        spectra_data = spectra_data.reshape(1, -1)
    n_samples, n_features = spectra_data.shape
    if reference_spectrum is None:
        mean_spectrum = np.mean(spectra_data, axis=0)
    else:
        if reference_spectrum.shape[0] != n_features:
            raise ValueError("参考光谱的特征数与输入光谱不匹配。")
        mean_spectrum = reference_spectrum
    corrected_spectra = np.zeros_like(spectra_data)
    for i in range(n_samples):
        spectrum = spectra_data[i, :]
        X_ref = mean_spectrum.reshape(-1, 1)
        y_spec = spectrum.reshape(-1, 1)
        try:
            reg = LinearRegression().fit(X_ref, y_spec)
            slope = reg.coef_[0][0]
            intercept = reg.intercept_[0]
            if np.abs(slope) < 1e-6:
                print(f"警告: 样本 {i} 的 MSC 斜率接近于零，校正可能无效。")
                corrected_spectra[i, :] = spectrum
            else:
                corrected_spectra[i, :] = (spectrum - intercept) / slope
        except Exception as e:
            print(f"错误: 样本 {i} 的 MSC 回归失败: {e}")
            corrected_spectra[i, :] = spectrum
    if input_is_1d:
        return corrected_spectra.flatten(), mean_spectrum
    else:
        return corrected_spectra, mean_spectrum


# --- VVVV 修改: 简化 GUI API 函数 VVVV ---
def process_spectrum_msc(y_data, reference_spectrum):
    """
    【GUI 主要调用函数 - MSC v2】
    对单个光谱运行 MSC (使用 *已确定* 的参考光谱)。

    参数:
    - y_data: 原始光谱 (1D numpy 数组)
    - reference_spectrum: 用作参考的光谱 (1D numpy 数组)

    返回:
    - corrected_spectrum: 校正后的光谱 (1D numpy 数组)
    - reference_spectrum: 传入的参考光谱 (1D numpy 数组) (为了绘图方便传回)
    - error_message: 如果发生错误，则返回错误消息字符串，否则为 None
    """
    try:
        if not isinstance(y_data, np.ndarray):
            y_data = np.array(y_data)
        if not isinstance(reference_spectrum, np.ndarray):
             reference_spectrum = np.array(reference_spectrum)

        if y_data.shape != reference_spectrum.shape:
            return y_data, reference_spectrum, "错误: 输入光谱与参考光谱形状不匹配。"

        # 直接调用核心计算函数
        corrected_spectrum, _ = calculate_msc(y_data, reference_spectrum=reference_spectrum)
        return corrected_spectrum, reference_spectrum, None

    except Exception as e:
        error_msg = f"MSC 处理失败: {e}"
        print(f"【MSC 模块错误】: {error_msg}")
        return y_data, reference_spectrum, error_msg # 返回原始数据和错误消息
# --- ^^^^ 修改结束 ^^^^ ---