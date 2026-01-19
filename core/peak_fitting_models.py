# -*- coding: utf-8 -*-
# 文件名: core/peak_fitting_models.py
# 描述: 定义峰拟合的数学模型函数

import numpy as np
from scipy.special import wofz

# --- 高斯模型 ---

def gaussian(x, amplitude, center, sigma):
    """单个高斯函数"""
    # 添加保护，防止 sigma 过小导致溢出
    sigma = max(sigma, 1e-6)
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

def multi_gaussian(x, *params):
    """多个高斯峰的叠加"""
    y_peaks = np.zeros_like(x, dtype=float)
    num_params_per_peak = 3
    if len(params) % num_params_per_peak != 0:
        raise ValueError(f"参数数量 ({len(params)}) 必须是 {num_params_per_peak} 的倍数。")

    for i in range(0, len(params), num_params_per_peak):
        amp, cen, sig = params[i : i + num_params_per_peak]
        # 仅在振幅为正时添加峰
        if amp > 0:
            y_peaks += gaussian(x, amp, cen, sig)
    return y_peaks

# --- Voigt 模型 ---

def voigt(x, amplitude, center, sigma, gamma):
    """单个Voigt峰型函数"""
    # 添加保护
    sigma = max(sigma, 1e-6)
    gamma = max(gamma, 1e-6)

    # Faddeeva 函数 (wofz) 计算 Voigt 线型
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    # wofz(z) = exp(-z^2) * erfc(-i*z)
    V = wofz(z).real / (sigma * np.sqrt(2.0 * np.pi))

    # 计算峰顶处的 Voigt 值以进行归一化 (确保 amplitude 是峰高)
    # 峰顶在 x = center, 此时 z_peak = (i*gamma) / (sigma * sqrt(2))
    z_peak = (1j * gamma) / (sigma * np.sqrt(2.0))
    V_peak = wofz(z_peak).real / (sigma * np.sqrt(2.0 * np.pi))

    # 避免除零
    if V_peak > 1e-9:
        return amplitude * (V / V_peak)
    else:
        # 如果峰顶值接近零（可能因为 sigma 或 gamma 极大），返回零
        return np.zeros_like(x)

def multi_voigt(x, *params):
    """多个Voigt峰的叠加"""
    y_peaks = np.zeros_like(x, dtype=float)
    num_params_per_peak = 4
    if len(params) % num_params_per_peak != 0:
        raise ValueError(f"参数数量 ({len(params)}) 必须是 {num_params_per_peak} 的倍数。")

    for i in range(0, len(params), num_params_per_peak):
        amp, cen, sig, gam = params[i : i + num_params_per_peak]
        # 仅在振幅为正时添加峰
        if amp > 0:
            y_peaks += voigt(x, amp, cen, sig, gam)
    return y_peaks

# --- 辅助函数，用于获取模型 ---
def get_fitting_function(peak_shape):
    """根据名称返回多峰拟合函数"""
    shape = peak_shape.lower()
    if shape == 'voigt':
        return multi_voigt
    elif shape == 'gaussian':
        return multi_gaussian
    else:
        raise ValueError(f"未知的峰形: '{peak_shape}'")

def get_single_peak_function(peak_shape):
    """根据名称返回单峰函数"""
    shape = peak_shape.lower()
    if shape == 'voigt':
        return voigt
    elif shape == 'gaussian':
        return gaussian
    else:
        raise ValueError(f"未知的峰形: '{peak_shape}'")

def get_params_per_peak(peak_shape):
    """根据名称返回每个峰的参数数量"""
    shape = peak_shape.lower()
    if shape == 'voigt':
        return 4
    elif shape == 'gaussian':
        return 3
    else:
        raise ValueError(f"未知的峰形: '{peak_shape}'")

def get_param_names(peak_shape):
    """根据名称返回参数列名 (匹配 peak_master.py)"""
    shape = peak_shape.lower()
    if shape == 'voigt':
        return ['Amplitude', 'Center (cm)', 'Sigma (cm)', 'Gamma (cm)']
    elif shape == 'gaussian':
        return ['Amplitude', 'Center (cm)', 'Sigma (cm)']
    else:
        raise ValueError(f"未知的峰形: '{peak_shape}'")