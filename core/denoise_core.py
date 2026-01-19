# -*- coding: utf-8 -*-
# 文件名: core/denoise_core.py
# 描述: 包含各种降噪算法的核心模块

import numpy as np
from scipy.signal import savgol_filter


# --- 1. Savitzky-Golay 滤波 ---

def savgol_denoise(y_data, window_length=15, polyorder=3):
    """
    使用 Savitzky-Golay 滤波器对光谱进行降噪。

    参数:
    - y_data: 原始光谱 (1D numpy 数组)
    - window_length: 滤波窗口大小 (必须是大于多项式阶数的奇数)
    - polyorder: 拟合的多项式阶数

    返回:
    - y_smoothed: 降噪后的光谱 (1D numpy 数组)
    - error_message: 如果发生错误则返回字符串，否则为 None
    """
    try:
        if not isinstance(y_data, np.ndarray):
            y_data = np.array(y_data)

        # --- 输入验证 ---
        if not isinstance(window_length, int) or window_length <= 0:
            return y_data, "错误: 窗口大小必须是正整数。"
        if not isinstance(polyorder, int) or polyorder < 0:
            return y_data, "错误: 多项式阶数必须是非负整数。"

        # 确保窗口大小是奇数
        if window_length % 2 == 0:
            window_length += 1
            print(f"提示: 窗口大小调整为奇数: {window_length}")

        # 确保窗口大小大于阶数
        if window_length <= polyorder:
            # 自动调整窗口大小为 polyorder + 1 或 polyorder + 2 (确保为奇数)
            new_window = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2
            print(f"警告: 窗口大小 ({window_length}) 必须大于多项式阶数 ({polyorder})。自动调整为 {new_window}。")
            window_length = new_window

        # 确保窗口大小不超过数据长度
        if window_length > len(y_data):
            # 自动减小窗口大小
            new_window = len(y_data) if len(y_data) % 2 != 0 else len(y_data) - 1
            if new_window <= polyorder:  # 如果数据太短，无法应用
                return y_data, f"错误: 数据点 ({len(y_data)}) 过少，无法应用阶数 {polyorder} 的 Savitzky-Golay 滤波 (最小窗口 {polyorder + 1} 或 {polyorder + 2})。"
            print(f"警告: 窗口大小 ({window_length}) 大于数据长度 ({len(y_data)})。自动调整为 {new_window}。")
            window_length = new_window

        # --- 执行滤波 ---
        y_smoothed = savgol_filter(y_data, window_length, polyorder)
        return y_smoothed, None

    except Exception as e:
        error_msg = f"Savitzky-Golay 滤波失败: {e}"
        print(f"【Denoise 模块错误】: {error_msg}")
        return y_data, error_msg  # 返回原始数据和错误


# --- 2. GUI 可调用的高级 API ---

def process_spectrum_denoise(y_data, algorithm="Savitzky-Golay", **kwargs):
    """
    【GUI 主要调用函数 - 降噪】
    根据选择的算法对单个光谱进行降噪。

    参数:
    - y_data: 原始光谱 (1D numpy 数组)
    - algorithm: 算法名称 (目前仅支持 "Savitzky-Golay")
    - **kwargs: 特定于算法的参数 (例如 window_length, polyorder)

    返回:
    - denoised_spectrum: 降噪后的光谱 (1D numpy 数组)
    - error_message: 如果发生错误则返回字符串，否则为 None
    """
    if algorithm == "Savitzky-Golay":
        window = kwargs.get('window_length', 15)
        order = kwargs.get('polyorder', 3)
        denoised_spectrum, error_message = savgol_denoise(y_data, window_length=window, polyorder=order)
        return denoised_spectrum, error_message
    # elif algorithm == "Moving Average": # 未来可以添加其他算法
    #     # ... 实现移动平均 ...
    #     pass
    else:
        error_msg = f"错误: 未知的降噪算法: {algorithm}"
        print(f"【Denoise 模块错误】: {error_msg}")
        return y_data, error_msg  # 返回原始数据