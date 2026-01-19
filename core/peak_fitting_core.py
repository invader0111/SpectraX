# -*- coding: utf-8 -*-
# 文件名: core/peak_fitting_core.py
# 描述: 包含核心峰拟合、自动寻峰逻辑的模块 (新增 parse_popt 辅助函数)

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import time

# 从同级目录导入模型
from .peak_fitting_models import (
    get_fitting_function,
    get_params_per_peak,
    get_param_names  # 新增导入
)


# --- 1. 核心拟合函数 (改编自 _perform_fit) ---

def perform_fit(x_data, y_data, peak_anchors, config):
    """
    对给定的光谱数据执行引导峰拟合 (无基线)。
    ... (函数逻辑不变) ...
    """
    try:
        # --- 数据准备和验证 ---
        if not isinstance(x_data, (np.ndarray, pd.Series)): x_data = np.array(x_data)
        if not isinstance(y_data, (np.ndarray, pd.Series)): y_data = np.array(y_data)
        if len(x_data) != len(y_data): return None, None, np.inf, "X 和 Y 数据长度不匹配"
        if not peak_anchors: return None, None, np.inf, "峰锚点列表不能为空"

        peak_shape = config.get("peak_shape", "voigt").lower()
        peak_regions = config.get('peak_regions', [(x_data.min(), x_data.max())])  # 默认整个区域
        shift_limit = config.get('center_shift_limit', 10.0)
        bounds_multiplier = config.get('bounds_multiplier', 2.5)
        max_sigma = config.get('max_sigma', 50.0)
        max_gamma = config.get('max_gamma', 50.0)
        maxfev = config.get('maxfev', 20000)
        params_per_peak = get_params_per_peak(peak_shape)
        fit_function = get_fitting_function(peak_shape)

        # --- 创建拟合区域掩码 ---
        peak_mask = create_peak_region_mask(x_data, peak_regions)
        x_fit = x_data[peak_mask]
        y_fit = y_data[peak_mask]
        if len(x_fit) == 0: return None, None, np.inf, "在指定的峰区域内没有数据点"

        # 确保 x_fit 和 y_fit 是 numpy 数组
        if isinstance(x_fit, pd.Series): x_fit = x_fit.values
        if isinstance(y_fit, pd.Series): y_fit = y_fit.values

        # --- 构建初始参数和边界 ---
        initial_params, lower_bounds, upper_bounds = [], [], []
        for center_anchor in peak_anchors:
            # 在拟合区域内查找最接近锚点的位置及其高度
            idx_in_fit_region = np.abs(x_fit - center_anchor).argmin()
            height = y_fit[idx_in_fit_region]
            if height <= 0: height = 1e-6  # 避免零或负高度

            # 查找此锚点所属的区域边界
            current_region_start, current_region_end = -np.inf, np.inf
            found_region = False
            for start, end in peak_regions:
                if start <= center_anchor <= end:
                    current_region_start, current_region_end = start, end
                    found_region = True
                    break
            if not found_region:
                print(f"警告: 锚点 {center_anchor:.2f} 未在 peak_regions 中找到。使用非约束边界。")
                center_lb = center_anchor - shift_limit
                center_ub = center_anchor + shift_limit
            else:
                # 应用双重约束
                proposed_lb = center_anchor - shift_limit
                proposed_ub = center_anchor + shift_limit
                center_lb = max(proposed_lb, current_region_start)
                center_ub = min(proposed_ub, current_region_end)
                # 防止下限高于上限
                if center_lb > center_ub: center_lb = center_ub = center_anchor

            # 添加参数和边界
            if peak_shape == 'voigt':
                initial_params.extend([height, center_anchor, 5.0, 2.0])  # 初始宽度猜测值调整
                lower_bounds.extend([0, center_lb, 0.1, 0.01])  # 宽度下限调整
                upper_bounds.extend([height * bounds_multiplier + 1, center_ub, max_sigma, max_gamma])
            else:  # gaussian
                initial_params.extend([height, center_anchor, 5.0])  # 初始宽度猜测值调整
                lower_bounds.extend([0, center_lb, 0.1])  # 宽度下限调整
                upper_bounds.extend([height * bounds_multiplier + 1, center_ub, max_sigma])

        # --- 执行拟合 ---
        popt, pcov = curve_fit(fit_function, x_fit, y_fit, p0=initial_params, bounds=(lower_bounds, upper_bounds),
                               maxfev=maxfev)

        # 检查参数数量是否正确
        if len(popt) != len(peak_anchors) * params_per_peak:
            return None, None, np.inf, f"拟合参数数量 ({len(popt)}) 与预期 ({len(peak_anchors)}*{params_per_peak}) 不匹配"

        # --- 计算结果 ---
        fit_y_in_region = fit_function(x_fit, *popt)
        sse = np.sum((y_fit - fit_y_in_region) ** 2)

        # 计算完整 X 轴上的拟合曲线
        # 确保 x_data 是 numpy 数组
        if isinstance(x_data, pd.Series):
            x_data_np = x_data.values
        else:
            x_data_np = x_data
        fit_y_full = fit_function(x_data_np, *popt)

        return popt, fit_y_full, sse, None  # 成功时返回 None 错误消息

    except (RuntimeError, ValueError) as e:
        error_msg = f"拟合期间发生错误: {e}"
        print(error_msg)
        return None, None, np.inf, error_msg
    except Exception as e:
        error_msg = f"拟合期间发生意外错误: {e}"
        import traceback
        traceback.print_exc()
        return None, None, np.inf, error_msg


# --- 2. 自动寻峰函数 (改编自 find_peak_anchors) ---

def find_auto_anchors(x_data, y_data, manual_anchors, config):
    """
    在指定区域内自动寻找峰锚点，并移除与手动锚点重叠的部分。
    ... (函数逻辑不变) ...
    """
    try:
        start_time = time.time()
        # --- 数据准备和参数获取 ---
        if not isinstance(x_data, (np.ndarray, pd.Series)): x_data = np.array(x_data)
        if not isinstance(y_data, (np.ndarray, pd.Series)): y_data = np.array(y_data)
        if len(x_data) != len(y_data): return [], "X 和 Y 数据长度不匹配"

        peak_regions = config.get('peak_regions', [(x_data.min(), x_data.max())])
        height_threshold = config.get('peak_height_threshold', 10)  # 降低默认阈值?
        distance = config.get('peak_distance', 15)
        tolerance = config.get('peak_tolerance', 10.0)
        use_smoothing = config.get('use_smoothing', False)
        window = config.get('filter_window_length', 13)
        order = config.get('filter_polyorder', 3)
        max_peaks = config.get('max_peaks_to_find', None)  # 新增参数

        print(f"--- 自动寻峰开始 (容差={tolerance} cm⁻¹) ---")

        # --- 平滑处理 (可选) ---
        if use_smoothing:
            if window % 2 == 0: window += 1
            if window <= order: window = order + 1 if (order + 1) % 2 != 0 else order + 2
            if window > len(y_data): window = len(y_data) if len(y_data) % 2 != 0 else len(y_data) - 1
            if window > order:  # 再次检查
                y_processed = savgol_filter(y_data, window, order)
                print(f"  > 已应用 Savitzky-Golay 平滑 (Win={window}, Ord={order})")
            else:
                y_processed = y_data  # 数据太短无法平滑
                print(f"  > 警告: 数据点 ({len(y_data)}) 过少，无法应用平滑处理。")
        else:
            y_processed = y_data

        # --- 创建掩码并应用 ---
        peak_mask = create_peak_region_mask(x_data, peak_regions)
        y_masked = y_processed.copy()
        # 确保是 numpy array
        if isinstance(peak_mask, pd.Series):
            peak_mask_np = peak_mask.values
        else:
            peak_mask_np = peak_mask
        y_masked[~peak_mask_np] = 0  # 将区域外设为零

        # --- 执行 scipy 寻峰 ---
        peak_indices, properties = find_peaks(y_masked, height=height_threshold, distance=distance)

        # 确保 x_data 是 numpy array
        if isinstance(x_data, pd.Series):
            x_data_np = x_data.values
        else:
            x_data_np = x_data

        all_found_peaks = [(x_data_np[i], properties['peak_heights'][j])
                           for j, i in enumerate(peak_indices) if peak_mask_np[i]]  # 再次确认在区域内

        if not all_found_peaks:
            print("  > 在指定区域内未能找到任何自动峰。")
            return [], None  # 返回空列表和 None 错误

        print(f"  > 初始找到 {len(all_found_peaks)} 个候选峰。")

        # --- 聚类 (与 peak_master.py 逻辑类似) ---
        sorted_peaks_by_pos = sorted(all_found_peaks, key=lambda p: p[0])
        clusters, current_cluster = [], [sorted_peaks_by_pos[0]]
        for i in range(1, len(sorted_peaks_by_pos)):
            if sorted_peaks_by_pos[i][0] - current_cluster[-1][0] <= tolerance:
                current_cluster.append(sorted_peaks_by_pos[i])
            else:
                clusters.append(current_cluster);
                current_cluster = [sorted_peaks_by_pos[i]]
        clusters.append(current_cluster)

        # 使用簇中最高点的位置作为锚点（或平均位置？）- 这里用最高点的位置
        cluster_anchors_raw = []
        for c in clusters:
            best_peak = max(c, key=lambda p: p[1])  # 找到簇中最高的峰
            cluster_anchors_raw.append(best_peak[0])  # 使用最高峰的位置

        print(f"  > 聚类后得到 {len(cluster_anchors_raw)} 个潜在自动锚点。")

        # --- 过滤与手动锚点重叠的部分 ---
        auto_anchors_filtered = []
        manual_anchors_set = set(manual_anchors)  # 转换为集合以便快速查找 (虽然列表小可能没必要)

        for auto_peak in cluster_anchors_raw:
            is_overlapping = False
            for manual_peak in manual_anchors_set:
                if np.abs(auto_peak - manual_peak) <= tolerance:
                    is_overlapping = True
                    # print(f"  > 忽略自动峰 {auto_peak:.2f} (与手动峰 {manual_peak:.2f} 重叠)")
                    break
            if not is_overlapping:
                auto_anchors_filtered.append(auto_peak)

        print(f"  > 移除与手动锚点重叠后，保留 {len(auto_anchors_filtered)} 个自动锚点。")

        # --- 限制数量 (可选) ---
        if max_peaks is not None and len(auto_anchors_filtered) > max_peaks:
            # 如果需要按重要性排序，需要修改聚类部分以存储重要性
            # 这里简单地按位置排序后截断
            auto_anchors_filtered = sorted(auto_anchors_filtered)[:max_peaks]
            print(f"  > 根据 max_peaks_to_find 限制，最终返回 {len(auto_anchors_filtered)} 个自动锚点。")

        final_auto_anchors = sorted(auto_anchors_filtered)
        print(f"--- 自动寻峰完成 (耗时: {time.time() - start_time:.2f} 秒): 返回 {len(final_auto_anchors)} 个锚点 ---")
        return final_auto_anchors, None  # 返回列表和 None 错误

    except Exception as e:
        error_msg = f"自动寻峰失败: {e}"
        import traceback
        traceback.print_exc()
        return [], error_msg  # 返回空列表和错误消息


# --- VVVV 新增: 辅助函数，用于解析 popt VVVV ---
def parse_popt_to_dataframe(popt, peak_shape):
    """
    将 curve_fit 返回的扁平化 popt 数组转换为易于阅读的 DataFrame。
    """
    try:
        params_per_peak = get_params_per_peak(peak_shape)
        column_names = get_param_names(peak_shape)

        if len(popt) % params_per_peak != 0:
            print(f"警告: popt 长度 ({len(popt)}) 不是 {params_per_peak} 的倍数。")
            return pd.DataFrame()  # 返回空

        num_peaks = len(popt) // params_per_peak
        data = np.reshape(popt, (num_peaks, params_per_peak))

        df = pd.DataFrame(data, columns=column_names)
        df.index = [f"Peak {i + 1}" for i in range(num_peaks)]

        return df

    except Exception as e:
        print(f"解析 popt 失败: {e}")
        return pd.DataFrame()  # 返回空


# ... (在 parse_popt_to_dataframe 函数之后) ...

# --- VVVV 新增: 辅助函数，用于从 popt 重建曲线 VVVV ---
def reconstruct_fit_from_popt(x_data, popt, peak_shape):
    """
    根据拟合参数 (popt) 和峰形，重建完整的拟合曲线。

    Args:
        x_data (np.ndarray): 完整的 X 轴数据。
        popt (np.ndarray): 扁平化的拟合参数数组。
        peak_shape (str): 峰形 ('voigt', 'gaussian', etc.)。

    Returns:
        np.ndarray: 重建的 Y 轴拟合曲线。
    """
    try:
        # 1. 从模型文件中获取拟合函数
        # (get_fitting_function 已在文件顶部导入)
        fit_function = get_fitting_function(peak_shape)

        # 2. 确保 x_data 是 numpy 数组 (与 perform_fit 中的逻辑一致)
        if isinstance(x_data, pd.Series):
            x_data_np = x_data.values
        elif not isinstance(x_data, np.ndarray):
            x_data_np = np.array(x_data)
        else:
            x_data_np = x_data

        if x_data_np.ndim == 0 or x_data_np.size == 0:  # 处理空数据或标量
            print("警告: reconstruct_fit_from_popt 接收到空的 x_data。")
            return np.array([])

        # 3. 计算 y 曲线
        fit_y_full = fit_function(x_data_np, *popt)
        return fit_y_full

    except Exception as e:
        print(f"错误: 从 popt 重建拟合曲线失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个零数组作为后备，以防绘图崩溃
        if 'x_data_np' in locals() and hasattr(x_data_np, 'shape'):
            return np.zeros_like(x_data_np)
        else:
            return np.array([])




# --- 3. 辅助函数 (来自 peak_master.py) ---

def create_peak_region_mask(x_data, peak_regions):
    """
    根据定义的峰区域创建布尔掩码。
    ... (函数逻辑不变) ...
    """
    # 确保 x_data 是 numpy array
    if isinstance(x_data, pd.Series):
        x_data_np = x_data.values
    else:
        x_data_np = np.array(x_data)

    mask = np.zeros_like(x_data_np, dtype=bool)
    if not peak_regions:  # 如果列表为空，则不限制区域
        mask[:] = True
        return mask

    for start, end in peak_regions:
        mask |= (x_data_np >= start) & (x_data_np <= end)
    return mask