# -*- coding: utf-8 -*-
# 文件名: core/augmentation_utils.py
# 描述: 数据增强模块共享的辅助函数

import os
import re
import pandas as pd
import numpy as np
from scipy.special import wofz
from typing import Dict, Any, List, Tuple


# --- 光谱重建函数 (来自 加噪.py / MOGP_master.py) ---

def voigt(x, amp, cen, sig, gam):
    """单个Voigt峰"""
    # 添加保护，防止负值或零导致计算错误
    sig = max(abs(sig), 1e-6)
    gam = max(abs(gam), 1e-6)
    amp = max(amp, 0.0)  # 振幅不能为负

    z = ((x - cen) + 1j * gam) / (sig * np.sqrt(2.0))
    V = wofz(z).real / (sig * np.sqrt(2.0 * np.pi))

    # 归一化因子，确保 amp 是峰高
    z_peak = (1j * gam) / (sig * np.sqrt(2.0))
    V_peak = wofz(z_peak).real / (sig * np.sqrt(2.0 * np.pi))

    if V_peak > 1e-9:  # 避免除零
        return amp * (V / V_peak)
    else:
        return np.zeros_like(x)


def gaussian(x, amp, cen, sig):
    """单个Gaussian峰"""
    sig = max(abs(sig), 1e-6)
    amp = max(amp, 0.0)
    return amp * np.exp(-(x - cen) ** 2 / (2 * sig ** 2))


def reconstruct_spectrum_from_params(x_axis: np.ndarray, params_row: pd.Series) -> np.ndarray:
    """
    根据包含峰参数的 Pandas Series 重建光谱。
    自动检测峰形 (Voigt 或 Gaussian)。
    """
    y = np.zeros_like(x_axis, dtype=float)

    # 检测参数开始的索引和峰形
    param_start_index = -1
    peak_shape = 'gaussian'  # 默认
    params_per_peak = 3

    # 查找第一个以 'peak_' 开头的列
    for i, col in enumerate(params_row.index):
        if col.startswith('peak_'):
            param_start_index = i
            # 通过检查第一个峰的参数名确定峰形
            if f'peak_1_gamma' in params_row.index:
                peak_shape = 'voigt'
                params_per_peak = 4
            break

    if param_start_index == -1:
        print("警告: 在 params_row 中未找到 'peak_' 开头的参数列。")
        return y  # 返回零光谱

    # 提取所有峰参数到一个 numpy 数组
    peak_params_flat = params_row.iloc[param_start_index:].to_numpy(dtype=float)
    num_peaks_detected = len(peak_params_flat) // params_per_peak

    if len(peak_params_flat) % params_per_peak != 0:
        print(f"警告: 参数数量 ({len(peak_params_flat)}) 不是 {params_per_peak} 的整数倍。可能解析错误。")
        # 尝试继续，但可能出错

    # 循环重建
    for i in range(num_peaks_detected):
        start = i * params_per_peak
        end = start + params_per_peak
        params = peak_params_flat[start:end]

        try:
            if peak_shape == 'voigt':
                # Voigt 参数顺序: MOGP/peak插值 输出的是 pos, height, sigma, gamma
                # 但我们需要检查 params_row 的列名来确定实际顺序
                # 假设顺序是 MOGP 输出的标准顺序：
                # peak_N_pos, peak_N_height, peak_N_sigma, peak_N_gamma
                pos, amp, sig, gam = params[0], params[1], params[2], params[3]
                if amp > 1e-6:  # 只添加有意义的峰
                    y += voigt(x_axis, amp, pos, sig, gam)
            else:  # Gaussian
                # 假设顺序：peak_N_pos, peak_N_height, peak_N_sigma
                pos, amp, sig = params[0], params[1], params[2]
                if amp > 1e-6:
                    y += gaussian(x_axis, amp, pos, sig)
        except IndexError:
            print(f"警告: 重建第 {i + 1} 个峰时参数不足。跳过此峰。")
            continue

    return y


# --- 残差处理函数 (来自 加噪.py) ---

def load_residual_library(residual_dir: str, interp_dim_index: int) -> Tuple[
    Dict[Tuple, np.ndarray] | None, List[Tuple] | None]:
    """
    加载所有真实的残差文件到内存中。
    根据指定的插值维度对锚点进行排序。
    返回: (residual_library, sorted_anchors) 或 (None, None) 如果失败。
    """
    print(f"--- 正在从 '{residual_dir}' 加载残差库 ---")
    file_pattern = re.compile(r'residual_conc_(.*)\.csv')
    residual_library = {}
    anchor_list = []

    if not os.path.isdir(residual_dir):
        print(f"警告: 找不到残差目录 '{residual_dir}'。将无法添加残差。")
        return None, None

    files_found = 0
    for filename in os.listdir(residual_dir):
        match = file_pattern.match(filename)
        if match:
            conc_str = match.group(1).strip('_')
            try:
                # 解析浓度向量为浮点数元组
                conc_vector = tuple(float(c) for c in conc_str.split('_'))

                # 读取残差数据
                df = pd.read_csv(os.path.join(residual_dir, filename))
                if 'residual_intensity' not in df.columns:
                    print(f"警告: 残差文件 '{filename}' 缺少 'residual_intensity' 列。跳过。")
                    continue

                residual_library[conc_vector] = df['residual_intensity'].to_numpy()
                anchor_list.append(conc_vector)
                files_found += 1
            except ValueError:
                print(f"警告: 无法解析文件 '{filename}' 中的浓度。跳过。")
            except Exception as e:
                print(f"警告: 读取残差文件 '{filename}' 时出错: {e}。跳过。")

    if not residual_library:
        print(f"警告: 在 '{residual_dir}' 中未找到有效的残差文件。将无法添加残差。")
        return None, None

    # 根据指定的插值维度对锚点进行排序
    try:
        # 确保 interp_dim_index 在浓度向量的有效范围内
        if not anchor_list or interp_dim_index < 0 or interp_dim_index >= len(anchor_list[0]):
            raise IndexError(f"插值维度索引 {interp_dim_index} 无效。")
        sorted_anchors = sorted(anchor_list, key=lambda c: c[interp_dim_index])
    except IndexError as e:
        print(f"错误: 排序残差锚点时出错: {e}。将返回未排序的锚点。")
        sorted_anchors = anchor_list  # Fallback
    except Exception as e:
        print(f"错误: 排序残差锚点时发生未知错误: {e}。将返回未排序的锚点。")
        sorted_anchors = anchor_list  # Fallback

    print(f"成功加载 {files_found} 个残差样本。")
    return residual_library, sorted_anchors


def get_interpolated_residual(target_conc_vector: Tuple[float, ...],
                              library: Dict[Tuple, np.ndarray],
                              sorted_anchors: List[Tuple],
                              interp_dim_index: int) -> np.ndarray | None:
    """
    使用 K-NN (K=2, 线性插值) 来“借用”并插值残差。
    """
    if not library or not sorted_anchors:
        return None  # 如果库为空，返回 None

    try:
        target_conc_val = target_conc_vector[interp_dim_index]
    except IndexError:
        print(f"错误: 目标浓度向量中索引 {interp_dim_index} 无效。")
        return library[sorted_anchors[0]]  # Fallback to first residual

    # K-NN 查找: 找到包围 target_conc_val 的两个最近的锚点
    lower_anchor, upper_anchor = None, None
    for i in range(len(sorted_anchors) - 1):
        c1 = sorted_anchors[i]
        c2 = sorted_anchors[i + 1]
        # 确保维度存在
        if interp_dim_index >= len(c1) or interp_dim_index >= len(c2):
            print(f"警告: 残差锚点维度不足 {interp_dim_index + 1}。无法插值。")
            return library[sorted_anchors[0]]  # Fallback

        if c1[interp_dim_index] <= target_conc_val <= c2[interp_dim_index]:
            lower_anchor = c1
            upper_anchor = c2
            break

    # 如果找不到 (例如 target_conc 超出了锚点的范围)，则使用最近的锚点
    if lower_anchor is None or upper_anchor is None:
        # 简化处理：使用最近邻
        min_dist = float('inf')
        nearest_anchor = sorted_anchors[0]
        for anchor in sorted_anchors:
            try:
                dist = abs(anchor[interp_dim_index] - target_conc_val)
                if dist < min_dist:
                    min_dist = dist
                    nearest_anchor = anchor
            except IndexError:
                continue  # 跳过维度不足的锚点

        print(
            f"警告: 目标浓度 {target_conc_val} 超出范围或未找到包围锚点。使用最近邻 ({nearest_anchor[interp_dim_index]:.4f}) 的残差。")
        return library.get(nearest_anchor)  # 使用 .get() 以防万一

    # K-NN (K=2) 插值
    lower_res = library.get(lower_anchor)
    upper_res = library.get(upper_anchor)

    if lower_res is None or upper_res is None:
        print(f"错误: 在库中找不到锚点 {lower_anchor} 或 {upper_anchor} 的残差。")
        return lower_res if lower_res is not None else upper_res  # Fallback

    lower_conc_val = lower_anchor[interp_dim_index]
    upper_conc_val = upper_anchor[interp_dim_index]

    if np.isclose(lower_conc_val, upper_conc_val):
        return lower_res  # 避免除以零

    # 计算插值权重 (限制在 0-1 之间)
    weight = (target_conc_val - lower_conc_val) / (upper_conc_val - lower_conc_val)
    weight = np.clip(weight, 0.0, 1.0)

    # 线性插值残差
    try:
        interpolated_residual = lower_res * (1.0 - weight) + upper_res * weight
        return interpolated_residual
    except ValueError as e:
        print(f"错误: 插值残差时维度不匹配: {e} (形状: {lower_res.shape} vs {upper_res.shape})")
        return lower_res  # Fallback


# --- 目标浓度生成函数 (来自 MOGP_master.py / linear_master.py) ---

def prepare_target_concentrations(real_conc_vectors: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    根据 config 中的 GENERATION_MODE 准备生成浓度列表。
    输入 real_conc_vectors 是 N x D 数组。
    """
    mode = config.get("GENERATION_MODE", "interpolate").lower()
    interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)
    num_conc_dims = real_conc_vectors.shape[1]

    print(f"\n--- 准备目标浓度 (模式: '{mode}', 基于维度 {interp_dim}) ---")

    if mode == 'specific':
        # 处理 'Specific' 模式，允许单值 (用于线性) 或多维列表 (用于参数)
        specific_targets = config.get("SPECIFIC_CONCENTRATIONS_TO_GENERATE", [])
        if not specific_targets:
            raise ValueError("选择了 'Specific' 模式，但 SPECIFIC_CONCENTRATIONS_TO_GENERATE 列表为空。")

        # 尝试将输入统一为 N x D 数组
        try:
            target_array = np.array(specific_targets)
            if target_array.ndim == 0:  # 单个数字
                if num_conc_dims > 1: raise ValueError("输入是单值，但原始标签是多维。")
                target_array = target_array.reshape(1, 1)
            elif target_array.ndim == 1:  # 一维列表 (线性模式常用)
                # 如果原始标签是多维，需要决定如何处理其他维度 (用0填充？)
                if num_conc_dims > 1:
                    print(f"警告: 输入是一维特定值，但原始标签有 {num_conc_dims} 维。其他维度将用 0 填充。")
                    padded_targets = np.zeros((len(target_array), num_conc_dims))
                    padded_targets[:, interp_dim] = target_array  # 将值放入插值维度
                    target_array = padded_targets
                else:
                    target_array = target_array.reshape(-1, 1)  # 转为 N x 1
            elif target_array.ndim == 2:  # 二维列表 (参数模式常用)
                if target_array.shape[1] != num_conc_dims:
                    raise ValueError(
                        f"输入的特定值维度 ({target_array.shape[1]}) 与原始标签维度 ({num_conc_dims}) 不匹配。")
            else:
                raise ValueError("SPECIFIC_CONCENTRATIONS_TO_GENERATE 格式无效（维度>2）。")

        except Exception as e:
            raise ValueError(f"解析 SPECIFIC_CONCENTRATIONS_TO_GENERATE 时出错: {e}")

        concentrations = target_array
        print(f"将为 {len(concentrations)} 个指定浓度点生成数据。")
        return concentrations

    elif mode == 'interpolate':
        step = config.get("INTERPOLATION_STEP", 0.01)
        if step <= 0: raise ValueError("INTERPOLATION_STEP 必须大于 0。")

        all_interp_points = []
        # 按插值维度排序
        try:
            sort_indices = real_conc_vectors[:, interp_dim].argsort()
            X_train_sorted = real_conc_vectors[sort_indices]
        except IndexError:
            raise IndexError(f"插值维度索引 {interp_dim} 无效（标签维度为 {num_conc_dims}）。")

        for i in range(len(X_train_sorted) - 1):
            start_point = X_train_sorted[i]
            end_point = X_train_sorted[i + 1]
            diff_vec = end_point - start_point

            # 检查插值维度上的距离
            interp_dim_dist = abs(end_point[interp_dim] - start_point[interp_dim])

            # 只有当插值维度不同时才进行插值
            if np.isclose(interp_dim_dist, 0):
                continue

            # 计算步数（至少插一个点）
            num_steps = max(1, int(np.floor(interp_dim_dist / step)))

            # 生成插值点 (不包括端点)
            # np.linspace(0, 1, num_steps + 2) -> [0, ..., 1] 共 num_steps+2 个点
            # [1:-1] -> 去掉 0 和 1
            interp_factors = np.linspace(0, 1, num_steps + 2)[1:-1]
            if interp_factors.size > 0:
                # 使用广播生成所有插值点
                interp_points = start_point + interp_factors[:, np.newaxis] * diff_vec
                all_interp_points.append(interp_points)

        if not all_interp_points:
            print("警告：在真实样本之间未能生成任何插值点。检查 INTERPOLATION_STEP。")
            return np.array([])  # 返回空数组

        concentrations = np.vstack(all_interp_points)
        # 去重（可选，但可能有用）
        concentrations = np.unique(concentrations, axis=0)
        print(f"在 {len(real_conc_vectors)} 个真实数据点之间生成了 {len(concentrations)} 个插值浓度点。")
        return concentrations

    else:
        raise ValueError(f"未知的生成模式: '{mode}'. 应为 'interpolate' 或 'specific'。")


# --- M4 结果解析辅助函数 ---
def get_m4_input_data(project: 'SpectralProject',
                      m4_fit_results_dict: Dict[int, pd.DataFrame],
                      config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, pd.DataFrame], List[Tuple], int, int, str]:
    """
    从主项目和M4结果字典中提取线性参数插值所需的数据。
    返回: x_axis, relevant_m4_results, real_samples_conc_params, num_peaks, params_per_peak, peak_shape
    """
    interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)

    # 1. 检查 M4 结果
    if not m4_fit_results_dict:
        raise ValueError("M4 拟合结果字典为空。")

    # 2. 获取 x_axis
    x_axis = project.wavelengths

    # 3. 确定峰形和参数数量 (从第一个 M4 结果 DataFrame 推断)
    first_idx = next(iter(m4_fit_results_dict))
    first_df = m4_fit_results_dict[first_idx]
    peak_shape = 'voigt' if 'Gamma (cm)' in first_df.columns else 'gaussian'
    params_per_peak = 4 if peak_shape == 'voigt' else 3
    num_peaks = len(first_df)

    # 4. 定义要提取的参数列 (与 peak插值.py 一致)
    param_col_order = ['Center (cm)', 'Amplitude', 'Sigma (cm)']
    if peak_shape == 'voigt':
        param_col_order.append('Gamma (cm)')

    # 5. 构建 real_samples 列表 (包含浓度和参数)
    real_samples_conc_params = []
    labels_df = project.labels_dataframe

    # 查找所有被 M4 拟合的样本索引
    processed_indices = sorted(m4_fit_results_dict.keys())

    for idx in processed_indices:
        # 获取浓度标签
        try:
            # 假设标签列是 'conc_0', 'conc_1', ... 或直接是数值列
            conc_vector_list = []
            if f'conc_{interp_dim}' in labels_df.columns:  # 检查标准命名
                num_conc_dims = sum(1 for col in labels_df.columns if col.startswith('conc_'))
                conc_vector_list = [labels_df.loc[idx, f'conc_{d}'] for d in range(num_conc_dims)]
            else:  # 尝试直接使用标签列
                # (需要更复杂的逻辑来确定哪些列是浓度)
                # 简化：假设所有标签列都是浓度
                conc_vector_list = labels_df.loc[idx].tolist()

            conc_vector = np.array(conc_vector_list)
        except KeyError:
            print(f"警告: 无法为样本 {idx} 找到浓度标签。跳过此样本。")
            continue
        except Exception as e:
            print(f"警告: 获取样本 {idx} 浓度时出错: {e}。跳过此样本。")
            continue

        # 获取 M4 参数
        params_df = m4_fit_results_dict[idx]
        if len(params_df) != num_peaks:
            print(f"警告: 样本 {idx} 的 M4 结果峰数 ({len(params_df)}) 与预期 ({num_peaks}) 不符。跳过。")
            continue
        try:
            # 按固定顺序提取参数
            params_array = params_df[param_col_order].to_numpy()  # Shape [num_peaks, params_per_peak]
        except KeyError as e:
            print(f"警告: 样本 {idx} 的 M4 结果缺少列: {e}。跳过。")
            continue

        real_samples_conc_params.append({
            'concentration': conc_vector,
            'all_params': params_array  # 存储 [num_peaks, params_per_peak] 形状的数组
        })

    if not real_samples_conc_params:
        raise ValueError("未能从 M4 结果中提取任何有效的样本数据。")

    # 按插值维度排序
    try:
        if interp_dim < 0 or interp_dim >= len(real_samples_conc_params[0]['concentration']):
            raise IndexError(f"插值维度索引 {interp_dim} 无效。")
        real_samples_conc_params.sort(key=lambda r: r['concentration'][interp_dim])
    except IndexError as e:
        raise IndexError(f"排序样本时出错: {e}")

    return x_axis, m4_fit_results_dict, real_samples_conc_params, num_peaks, params_per_peak, peak_shape


# --- 用于线性参数插值的查找函数 (来自 peak插值.py) ---
def find_adjacent_samples_for_params(target_conc_val: float,
                                     data_records: List[Dict],
                                     interp_dim: int) -> Tuple[Dict | None, Dict | None]:
    """
    (来自 peak插值.py) 查找目标浓度的相邻样本 (用于参数插值)。
    Handles edge cases (out of bounds).
    """
    if not data_records: return None, None

    # 检查维度
    if interp_dim < 0 or interp_dim >= len(data_records[0]['concentration']):
        raise IndexError(f"插值维度索引 {interp_dim} 无效。")

    # 如果目标低于最小值
    if target_conc_val < data_records[0]['concentration'][interp_dim]:
        print(
            f"警告: 目标值 {target_conc_val:.4f} 低于最小锚点 ({data_records[0]['concentration'][interp_dim]:.4f})。将使用最小锚点。")
        return data_records[0], data_records[0]  # 使用边界点

    # 如果目标高于最大值
    if target_conc_val > data_records[-1]['concentration'][interp_dim]:
        print(
            f"警告: 目标值 {target_conc_val:.4f} 高于最大锚点 ({data_records[-1]['concentration'][interp_dim]:.4f})。将使用最大锚点。")
        return data_records[-1], data_records[-1]  # 使用边界点

    # 在中间查找
    for i in range(len(data_records) - 1):
        lower_sample = data_records[i]
        upper_sample = data_records[i + 1]

        lower_conc = lower_sample['concentration'][interp_dim]
        upper_conc = upper_sample['concentration'][interp_dim]

        # 找到区间
        # 使用 np.isclose 处理浮点数比较
        if (np.isclose(target_conc_val, lower_conc) or target_conc_val > lower_conc) and \
                (np.isclose(target_conc_val, upper_conc) or target_conc_val < upper_conc):

            # 确保分母不为零 (除非 lower == upper)
            if not np.isclose(lower_conc, upper_conc):
                return lower_sample, upper_sample
            else:
                # 如果两个锚点浓度非常接近，只需返回一个
                return lower_sample, lower_sample  # 或 upper_sample

    # 如果恰好等于最大值 (上面的循环不会捕捉到 i = len-1 的情况)
    if np.isclose(target_conc_val, data_records[-1]['concentration'][interp_dim]):
        return data_records[-1], data_records[-1]

    # 理论上不应到达这里，但作为回退
    print(f"警告: 未能为 {target_conc_val:.4f} 找到相邻样本 (可能数据未排序或有间隙？)。")
    return None, None


def load_m4_fit_results_from_csv(file_path: str) -> Dict[int, pd.DataFrame]:
    """
    (新增) 从 CSV 文件加载 M4 拟合结果。

    参数:
        file_path: CSV 文件路径。

    返回:
        Dict[int, pd.DataFrame]: 格式为 {sample_index: params_df} 的字典，
                                 其中 params_df 的每一行代表一个峰。
    """
    print(f"正在从 CSV 加载 M4 结果: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"读取 CSV 文件失败: {e}")

    # 验证关键列
    if 'sample_index' not in df.columns:
        raise ValueError("CSV 文件缺少 'sample_index' 列。无法解析为 M4 拟合结果。\n"
                         "请确保这是由本软件生成的 fit_results CSV 文件。")

    results_dict = {}

    # 按 sample_index 分组 (因为一个样本可能有多个峰/多行)
    grouped = df.groupby('sample_index')

    for sample_idx, group in grouped:
        try:
            idx = int(sample_idx)

            # 移除 sample_index 列，保留参数列 (和可能的标签列)
            # 使用 copy() 避免 SettingWithCopyWarning
            clean_df = group.drop(columns=['sample_index'], errors='ignore').copy()

            # 重置索引 (0, 1, 2... 对应 Peak 1, Peak 2...)
            clean_df = clean_df.reset_index(drop=True)

            results_dict[idx] = clean_df

        except Exception as e:
            print(f"警告: 解析样本 {sample_idx} 时出错: {e}")
            continue

    print(f"成功加载 {len(results_dict)} 个样本的拟合结果。")
    return results_dict