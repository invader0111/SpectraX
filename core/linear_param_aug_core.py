# -*- coding: utf-8 -*-
# 文件名: core/linear_param_aug_core.py
# 描述: 实现 M5 - 峰参数线性插值与重建
# (修正: task_info 创建; 返回 generated_params; 修复重建参数数量警告)

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from core.data_model import SpectralProject
from .augmentation_utils import ( # 导入共享函数
    prepare_target_concentrations,
    reconstruct_spectrum_from_params,
    load_residual_library,
    get_interpolated_residual,
    get_m4_input_data,
    find_adjacent_samples_for_params
)
# --- VVVV 新增导入 VVVV ---
from .peak_fitting_models import get_param_names # 用于获取 M4 风格的列名
import traceback # 用于更详细的错误打印
# --- ^^^^ 新增结束 ^^^^ ---


def run_linear_parameter_interpolation(project: SpectralProject,
                                       m4_fit_results_all: Dict[str, Dict[int, pd.DataFrame]],
                                       config: Dict[str, Any]) \
        -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, Dict, Dict[int, pd.DataFrame]]: # <-- 修改返回类型
    """
    执行峰参数线性插值、重建，并可选添加残差。
    返回: (generated_spectra_array, generated_labels_df, x_axis, task_info, generated_params) # <-- 修改返回值
    """
    m4_version = config.get("M4_RESULTS_VERSION")
    interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)
    add_residuals = config.get("ADD_RESIDUALS", False)

    print(f"--- M5: 开始峰参数线性插值 (基于 M4: '{m4_version}', 维度: {interp_dim}, 残差: {add_residuals}) ---")

    # 1. 检查并获取 M4 结果
    if not m4_version or m4_version not in m4_fit_results_all:
        raise ValueError(f"未找到或无效的 M4 拟合结果版本: '{m4_version}'")
    m4_results_dict = m4_fit_results_all.get(m4_version) # 使用 .get() 更安全
    if m4_results_dict is None:
        raise ValueError(f"在 m4_fit_results_all 中未找到键 '{m4_version}'")

    # 2. 从项目和 M4 结果准备输入数据 (使用辅助函数)
    try:
        x_axis, _, real_samples_conc_params, num_peaks, params_per_peak, peak_shape = get_m4_input_data(
            project, m4_results_dict, config
        )
        print(f"M4 输入数据准备完毕: {len(real_samples_conc_params)} 样本, {num_peaks} 峰/样本, 峰形: {peak_shape}")
    except Exception as e:
         # 打印更详细的错误
         traceback.print_exc()
         raise ValueError(f"准备 M4 输入数据时出错: {e}")

    # 3. 准备目标浓度 (使用共享函数)
    try:
        real_conc_vectors = np.array([r['concentration'] for r in real_samples_conc_params])
        target_concentrations_full = prepare_target_concentrations(real_conc_vectors, config)
    except Exception as e:
         traceback.print_exc()
         raise ValueError(f"准备目标浓度时出错: {e}")


    if target_concentrations_full.size == 0:
        # 返回空结果，而不是报错
        print("警告: 未能生成任何目标浓度点。将返回空结果。")
        return np.array([]), pd.DataFrame(), x_axis, {}, {}

    # 4. 加载残差库 (如果需要)
    residual_library, sorted_anchors = None, None
    if add_residuals:
        # 尝试根据 M4 版本名找到残差目录
        residuals_dir = config.get("RESIDUALS_PATH_ABSOLUTE")
        if not residuals_dir or not os.path.isdir(residuals_dir):
            print(f"警告: 'RESIDUALS_PATH_ABSOLUTE' 未在 config 中提供或路径无效: {residuals_dir}")
            print("将生成无残差光谱。")
            add_residuals = False  # 禁用
        else:
            print(f"--- 正在从绝对路径加载残差: {residuals_dir} ---")
            try:
                residual_library, sorted_anchors = load_residual_library(residuals_dir, interp_dim)
                if residual_library is None:
                    print("警告: 启用添加残差但未能加载残差库。将生成无残差光谱。")
                    add_residuals = False  # 禁用残差
            except Exception as e_res:
                print(f"加载残差库时出错: {e_res}。将生成无残差光谱。")
                add_residuals = False

    # 5. 执行参数插值 (来自 peak插值.py)
    print(f"--- 正在为 {len(target_concentrations_full)} 个目标浓度执行参数插值 ---")
    generated_spectra = []
    generated_spectra_noiseless = []
    generated_labels = []
    # --- VVVV 修改: 初始化存储生成的峰参数 VVVV ---
    temp_gen_params: Dict[int, pd.DataFrame] = {}
    # --- ^^^^ 修改结束 ^^^^ ---

    # 确定参数列名，用于重建和保存
    m4_style_columns = get_param_names(peak_shape) # ['Center (cm)', 'Amplitude', ...]
    # 确定重建函数需要的内部列名 (peak_1_pos, ...)
    param_names_internal = ['pos', 'height', 'sigma']
    if peak_shape == 'voigt': param_names_internal.append('gamma')
    param_cols_flat_internal = [f'peak_{j+1}_{name}' for j in range(num_peaks) for name in param_names_internal]
    # 确定浓度列名
    num_conc_dims = real_conc_vectors.shape[1]
    conc_cols = [f'conc_{i}' for i in range(num_conc_dims)] # ['conc_0', 'conc_1']


    for i, target_conc_vector in enumerate(target_concentrations_full): # <-- 使用 enumerate 获取新索引 i
        try:
            target_conc_val = target_conc_vector[interp_dim] # 获取用于查找的值
        except IndexError:
             print(f"警告: 目标浓度向量维度不足 {interp_dim+1}。跳过 {target_conc_vector}")
             continue

        lower_sample, upper_sample = find_adjacent_samples_for_params(target_conc_val, real_samples_conc_params, interp_dim)
        if lower_sample is None:
            print(f"警告: 未能为 {target_conc_val:.4f} 找到相邻样本进行参数插值，跳过。")
            continue

        lower_params = lower_sample['all_params'] # Shape [num_peaks, params_per_peak]
        upper_params = upper_sample['all_params']
        lower_conc_vec = lower_sample['concentration']
        upper_conc_vec = upper_sample['concentration']

        lower_conc_val_dim = lower_conc_vec[interp_dim]
        upper_conc_val_dim = upper_conc_vec[interp_dim]

        # 计算权重
        if np.isclose(lower_conc_val_dim, upper_conc_val_dim):
            weight = 0.0
        else:
            weight = (target_conc_val - lower_conc_val_dim) / (upper_conc_val_dim - lower_conc_val_dim)
            weight = np.clip(weight, 0.0, 1.0)

        # 线性插值 *所有* 参数 (形状 [num_peaks, params_per_peak])
        new_params_matrix = lower_params * (1.0 - weight) + upper_params * weight

        # 线性插值 *所有* 浓度维度
        new_label_conc = lower_conc_vec * (1.0 - weight) + upper_conc_vec * weight

        # 6. 重建光谱 (使用共享函数)
        # --- VVVV 修正：只将峰参数传递给重建函数 VVVV ---
        params_flat = new_params_matrix.flatten()
        # 创建 Series 时使用内部列名 (peak_1_pos...)
        if len(params_flat) != len(param_cols_flat_internal):
             print(f"!!! 严重警告: 样本 {i} 插值参数长度 ({len(params_flat)}) 与预期列名长度 ({len(param_cols_flat_internal)}) 不符！重建可能失败。")
             # 尝试截断或填充？或者直接跳过？这里选择跳过
             continue
        params_series = pd.Series(params_flat, index=param_cols_flat_internal)
        # (不再将浓度添加到 params_series 中)
        # --- ^^^^ 修正结束 ^^^^ ---
        try:
            spectrum_noiseless = reconstruct_spectrum_from_params(x_axis, params_series)
            generated_spectra_noiseless.append(spectrum_noiseless)
        except Exception as e_recon:
            print(f"警告: 重建样本 {i} 光谱时失败: {e_recon}。跳过此样本。")
            traceback.print_exc()
            continue # 跳到下一个目标浓度

        # 7. 添加残差 (如果启用)
        final_spectrum = spectrum_noiseless
        if add_residuals and residual_library and sorted_anchors:
            try:
                interpolated_residual = get_interpolated_residual(tuple(new_label_conc), residual_library, sorted_anchors, interp_dim)
                if interpolated_residual is not None:
                    # 确保残差长度与光谱匹配
                    if len(interpolated_residual) == len(final_spectrum):
                        final_spectrum = final_spectrum + interpolated_residual
                    else:
                        print(f"警告: 样本 {i} ({tuple(new_label_conc)}) 的插值残差长度 ({len(interpolated_residual)}) 与光谱 ({len(final_spectrum)}) 不匹配。跳过残差添加。")
                # else: # get_interpolated_residual 内部已有警告
                #      print(f"警告: 未能为样本 {i} ({tuple(new_label_conc)}) 插值残差。")
            except Exception as e_add_res:
                 print(f"警告: 添加样本 {i} 残差时出错: {e_add_res}。跳过残差添加。")


        generated_spectra.append(final_spectrum)
        generated_labels.append(new_label_conc)

        # --- VVVV 新增: 保存峰参数DataFrame VVVV ---
        try:
            # 使用 M4 风格的列名保存
            params_df_for_storage = pd.DataFrame(new_params_matrix, columns=m4_style_columns)
            temp_gen_params[i] = params_df_for_storage # i 是新样本索引 (0, 1, 2...)
        except Exception as e_store_df:
             print(f"警告: 存储样本 {i} 的峰参数 DataFrame 失败: {e_store_df}")
        # --- ^^^^ 新增结束 ^^^^ ---


    if not generated_spectra:
        # 如果没有生成任何有效光谱，也返回空结果
        print("警告: 参数插值或重建失败，未生成任何光谱。")
        return np.array([]), pd.DataFrame(), x_axis, {}, {}


    # 8. 格式化输出
    generated_spectra_array = np.array(generated_spectra)
    generated_labels_df = pd.DataFrame(generated_labels, columns=conc_cols)

    # --- VVVV 修正：创建新的 task_info VVVV ---
    print("创建新的 task_info 用于生成的 LinearParam 项目...")
    new_task_info = {}
    # 将所有生成的 conc_ 列标记为回归目标
    for col_name in generated_labels_df.columns:
        if col_name.startswith('conc_'):
            new_task_info[col_name] = {'role': 'target', 'type': 'regression'}
            print(f"  - 列 '{col_name}' 设为 target/regression")
        # (可选：可以尝试从原始 task_info 复制 'id' 列信息，如果存在且适用)
        id_col = project.get_id_col()
        if id_col and id_col in project.task_info:
            # 检查 ID 列是否也存在于 generated_labels_df 中 (不太可能)
            # 或者需要生成新的 ID？ 暂时不复制 ID
            print(f"  - (注意: 原始 ID 列 '{id_col}' 未复制到新项目)")
            pass

    # 使用新的 task_info
    task_info = new_task_info
    # --- ^^^^ 修正结束 ^^^^ ---

    print(f"--- 峰参数线性插值与重建完成，生成 {len(generated_spectra_array)} 个样本 ---")
    generated_spectra_array_noiseless = np.array(generated_spectra_noiseless)
    # 如果没有添加残差，确保两个数组是相同的
    if not add_residuals:
        generated_spectra_array = generated_spectra_array_noiseless

    return generated_spectra_array, generated_spectra_array_noiseless, generated_labels_df, x_axis, task_info, temp_gen_params