# -*- coding: utf-8 -*-
# 文件名: core/linear_spectrum_aug_core.py
# 描述: 实现 M5 - 线性光谱插值

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from core.data_model import SpectralProject
from .augmentation_utils import prepare_target_concentrations # 导入共享函数

def _find_adjacent_samples_spec(target_conc_val: float,
                                real_samples: List[Dict],
                                interp_dim: int) -> Tuple[Dict | None, Dict | None]:
    """
    (来自 linear_master.py 的查找逻辑)
    查找目标浓度的相邻样本 (用于光谱插值)。
    Handles edge cases (out of bounds).
    """
    if not real_samples: return None, None
    # 检查维度
    if interp_dim < 0 or interp_dim >= len(real_samples[0]['conc']):
        raise IndexError(f"光谱插值维度索引 {interp_dim} 无效。")

    # 低于最小值
    if target_conc_val < real_samples[0]['conc'][interp_dim]:
        print(f"警告: 目标值 {target_conc_val:.4f} 低于最小锚点 ({real_samples[0]['conc'][interp_dim]:.4f})。使用最小锚点光谱。")
        return real_samples[0], real_samples[0]
    # 高于最大值
    if target_conc_val > real_samples[-1]['conc'][interp_dim]:
        print(f"警告: 目标值 {target_conc_val:.4f} 高于最大锚点 ({real_samples[-1]['conc'][interp_dim]:.4f})。使用最大锚点光谱。")
        return real_samples[-1], real_samples[-1]
    # 在中间查找
    for i in range(len(real_samples) - 1):
        lower_sample = real_samples[i]
        upper_sample = real_samples[i + 1]
        lower_conc = lower_sample['conc'][interp_dim]
        upper_conc = upper_sample['conc'][interp_dim]
        if (np.isclose(target_conc_val, lower_conc) or target_conc_val > lower_conc) and \
           (np.isclose(target_conc_val, upper_conc) or target_conc_val < upper_conc):
            if not np.isclose(lower_conc, upper_conc):
                return lower_sample, upper_sample
            else:
                return lower_sample, lower_sample
    # 等于最大值
    if np.isclose(target_conc_val, real_samples[-1]['conc'][interp_dim]):
        return real_samples[-1], real_samples[-1]

    print(f"警告: 未能为 {target_conc_val:.4f} 找到相邻样本进行光谱插值。")
    return None, None


def run_linear_spectrum_interpolation(project: SpectralProject, config: Dict[str, Any]) \
        -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, Dict]:
    """
    执行线性光谱插值。
    返回: (generated_spectra_array, generated_labels_df, x_axis, task_info)
    """
    interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)
    input_version = config.get("INPUT_VERSION")

    print(f"--- M5: 开始线性光谱插值 (基于版本: '{input_version}', 维度: {interp_dim}) ---")

    # 1. 准备真实样本 (来自 linear_master.py 的 load_and_prepare_data 改编)
    input_spectra = project.spectra_versions.get(input_version)
    labels_df = project.labels_dataframe
    x_axis = project.wavelengths
    if input_spectra is None:
        raise ValueError(f"找不到输入光谱版本: {input_version}")

    real_samples = []
    real_conc_vectors_list = [] # 用于生成目标浓度
    num_conc_dims = 0
    # 尝试提取所有以 conc_ 开头的列，否则使用所有列
    conc_cols = [col for col in labels_df.columns if col.startswith('conc_')]
    if not conc_cols:
        conc_cols = labels_df.columns.tolist()
        print(f"警告: 未找到 'conc_' 前缀的标签列，将使用所有列作为浓度: {conc_cols}")
    num_conc_dims = len(conc_cols)

    for idx, row in labels_df.iterrows():
        try:
            conc_vector = np.array([row[col] for col in conc_cols], dtype=float)
            spectrum_data = input_spectra[idx]
            sample_data = {'conc': conc_vector, 'spectrum': spectrum_data, 'index': idx}
            real_samples.append(sample_data)
            real_conc_vectors_list.append(conc_vector)
        except IndexError:
             print(f"警告: 光谱数据中索引 {idx} 超出范围。跳过。")
        except KeyError as e:
             print(f"警告: 标签 DataFrame 中找不到列 {e} (索引 {idx})。跳过。")
        except ValueError as e:
             print(f"警告: 转换样本 {idx} 浓度时出错: {e}。跳过。")

    if not real_samples:
        raise ValueError("未能加载任何有效的真实样本进行插值。")

    # 按插值维度排序
    try:
        if interp_dim < 0 or interp_dim >= num_conc_dims:
             raise IndexError(f"插值维度索引 {interp_dim} 无效（标签维度 {num_conc_dims}）。")
        real_samples.sort(key=lambda s: s['conc'][interp_dim])
    except IndexError as e:
         raise IndexError(f"排序样本时出错: {e}")

    # 2. 准备目标浓度 (使用共享函数)
    real_conc_vectors = np.array(real_conc_vectors_list)
    target_concentrations_full = prepare_target_concentrations(real_conc_vectors, config)

    if target_concentrations_full.size == 0:
        raise ValueError("未能生成任何目标浓度点。")

    # 3. 执行插值 (来自 linear_master.py)
    print(f"--- 正在为 {len(target_concentrations_full)} 个目标浓度执行光谱插值 ---")
    generated_spectra = []
    generated_labels = []

    for target_conc_vector in target_concentrations_full:
        try:
            target_conc_val = target_conc_vector[interp_dim] # 获取用于查找的值
        except IndexError:
             print(f"警告: 目标浓度向量维度不足 {interp_dim+1}。跳过 {target_conc_vector}")
             continue

        lower_sample, upper_sample = _find_adjacent_samples_spec(target_conc_val, real_samples, interp_dim)
        if lower_sample is None:
            print(f"警告: 未能为 {target_conc_val:.4f} 找到相邻样本，跳过。")
            continue

        lower_spec = lower_sample['spectrum']
        upper_spec = upper_sample['spectrum']
        lower_conc_vec = lower_sample['conc']
        upper_conc_vec = upper_sample['conc']

        lower_conc_val_dim = lower_conc_vec[interp_dim]
        upper_conc_val_dim = upper_conc_vec[interp_dim]

        # 计算权重 (处理除零和边界)
        if np.isclose(lower_conc_val_dim, upper_conc_val_dim):
            weight = 0.0 # 浓度相同，取第一个
        else:
            weight = (target_conc_val - lower_conc_val_dim) / (upper_conc_val_dim - lower_conc_val_dim)
            weight = np.clip(weight, 0.0, 1.0) # 确保权重在 [0, 1]

        # 线性插值光谱
        new_spectrum = lower_spec * (1.0 - weight) + upper_spec * weight

        # 线性插值 *所有* 浓度维度
        new_label_conc = lower_conc_vec * (1.0 - weight) + upper_conc_vec * weight

        generated_spectra.append(new_spectrum)
        generated_labels.append(new_label_conc)

    if not generated_spectra:
        raise ValueError("插值失败，未生成任何光谱。")

    # 4. 格式化输出
    generated_spectra_array = np.array(generated_spectra)

    # 构建标签 DataFrame
    # 使用原始标签的列名（如果可用且匹配维度）
    if len(conc_cols) == len(generated_labels[0]):
        output_conc_cols = conc_cols
    else:
        output_conc_cols = [f'conc_{i}' for i in range(len(generated_labels[0]))]

    generated_labels_df = pd.DataFrame(generated_labels, columns=output_conc_cols)

    # 复制 task_info (因为标签结构应该保持一致)
    task_info = project.task_info.copy()

    print(f"--- 线性光谱插值完成，生成 {len(generated_spectra_array)} 个样本 ---")
    return generated_spectra_array, generated_labels_df, x_axis, task_info