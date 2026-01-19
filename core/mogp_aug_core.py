# -*- coding: utf-8 -*-
# 文件名: core/mogp_aug_core.py
# 描述: 实现 M5 - MOGP 参数化生成与重建 (完整升级版: 集成 RANSAC 与 积分面积约束)

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import GPy
from core.data_model import SpectralProject
from .augmentation_utils import (
    prepare_target_concentrations,
    reconstruct_spectrum_from_params,
    load_residual_library,
    get_interpolated_residual,
)
import matplotlib.pyplot as plt
from .peak_fitting_models import get_param_names
import traceback

# --- 新增科学计算与机器学习库 ---
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import wofz
from sklearn.linear_model import RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from tqdm import tqdm

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 核心数学与物理函数 (来自 MOGP_1.py)
# =============================================================================

def gaussian_shape(x, amp, pos, sigma):
    """高斯线型函数"""
    return amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))


def voigt_shape(x, amp, pos, sigma, gamma):
    """Voigt 线型函数"""
    if sigma == 0:  # 纯洛伦兹
        return amp * (gamma ** 2 / ((x - pos) ** 2 + gamma ** 2))

    z = ((x - pos) + 1j * gamma) / (sigma * np.sqrt(2))
    val = np.real(wofz(z))
    # 归一化因子
    z0 = (1j * gamma) / (sigma * np.sqrt(2))
    val0 = np.real(wofz(z0))

    if np.abs(val0) < 1e-9: return np.zeros_like(x)
    return amp * (val / val0)


def integrate_peak_area(peak_shape, r_min, r_max, amp, pos, sigma, gamma=None):
    """对单个峰在指定范围内进行数值积分"""
    if amp <= 1e-9: return 0.0

    try:
        if peak_shape == 'gaussian':
            area, _ = quad(gaussian_shape, r_min, r_max, args=(amp, pos, sigma))
            return area
        elif peak_shape == 'voigt':
            # 确保参数非负且合理
            sigma = max(1e-6, sigma)
            gamma = max(1e-6, gamma) if gamma is not None else 0
            area, _ = quad(voigt_shape, r_min, r_max, args=(amp, pos, sigma, gamma))
            return area
    except Exception:
        return 0.0
    return 0.0


def get_peak_groups_by_range(mean_positions, ranges):
    """根据给定的波长范围对峰进行分组"""
    groups = {}
    for r_min, r_max in ranges:
        group_name = f"Range_{int(r_min)}_{int(r_max)}"
        indices_in_range = []
        for idx, pos in enumerate(mean_positions):
            if r_min <= pos <= r_max:
                indices_in_range.append(idx)

        if indices_in_range:
            groups[group_name] = {'indices': indices_in_range, 'range': (r_min, r_max)}
    return groups


# =============================================================================
# 2. 优化目标与约束函数 (用于 optimize_params_with_integration_constraint)
# =============================================================================

def objective_func(params, amps, pos_list, sigmas_init, gammas_init, peak_shape):
    """优化目标：最小化参数相对于初始预测值(MOGP预测值)的偏差"""
    total_error = 0
    num_peaks = len(amps)

    if peak_shape == 'voigt':
        for i in range(num_peaks):
            s_new = params[2 * i]
            g_new = params[2 * i + 1]
            s_init = sigmas_init[i]
            g_init = gammas_init[i]
            # 相对偏差平方和
            err_s = ((s_new - s_init) / (s_init + 1e-9)) ** 2
            err_g = ((g_new - g_init) / (g_init + 1e-9)) ** 2
            total_error += err_s + err_g
    else:  # gaussian
        for i in range(num_peaks):
            s_new = params[i]
            s_init = sigmas_init[i]
            err_s = ((s_new - s_init) / (s_init + 1e-9)) ** 2
            total_error += err_s

    return total_error


def constraint_func(params, amps, pos_list, area_target, peak_shape, r_min, r_max):
    """约束函数：(当前参数在 [r_min, r_max] 的积分总面积) - 目标面积 = 0"""
    total_area_current = 0
    num_peaks = len(amps)

    if peak_shape == 'voigt':
        for i in range(num_peaks):
            s = params[2 * i]
            g = params[2 * i + 1]
            total_area_current += integrate_peak_area('voigt', r_min, r_max, amps[i], pos_list[i], s, g)
    else:  # gaussian
        for i in range(num_peaks):
            s = params[i]
            total_area_current += integrate_peak_area('gaussian', r_min, r_max, amps[i], pos_list[i], s)

    return total_area_current - area_target


# =============================================================================
# 3. 数据准备与训练逻辑
# =============================================================================

def _prepare_mogp_training_data(project: SpectralProject,
                                m4_fit_results_all: Dict[str, Dict[int, pd.DataFrame]],
                                config: Dict[str, Any]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str,
        np.ndarray | None, List[str]]:
    """
    从项目和 M4 结果准备 MOGP 训练数据。
    (修改: 增加了基于 TRAINING_INDICES 的样本过滤逻辑)
    """
    print("--- MOGP: 准备训练数据 ---")

    m4_version = config.get("M4_RESULTS_VERSION")
    if not m4_version:
        raise ValueError("MOGP 配置中缺少 'M4_RESULTS_VERSION'")

    m4_results_dict = m4_fit_results_all.get(m4_version)
    if m4_results_dict is None:
        raise ValueError(f"M4 拟合结果字典中未找到版本 '{m4_version}'。")

    if not m4_results_dict:
        raise ValueError(f"版本 '{m4_version}' 的 M4 拟合结果为空。")

    # --- 1. 推断峰形和参数结构 ---
    try:
        first_idx = next(iter(m4_results_dict))
        first_df = m4_results_dict[first_idx]
        if isinstance(first_df, dict): first_df = first_df['results_df']
        # 通过检查列名判断是 Voigt 还是 Gaussian
        peak_shape = 'voigt' if 'Gamma (cm)' in first_df.columns else 'gaussian'
    except Exception as e:
        raise RuntimeError(f"解析 M4 结果格式失败: {e}")

    params_per_peak = 4 if peak_shape == 'voigt' else 3
    num_peaks = len(first_df)
    shared_params_detected = config.get("ASSUME_SHARED_WIDTH", False)

    # --- 2. 确定需要 MOGP 预测的目标参数列 ---
    default_targets = ['Amplitude']
    center_col_name = 'Center (cm)'
    sigma_col_name = 'Sigma (cm)'
    gamma_col_name = 'Gamma (cm)'

    if not shared_params_detected:
        default_targets.append(sigma_col_name)
        if peak_shape == 'voigt':
            default_targets.append(gamma_col_name)

    user_targets = config.get("PARAMS_TO_MODEL_WITH_MOGP", default_targets)

    # 过滤掉 Center (因为它用多项式拟合) 和无效列，保留有效的 MOGP 目标
    valid_cols = [center_col_name, 'Amplitude', sigma_col_name, gamma_col_name]
    params_for_mogp = [p for p in user_targets if p in valid_cols and p != center_col_name]

    if shared_params_detected:
        params_for_mogp = [p for p in params_for_mogp if p not in [sigma_col_name, gamma_col_name]]

    # --- 3. 确定浓度列 (输入 X) ---
    labels_df = project.labels_dataframe
    conc_cols = config.get("CONCENTRATION_COLUMNS_TO_USE_NAMES")
    if not conc_cols:
        conc_cols = [col for col in labels_df.columns if col.startswith('conc_')]
        if not conc_cols: conc_cols = labels_df.columns.tolist()

    # --- 4. (关键修改) 确定要用于训练的样本索引 ---
    target_indices = config.get("TRAINING_INDICES")
    available_indices = sorted(m4_results_dict.keys())

    indices_to_process = []

    if target_indices is not None:
        # 情况 A: 用户指定了样本 (通过左侧勾选)
        # 取交集：确保用户选的索引在当前的 M4 结果中确实存在
        # (注意: 这里将 list 转换为 set 检查成员资格，保持 target_indices 的相对顺序或使用 available 的顺序均可，这里使用 available 的顺序以保持一致性)
        target_set = set(target_indices)
        indices_to_process = [idx for idx in available_indices if idx in target_set]

        if not indices_to_process:
            raise ValueError(
                f"用户选定的样本 (Count={len(target_indices)}) 均不在选定的 M4 结果版本 ('{m4_version}') 中。\n"
                f"请检查左侧选择的样本是否已完成该版本的拟合。")

        if len(indices_to_process) < len(target_indices):
            print(
                f"警告: 用户选了 {len(target_indices)} 个样本，但在当前 M4 结果中只找到了 {len(indices_to_process)} 个。将仅使用匹配的部分。")
    else:
        # 情况 B: 用户未指定 (兼容旧逻辑或全选)，使用所有可用结果
        indices_to_process = available_indices

    print(f"MOGP: 将使用 {len(indices_to_process)} 个样本构建训练集 (总可用 M4 结果: {len(available_indices)})。")

    # --- 5. 构建训练数据集 ---
    data_records = []
    global_shared_params = None

    for idx in indices_to_process:
        try:
            # 5.1 提取浓度 (X)
            conc_vector = np.array([labels_df.loc[idx, col] for col in conc_cols], dtype=float)

            # 5.2 提取拟合参数
            params_df = m4_results_dict[idx]
            if isinstance(params_df, dict): params_df = params_df['results_df']

            # 提取位置 (用于多项式拟合)
            positions = params_df[center_col_name].to_numpy()

            # 提取 MOGP 目标参数
            mogp_arrays = []
            for col in params_for_mogp:
                if col in params_df.columns:
                    mogp_arrays.append(params_df[col].to_numpy())

            # 展平为一维数组 [peak1_amp, peak2_amp, ..., peak1_sig, peak2_sig, ...]
            # 注意: 这里的顺序取决于 params_for_mogp 的顺序
            mogp_params = np.array(mogp_arrays).T.flatten() if mogp_arrays else np.array([])

            data_records.append({
                'concentration': conc_vector,
                'positions': positions,
                'mogp_params': mogp_params
            })

            # 5.3 处理共享宽度逻辑 (如果启用)
            if shared_params_detected and global_shared_params is None:
                try:
                    s = params_df[sigma_col_name].to_numpy()[0]
                    if peak_shape == 'voigt':
                        g = params_df[gamma_col_name].to_numpy()[0]
                        global_shared_params = np.array([s, g])
                    else:
                        global_shared_params = np.array([s])
                except Exception:
                    pass  # 如果读取失败，稍后会报错或保持 None

        except Exception as e:
            print(f"警告: 跳过样本 {idx} (数据提取失败): {e}")
            continue

    if not data_records:
        raise ValueError("未能构建任何有效的训练数据记录。请检查输入数据和列名配置。")

    # --- 6. 转换为 Numpy 数组 ---
    X_train = np.array([r['concentration'] for r in data_records])
    Y_train_linear = np.array([r['positions'] for r in data_records])
    Y_train_mogp = np.array([r['mogp_params'] for r in data_records])

    print(f"训练数据准备完毕: X_train shape={X_train.shape}, Y_mogp shape={Y_train_mogp.shape}")

    return X_train, Y_train_linear, Y_train_mogp, num_peaks, params_per_peak, peak_shape, global_shared_params, params_for_mogp


def _train_position_models(X_train, Y_train_linear, config):
    """训练峰位多项式模型"""
    degree = config.get("POSITION_POLY_DEGREE", 3)
    conc_cols_indices = config.get("CONCENTRATION_COLUMNS_TO_USE", [0])
    # 仅使用第一维浓度进行位置拟合 (假设位置主要受主浓度影响)
    x_input = X_train[:, [conc_cols_indices[0]]]

    num_peaks = Y_train_linear.shape[1]
    poly_coeffs = []

    print(f"\n--- 训练峰位多项式模型 (Deg {degree}) ---")
    for i in range(num_peaks):
        try:
            coeffs = np.polyfit(x_input.flatten(), Y_train_linear[:, i], degree)
            poly_coeffs.append(coeffs)
        except:
            poly_coeffs.append([np.nan] * (degree + 1))
    return np.array(poly_coeffs)


def _train_mogp_lmc_model(X_train, Y_train, num_latent_processes, config, model_name="Params"):
    """
    训练 MOGP 模型 (升级版：支持 RANSAC)
    model_name: 'Params' 或 'Area'
    """
    conc_cols_indices = config.get("CONCENTRATION_COLUMNS_TO_USE", [0])
    max_iters = config.get("max_iters", 1000)

    # 判断是否启用 RANSAC
    use_ransac = False
    if model_name == "Area":
        use_ransac = config.get("USE_RANSAC_FOR_AREA", False)
    elif model_name == "Params":
        use_ransac = config.get("USE_RANSAC_FOR_PARAMS", False)

    print(f"\n--- 正在为 {model_name} 训练 MOGP-LMC 模型 (RANSAC={use_ransac}) ---")

    if Y_train is None or Y_train.size == 0:
        return None, 0

    x_input = X_train[:, conc_cols_indices]
    input_dim = x_input.shape[1]
    num_outputs = Y_train.shape[1]

    X_list = []
    Y_list = []

    # 准备数据，可选应用 RANSAC
    for i in range(num_outputs):
        y_col = Y_train[:, i:i + 1]
        x_col = x_input

        if use_ransac:
            try:
                # 使用 GPR 作为 RANSAC 的基模型
                kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                base_estimator = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)

                min_samples = config.get("RANSAC_MIN_SAMPLES", 7)
                res_thresh = config.get("RANSAC_RESIDUAL_THRESHOLD", None)

                if min_samples > x_col.shape[0]:
                    min_samples = max(2, int(x_col.shape[0] * 0.5))

                ransac = RANSACRegressor(
                    estimator=base_estimator,
                    min_samples=min_samples,
                    residual_threshold=res_thresh,
                    random_state=42
                )
                ransac.fit(x_col, y_col.flatten())
                inlier_mask = ransac.inlier_mask_

                n_out = np.sum(~inlier_mask)
                if n_out > 0:
                    print(f"  > [{model_name} Dim {i}] RANSAC 剔除了 {n_out} 个异常点")
                    x_col = x_col[inlier_mask]
                    y_col = y_col[inlier_mask]
            except Exception as e:
                print(f"  ! [{model_name} Dim {i}] RANSAC 失败，使用全部数据: {e}")

        X_list.append(x_col)
        Y_list.append(y_col)

    # 训练 GPy 模型
    try:
        kernel_list = [GPy.kern.RBF(input_dim, ARD=True, name=f'rbf_{i}') for i in range(num_outputs)]
        lmc_kernel = GPy.util.multioutput.LCM(input_dim=input_dim, num_outputs=num_outputs,
                                              kernels_list=kernel_list, W_rank=num_latent_processes)
        model = GPy.models.GPCoregionalizedRegression(X_list, Y_list, kernel=lmc_kernel)
        model.optimize(messages=True, max_iters=max_iters)
        return model, num_outputs
    except Exception as e:
        print(f"GPy 模型训练失败: {e}")
        # traceback.print_exc()
        raise


# =============================================================================
# 4. 生成与优化逻辑
# =============================================================================

def _generate_positions_from_models(concentrations, poly_coeffs, config):
    """从多项式模型生成峰位"""
    conc_idx = config.get("CONCENTRATION_COLUMNS_TO_USE", [0])[0]
    x_gen = concentrations[:, conc_idx]

    num_peaks = poly_coeffs.shape[0]
    generated_positions = np.zeros((len(x_gen), num_peaks))

    for i in range(num_peaks):
        if np.isnan(poly_coeffs[i]).any():
            generated_positions[:, i] = np.nan
        else:
            generated_positions[:, i] = np.polyval(poly_coeffs[i], x_gen)
    return generated_positions


def _generate_mogp_samples(model, concentrations_to_gen, num_outputs, config, params_for_mogp, num_peaks):
    """从 MOGP 模型采样参数"""
    if model is None:
        return np.zeros((len(concentrations_to_gen), num_outputs)) * np.nan

    conc_indices = config.get("CONCENTRATION_COLUMNS_TO_USE", [0])
    pred_concs = concentrations_to_gen[:, conc_indices]
    N = pred_concs.shape[0]

    # 构建 GPy 输入
    X_pred_stacked = np.hstack([
        np.repeat(pred_concs, num_outputs, axis=0),
        np.tile(np.arange(num_outputs), N).reshape(-1, 1)
    ])

    Y_metadata = {'output_index': X_pred_stacked[:, -1].reshape(-1, 1).astype(int)}

    try:
        # 获取完整协方差进行采样以保留相关性
        mean_pred, cov_pred = model.predict(X_pred_stacked, Y_metadata=Y_metadata, full_cov=True)
        mean_pred = mean_pred.reshape(N, num_outputs)

        generated_samples = []
        for i in range(N):
            start = i * num_outputs
            end = start + num_outputs
            cov_sample = cov_pred[start:end, start:end]

            # 修正协方差矩阵的正定性
            cov_sample = (cov_sample + cov_sample.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(cov_sample))
            if min_eig < 1e-6:
                cov_sample += (1e-6 - min_eig) * np.eye(num_outputs)

            try:
                sample = np.random.multivariate_normal(mean_pred[i], cov_sample)
                # 简单非负约束 (针对 Amplitude, Sigma 等)
                sample = np.maximum(sample, 1e-6)  # 避免 0 或负数
                generated_samples.append(sample)
            except:
                generated_samples.append(mean_pred[i])  # 采样失败回退到均值

        return np.array(generated_samples)

    except Exception as e:
        print(f"MOGP 预测失败: {e}")
        return np.zeros((N, num_outputs)) * np.nan


def optimize_params_with_integration_constraint(
        generated_params_dict,
        generated_pos_flat,
        generated_area_target,
        peak_groups,
        peak_shape
):
    """
    (新增) 使用数值积分作为约束的优化函数
    """
    print("\n--- 执行积分面积约束优化 ---")
    num_samples = generated_params_dict['amp'].shape[0]

    # 复制初始参数
    optimized_sigma = generated_params_dict['sigma'].copy()
    optimized_gamma = generated_params_dict['gamma'].copy() if peak_shape == 'voigt' else None

    success_count = 0
    total_opt_tasks = num_samples * len(peak_groups)

    for i in tqdm(range(num_samples), desc="优化样本参数"):
        pos_current = generated_pos_flat[i]

        for group_name, group_info in peak_groups.items():
            peak_indices = group_info['indices']
            r_min, r_max = group_info['range']

            # 获取该样本该组的目标面积
            if group_name not in generated_area_target: continue
            target_area = generated_area_target[group_name][i]
            if target_area <= 1e-6: continue

            # 提取该组涉及的峰参数
            amps = generated_params_dict['amp'][i, peak_indices]
            pos_group = pos_current[peak_indices]
            sigmas_init = generated_params_dict['sigma'][i, peak_indices]

            if peak_shape == 'voigt':
                gammas_init = generated_params_dict['gamma'][i, peak_indices]
                # 优化变量: [s1, g1, s2, g2, ...]
                x0 = np.zeros(2 * len(peak_indices))
                bounds = []
                for k in range(len(peak_indices)):
                    x0[2 * k] = max(1e-5, sigmas_init[k])
                    x0[2 * k + 1] = max(1e-5, gammas_init[k])
                    bounds.extend([(1e-5, None), (1e-5, None)])

                args_obj = (amps, pos_group, sigmas_init, gammas_init, 'voigt')
                args_con = (amps, pos_group, target_area, 'voigt', r_min, r_max)

            else:  # gaussian
                # 优化变量: [s1, s2, ...]
                x0 = np.maximum(1e-5, sigmas_init)
                bounds = [(1e-5, None) for _ in range(len(peak_indices))]

                args_obj = (amps, pos_group, sigmas_init, None, 'gaussian')
                args_con = (amps, pos_group, target_area, 'gaussian', r_min, r_max)

            try:
                res = minimize(
                    objective_func, x0, args=args_obj,
                    method='SLSQP', bounds=bounds,
                    constraints={'type': 'eq', 'fun': constraint_func, 'args': args_con},
                    options={'ftol': 1e-4, 'disp': False, 'maxiter': 50}
                )

                if res.success:
                    success_count += 1
                    if peak_shape == 'voigt':
                        for k, p_idx in enumerate(peak_indices):
                            optimized_sigma[i, p_idx] = res.x[2 * k]
                            optimized_gamma[i, p_idx] = res.x[2 * k + 1]
                    else:
                        for k, p_idx in enumerate(peak_indices):
                            optimized_sigma[i, p_idx] = res.x[k]
            except Exception:
                pass

    print(f"优化完成 (成功率: {success_count}/{total_opt_tasks})")
    return optimized_sigma, optimized_gamma


# =============================================================================
# 5. 可视化函数
# =============================================================================

def plot_area_fit_results(area_model, X_train, Y_train_areas, X_gen, Y_gen_areas, group_names, config):
    """(新增) 绘制面积拟合结果图"""
    try:
        plt.figure()
        plt.close()
    except:
        return  # 无显示环境

    print(f"\n--- 生成面积拟合图 ---")
    num_groups = len(group_names)
    cols = 2
    rows = int(np.ceil(num_groups / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes = axes.flatten()

    conc_idx = config.get("CONCENTRATION_COLUMNS_TO_USE", [0])[0]
    x_train_plot = X_train[:, conc_idx]
    x_gen_plot = X_gen[:, conc_idx] if X_gen.size > 0 else None

    # 用于绘制曲线的网格
    x_grid = np.linspace(x_train_plot.min(), x_train_plot.max(), 200).reshape(-1, 1)

    for i, g_name in enumerate(group_names):
        if i >= len(axes): break
        ax = axes[i]

        # 1. 真实值
        ax.plot(x_train_plot, Y_train_areas[:, i], 'kD', label='真实积分面积')

        # 2. MOGP 预测
        if area_model:
            # 构造预测输入 [conc, output_index]
            Y_meta = {'output_index': np.full((len(x_grid), 1), i)}
            # 注意：如果有多维浓度，这里只可视化第一维，其他维取均值或0（这里简化为只可视化第一维）
            # 实际上 MOGP 输入应该是完整的。
            # 简单起见，我们构造完整的输入矩阵
            X_pred_grid = np.zeros((len(x_grid), X_train.shape[1]))  # 假设 X_train 只有浓度列
            X_pred_grid[:, 0] = x_grid.flatten()
            # 如果有更多维度，应填充均值
            if X_train.shape[1] > 1:
                X_pred_grid[:, 1:] = np.mean(X_train[:, 1:], axis=0)

            # 再次选择用户指定的列
            X_pred_input = X_pred_grid[:, config.get("CONCENTRATION_COLUMNS_TO_USE", [0])]

            X_pred_gpy = np.hstack([X_pred_input, Y_meta['output_index']])

            try:
                mean, var = area_model.predict(X_pred_gpy, Y_metadata=Y_meta)
                std = np.sqrt(np.maximum(var, 0))
                ax.plot(x_grid, mean, 'm-', label='MOGP 预测')
                ax.fill_between(x_grid.flatten(), (mean - 2 * std).flatten(), (mean + 2 * std).flatten(), color='m',
                                alpha=0.2)
            except:
                pass

        # 3. 生成目标
        if x_gen_plot is not None and Y_gen_areas is not None:
            ax.plot(x_gen_plot, Y_gen_areas[:, i], 'mo', label='生成目标')

        ax.set_title(g_name)
        if i == 0: ax.legend()

    plt.tight_layout()

    # 保存
    output_dir = "augmented_output"
    os.makedirs(output_dir, exist_ok=True)
    prefix = config.get("OUTPUT_PREFIX_FINAL", "mogp")
    save_path = os.path.join(output_dir, f"{prefix}_area_fit.png")
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"面积图表已保存至: {save_path}")
    except:
        pass
    plt.show()


def plot_mogp_fit_results(mogp_model, position_models, X_train, Y_train_linear, Y_train_mogp,
                          X_gen, Y_gen_linear, Y_gen_mogp,
                          num_peaks, params_per_peak, config,
                          global_shared_params, params_for_mogp):
    """(保持原有逻辑，稍作适配) 绘制参数拟合图"""
    try:
        plt.figure()
        plt.close()
    except:
        return

    print(f"\n--- 生成参数拟合图 ---")
    conc_idx = config.get("CONCENTRATION_COLUMNS_TO_USE", [0])[0]

    param_types = ['Center (cm)'] + params_for_mogp
    num_params = len(param_types)

    fig, axes = plt.subplots(num_peaks, num_params, figsize=(5 * num_params, 4 * num_peaks), squeeze=False)

    x_train_plot = X_train[:, conc_idx]
    x_gen_plot = X_gen[:, conc_idx] if X_gen.size > 0 else None
    x_grid = np.linspace(x_train_plot.min(), x_train_plot.max(), 200)

    # 构建 GPy 预测网格 (类似 Area plot 的逻辑)
    X_pred_grid = np.zeros((len(x_grid), X_train.shape[1]))
    X_pred_grid[:, 0] = x_grid
    if X_train.shape[1] > 1: X_pred_grid[:, 1:] = np.mean(X_train[:, 1:], axis=0)
    X_pred_input = X_pred_grid[:, config.get("CONCENTRATION_COLUMNS_TO_USE", [0])]

    for i in range(num_peaks):
        for j, p_type in enumerate(param_types):
            ax = axes[i, j]

            # A. 峰位
            if p_type == 'Center (cm)':
                ax.plot(x_train_plot, Y_train_linear[:, i], 'kx', label='Train')
                # 多项式拟合曲线
                y_poly = np.polyval(position_models[i], x_grid)
                ax.plot(x_grid, y_poly, 'b-', label='Poly Fit')
                if x_gen_plot is not None:
                    ax.plot(x_gen_plot, Y_gen_linear[:, i], 'ro', label='Gen')

            # B. MOGP 参数
            else:
                try:
                    p_idx_rel = params_for_mogp.index(p_type)
                    p_idx_abs = p_idx_rel * num_peaks + i

                    ax.plot(x_train_plot, Y_train_mogp[:, p_idx_abs], 'kx', label='Train')

                    if mogp_model:
                        Y_meta = {'output_index': np.full((len(x_grid), 1), p_idx_abs)}
                        X_pred = np.hstack([X_pred_input, Y_meta['output_index']])
                        mean, var = mogp_model.predict_noiseless(X_pred, Y_metadata=Y_meta)
                        std = np.sqrt(np.maximum(var, 0))

                        ax.plot(x_grid, mean, 'b-', label='GPR')
                        ax.fill_between(x_grid, (mean - 2 * std).flatten(), (mean + 2 * std).flatten(), color='b',
                                        alpha=0.15)

                    if x_gen_plot is not None and Y_gen_mogp.size > 0:
                        ax.plot(x_gen_plot, Y_gen_mogp[:, p_idx_abs], 'ro', label='Gen')
                except:
                    pass

            ax.set_title(f"Peak {i + 1} {p_type}")
            if i == 0 and j == 0: ax.legend()

    plt.tight_layout()
    output_dir = "augmented_output"
    prefix = config.get("OUTPUT_PREFIX_FINAL", "mogp")
    save_path = os.path.join(output_dir, f"{prefix}_params_fit.png")
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"参数图表已保存至: {save_path}")
    except:
        pass
    plt.show()


# =============================================================================
# 6. 主流程函数
# =============================================================================

def train_and_run_mogp_generation(project: SpectralProject,
                                  m4_fit_results_all: Dict[str, Dict[int, pd.DataFrame]],
                                  config: Dict[str, Any]) \
        -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, Dict, Dict[int, pd.DataFrame]]:
    """
    执行 MOGP 参数化生成 (完整流程)
    """

    # 1. 准备训练数据
    try:
        X_train, Y_train_linear, Y_train_mogp, num_peaks, params_per_peak, peak_shape, \
            global_shared_params, params_for_mogp = _prepare_mogp_training_data(project, m4_fit_results_all, config)
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"数据准备失败: {e}")

    # --- 新增: 准备面积训练数据 (如果启用) ---
    apply_area_constraint = config.get("APPLY_AREA_CONSTRAINT", False)
    area_model = None
    peak_groups = {}
    group_names = []

    if apply_area_constraint:
        print("\n--- 正在计算训练集积分面积... ---")
        try:
            # 分组
            mean_positions = np.mean(Y_train_linear, axis=0)
            ranges = config.get("PEAK_GROUP_RANGES", [])
            if not ranges:
                print("警告: 启用了面积约束但未提供分组范围，将跳过。")
                apply_area_constraint = False
            else:
                peak_groups = get_peak_groups_by_range(mean_positions, ranges)
                group_names = sorted(list(peak_groups.keys()))

                if not group_names:
                    print("警告: 未能根据范围匹配到任何峰。")
                    apply_area_constraint = False
        except:
            apply_area_constraint = False

        # 计算面积矩阵
        if apply_area_constraint:
            Y_train_areas = np.zeros((X_train.shape[0], len(group_names)))

            # 需要完整的参数来计算面积。我们需要把 Y_train_mogp 拆解回去。
            # 映射: param_name -> index in params_for_mogp list
            p_map = {p: i for i, p in enumerate(params_for_mogp)}

            for i in range(X_train.shape[0]):
                for g_idx, g_name in enumerate(group_names):
                    indices = peak_groups[g_name]['indices']
                    r_min, r_max = peak_groups[g_name]['range']

                    area_sum = 0
                    for p_idx in indices:
                        pos = Y_train_linear[i, p_idx]

                        # Amp
                        if 'Amplitude' in p_map:
                            amp_idx = p_idx + p_map['Amplitude'] * num_peaks
                            amp = Y_train_mogp[i, amp_idx]
                        else:
                            amp = 0

                        # Sigma
                        if 'Sigma (cm)' in p_map:
                            sig_idx = p_idx + p_map['Sigma (cm)'] * num_peaks
                            sigma = Y_train_mogp[i, sig_idx]
                        elif global_shared_params is not None:
                            sigma = global_shared_params[0]
                        else:
                            sigma = 1.0

                        # Gamma
                        gamma = None
                        if peak_shape == 'voigt':
                            if 'Gamma (cm)' in p_map:
                                gam_idx = p_idx + p_map['Gamma (cm)'] * num_peaks
                                gamma = Y_train_mogp[i, gam_idx]
                            elif global_shared_params is not None and len(global_shared_params) > 1:
                                gamma = global_shared_params[1]
                            else:
                                gamma = 1.0

                        area_sum += integrate_peak_area(peak_shape, r_min, r_max, amp, pos, sigma, gamma)

                    Y_train_areas[i, g_idx] = area_sum

    # 2. 训练模型
    num_latent = config.get("NUM_LATENT_PROCESSES", 10)

    # A. 位置模型
    pos_models = _train_position_models(X_train, Y_train_linear, config)

    # B. 参数模型 (Params)
    mogp_model, num_out_params = _train_mogp_lmc_model(
        X_train, Y_train_mogp, num_latent, config, model_name="Params"
    )

    # C. 面积模型 (Area)
    if apply_area_constraint:
        area_model, num_out_area = _train_mogp_lmc_model(
            X_train, Y_train_areas, min(5, len(group_names)), config, model_name="Area"
        )

    # 3. 生成数据
    target_concs = prepare_target_concentrations(X_train, config)
    if target_concs.size == 0:
        return np.array([]), pd.DataFrame(), project.wavelengths, {}, {}

    print("\n--- 生成预测参数 ---")
    gen_pos = _generate_positions_from_models(target_concs, pos_models, config)
    gen_params_mogp = _generate_mogp_samples(mogp_model, target_concs, num_out_params, config, params_for_mogp,
                                             num_peaks)

    gen_areas = None
    if apply_area_constraint and area_model:
        print("--- 生成目标面积 ---")
        gen_areas = _generate_mogp_samples(area_model, target_concs, num_out_area, config, [], 0)

    # 4. 优化 (如果有面积约束)
    # 我们需要将 flat 的 mogp 参数解构为 dict 形式 {'amp': ..., 'sigma': ...} 供优化器使用
    p_map = {p: i for i, p in enumerate(params_for_mogp)}

    # 提取辅助函数
    def extract_gen_matrix(name):
        if name in p_map:
            start = p_map[name] * num_peaks
            return gen_params_mogp[:, start: start + num_peaks].copy()
        return None

    gen_struct = {
        'amp': extract_gen_matrix('Amplitude'),
        'sigma': extract_gen_matrix('Sigma (cm)'),
        'gamma': extract_gen_matrix('Gamma (cm)') if peak_shape == 'voigt' else None
    }

    # 填充共享参数或缺失参数
    N_gen = len(target_concs)
    if gen_struct['sigma'] is None:
        val = global_shared_params[0] if global_shared_params is not None else 1.0
        gen_struct['sigma'] = np.full((N_gen, num_peaks), val)

    if peak_shape == 'voigt' and gen_struct['gamma'] is None:
        val = global_shared_params[1] if global_shared_params is not None and len(global_shared_params) > 1 else 1.0
        gen_struct['gamma'] = np.full((N_gen, num_peaks), val)

    # 运行优化
    if apply_area_constraint and gen_areas is not None:
        area_dict = {name: gen_areas[:, i] for i, name in enumerate(group_names)}
        opt_sigma, opt_gamma = optimize_params_with_integration_constraint(
            gen_struct, gen_pos, area_dict, peak_groups, peak_shape
        )
        # 使用优化后的值
        final_sigma = opt_sigma
        final_gamma = opt_gamma
    else:
        final_sigma = gen_struct['sigma']
        final_gamma = gen_struct['gamma']

    # 5. 可视化
    try:
        # 重组优化后的参数用于绘图 (仅为了绘图一致性，替换掉原 MOGP 输出中的 sigma/gamma)
        # 注意：这只是用于 display，不会修改原始 gen_params_mogp 数组结构
        plot_gen_mogp = gen_params_mogp.copy()

        # 将优化后的 sigma 写回
        if 'Sigma (cm)' in p_map:
            s_start = p_map['Sigma (cm)'] * num_peaks
            plot_gen_mogp[:, s_start: s_start + num_peaks] = final_sigma

        if peak_shape == 'voigt' and 'Gamma (cm)' in p_map:
            g_start = p_map['Gamma (cm)'] * num_peaks
            plot_gen_mogp[:, g_start: g_start + num_peaks] = final_gamma

        plot_mogp_fit_results(mogp_model, pos_models, X_train, Y_train_linear, Y_train_mogp,
                              target_concs, gen_pos, plot_gen_mogp,
                              num_peaks, params_per_peak, config, global_shared_params, params_for_mogp)

        if apply_area_constraint and gen_areas is not None:
            plot_area_fit_results(area_model, X_train, Y_train_areas, target_concs, gen_areas, group_names, config)

    except Exception as e:
        print(f"可视化失败: {e}")
        traceback.print_exc()

    # 6. 重建光谱
    print("\n--- 重建光谱 ---")
    generated_spectra = []
    generated_spectra_noiseless = []
    generated_labels = []
    temp_gen_params = {}

    m4_cols = get_param_names(peak_shape)

    # 残差准备
    add_residuals = config.get("ADD_RESIDUALS", False)
    residual_library, sorted_anchors = None, None
    interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)
    if add_residuals:
        res_dir = config.get("RESIDUALS_PATH_ABSOLUTE")
        if res_dir:
            try:
                residual_library, sorted_anchors = load_residual_library(res_dir, interp_dim)
            except:
                pass

    processed_count = 0
    for i in range(N_gen):
        conc = target_concs[i]

        # 构建参数 Series
        params_dict = {}
        for d in range(len(conc)): params_dict[f'conc_{d}'] = conc[d]

        # 存储矩阵
        storage_mat = np.zeros((num_peaks, params_per_peak))

        valid = True
        for p in range(num_peaks):
            p_pos = gen_pos[i, p]
            if np.isnan(p_pos):
                valid = False;
                break

            p_amp = gen_struct['amp'][i, p]  # 始终存在
            p_sig = final_sigma[i, p]

            params_dict[f'peak_{p + 1}_pos'] = p_pos
            params_dict[f'peak_{p + 1}_height'] = p_amp
            params_dict[f'peak_{p + 1}_sigma'] = p_sig

            storage_mat[p, 0] = p_pos  # Center
            storage_mat[p, 1] = p_amp  # Amp
            storage_mat[p, 2] = p_sig  # Sigma

            if peak_shape == 'voigt':
                p_gam = final_gamma[i, p]
                params_dict[f'peak_{p + 1}_gamma'] = p_gam
                storage_mat[p, 3] = p_gam  # Gamma

        if not valid: continue

        # 重建
        try:
            spec = reconstruct_spectrum_from_params(project.wavelengths, pd.Series(params_dict))
            generated_spectra_noiseless.append(spec)

            final_spec = spec
            if add_residuals and residual_library:
                res = get_interpolated_residual(tuple(conc), residual_library, sorted_anchors, interp_dim)
                if res is not None and len(res) == len(spec):
                    final_spec = spec + res

            generated_spectra.append(final_spec)
            generated_labels.append(conc)

            temp_gen_params[processed_count] = pd.DataFrame(storage_mat, columns=m4_cols)
            processed_count += 1

        except:
            continue

    if not generated_spectra:
        return np.array([]), pd.DataFrame(), project.wavelengths, {}, {}

    # 7. 返回
    gen_spectra_array = np.array(generated_spectra)
    gen_spectra_array_noiseless = np.array(generated_spectra_noiseless)

    conc_col_names = [f'conc_{k}' for k in range(X_train.shape[1])]
    gen_labels_df = pd.DataFrame(generated_labels, columns=conc_col_names)

    new_task_info = {}
    for c in conc_col_names: new_task_info[c] = {'role': 'target', 'type': 'regression'}

    if not add_residuals: gen_spectra_array = gen_spectra_array_noiseless

    print(f"--- MOGP 流程完成，生成 {len(generated_spectra)} 个样本 ---")
    return gen_spectra_array, gen_spectra_array_noiseless, gen_labels_df, project.wavelengths, new_task_info, temp_gen_params