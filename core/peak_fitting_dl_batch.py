# 文件名: core/peak_fitting_dl_batch.py
# 描述: (重构版) PyTorch 批量独立拟合器
# 这是一个“引擎”，由 main.py 调用。
# 它接收来自 M4 UI 的锚点，并动态构建 p0 矩阵。

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Tuple, Any

def gaussian_np(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

def multi_gaussian_np(x, *params):
    y_peaks = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp, cen, sig = params[i:i + 3]
        if amp > 0: y_peaks += gaussian_np(x, amp, cen, sig)
    return y_peaks

def pseudo_voigt_np(x, amplitude, center, sigma, gamma, eta):
    eta = np.clip(eta, 0.0, 1.0)
    gauss_part = np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    lorentz_part = (gamma ** 2 / ((x - center) ** 2 + gamma ** 2))
    return amplitude * ((1.0 - eta) * gauss_part + eta * lorentz_part)

def multi_pseudo_voigt_np(x, *params):
    y_peaks = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 5):
        if i + 4 < len(params):
            amp, cen, sig, gam, eta = params[i:i + 5]
            if amp > 0:
                y_peaks += pseudo_voigt_np(x, amp, cen, sig, gam, eta)
    return y_peaks

# --- GPU / CPU Device Setup ---
# (注意: 我们在 run_dl_batch_fit 内部选择设备，以便在需要时更容易调试)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. PyTorch 峰型函数 (保持不变) ---
def gaussian_torch(x, amplitude, center, sigma):
    sigma = torch.clamp(sigma, min=1e-6)
    return amplitude * torch.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def pseudo_voigt_torch(x, amplitude, center, sigma, gamma, eta):
    sigma = torch.clamp(sigma, min=1e-6)
    gamma = torch.clamp(gamma, min=1e-6)
    gauss_part = torch.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    lorentz_part = (gamma ** 2 / ((x - center) ** 2 + gamma ** 2))
    return amplitude * ((1.0 - eta) * gauss_part + eta * lorentz_part)


# --- 2. PyTorch 模型定义 (保持不变) ---
class BatchSpectralFitter(nn.Module):
    def __init__(self, num_spectra, num_peaks, initial_params_batch, peak_shape='voigt', device='cpu', config_dl: Dict[str, Any] = None):
        super().__init__()
        self.num_spectra = num_spectra
        self.num_peaks = num_peaks
        self.peak_shape = peak_shape.lower()
        self.device = device


        if config_dl is None: config_dl = {}
        self.min_sigma = config_dl.get('dl_min_sigma', 0.1)
        self.max_sigma = config_dl.get('dl_max_sigma', 60.0)
        self.min_gamma = config_dl.get('dl_min_gamma', 0.1)
        self.max_gamma = config_dl.get('dl_max_gamma', 60.0)
        print(f"[DL-Core] Sigma 边界: [{self.min_sigma}, {self.max_sigma}]")
        print(f"[DL-Core] Gamma 边界: [{self.min_gamma}, {self.max_gamma}]")

        if initial_params_batch.shape != (num_spectra, num_peaks, 5):
            raise ValueError(f"初始参数数组形状必须是 (num_spectra, num_peaks, 5)。"
                             f"得到: {initial_params_batch.shape}, 期望: {(num_spectra, num_peaks, 5)}")

        # 将 p0 移动到目标设备
        p0 = torch.tensor(initial_params_batch, dtype=torch.float32).to(self.device)

        self.log_amplitudes = nn.Parameter(torch.log(torch.clamp(p0[:, :, 0], min=1e-6)))
        self.centers = nn.Parameter(p0[:, :, 1])
        self.log_sigmas = nn.Parameter(torch.log(torch.clamp(p0[:, :, 2], min=1e-3)))

        if self.peak_shape == 'voigt':
            self.log_gammas = nn.Parameter(torch.log(torch.clamp(p0[:, :, 3], min=1e-3)))
            eta_clipped = torch.clamp(p0[:, :, 4], 1e-6, 1.0 - 1e-6)
            self.eta_logits = nn.Parameter(torch.log(eta_clipped / (1.0 - eta_clipped)))

        print(f"[DL-Core] BatchSpectralFitter 已初始化。设备: {self.device}")

    def forward(self, x_batch):
        # 确保 x_batch 也在正确的设备上
        x_for_broadcast = x_batch.to(self.device).unsqueeze(1)

        amplitudes = torch.exp(self.log_amplitudes).unsqueeze(2)
        centers = self.centers.unsqueeze(2)
        sigmas = torch.clamp(
                       torch.exp(self.log_sigmas),
                       min = self.min_sigma,
                   max = self.max_sigma).unsqueeze(2)

        if self.peak_shape == 'voigt':
            gammas = torch.clamp(
                             torch.exp(self.log_gammas),
                             min = self.min_gamma,
                          max = self.max_gamma  ).unsqueeze(2)
            etas = torch.sigmoid(self.eta_logits).unsqueeze(2)
            y_preds_all_peaks = pseudo_voigt_torch(x_for_broadcast, amplitudes, centers, sigmas, gammas, etas)
        else:  # gaussian
            # (高斯模式下也需要5个参数，只是最后两个未使用，以保持p0矩阵一致性)
            y_preds_all_peaks = gaussian_torch(x_for_broadcast, amplitudes, centers, sigmas)

        y_summed_spectra = torch.sum(y_preds_all_peaks, dim=1)
        return y_summed_spectra


# --- 3. (重构) 训练函数 (基本不变) ---
def _train_model(model, x_batch_tensor, y_true_batch, config_dl, log_callback=None):
    device = model.device
    if log_callback is None: log_callback = print

    log_callback("\n[DL-Core] --- 开始批量独立拟合 (PyTorch) ---")
    epochs = config_dl.get('dl_epochs', 5000)
    lr = config_dl.get('dl_learning_rate', 1e-3)
    print_interval = config_dl.get('dl_print_interval', 100)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    log_callback(f"[DL-Core] ... 优化器: Adam, LR: {lr}, Epochs: {epochs}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=max(500, epochs // 20))
    start_train_time = time.time()
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = max(1000, epochs // 10)  # 提前停止的耐心

    # 将目标Y值移动到设备
    y_true_batch = y_true_batch.to(device)

    use_l2_reg = config_dl.get('dl_use_l2_reg', False)
    lambda_l2 = config_dl.get('dl_l2_lambda', 1e-7)
    if use_l2_reg:
        log_callback(f"[DL-Core] ... 启用 L2 正则化 (Lambda: {lambda_l2:.2e})")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred_batch = model(x_batch_tensor)
        loss_matrix = criterion(y_pred_batch, y_true_batch)
        loss_per_spectrum = torch.mean(loss_matrix, dim=1)
        total_loss = torch.mean(loss_per_spectrum)

        # (可选的 L2 正则化)
        # l2_reg = torch.tensor(0., device=device)
        # lambda_l2 = 1e-7
        # for param in model.parameters():
        #     l2_reg += torch.norm(param, 2)
        # loss_with_reg = total_loss + lambda_l2 * l2_reg

        total_loss.backward()
        optimizer.step()
        current_loss_val = total_loss.item()

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_callback(f'[DL-Core] Epoch [{epoch + 1}/{epochs}], Loss: {current_loss_val:.8f}, LR: {current_lr:.1e}')

        scheduler.step(current_loss_val)

        if current_loss_val < best_loss - 1e-8:
            best_loss = current_loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            log_callback(f"[DL-Core] 提前停止: 损失在 {patience} 个周期内没有显著改善。")
            break

    log_callback(f"[DL-Core] --- 训练完成 (耗时: {time.time() - start_train_time:.2f} 秒) ---")
    log_callback(f"[DL-Core] 最佳平均损失 (MSE): {best_loss:.8f}")
    return best_loss


# --- 4. (新增) 核心入口函数 ---
def run_dl_batch_fit(
        x_axis_full: np.ndarray,
        y_batch_full: np.ndarray,
        indices_in_batch: List[int],
        peak_anchors: List[float],
        peak_shape: str,
        peak_regions: List[Tuple[float, float]],
        config_dl: Dict[str, Any],
        log_callback: Any  # (例如 main.py 的 self.add_to_log)
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, np.ndarray]]:
    """
    (新增) 这是从 main.py 调用的主入口点。
    它执行 DL 拟合的完整流程并返回结果。
    """

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log_callback(f"[DL-Core] 使用设备: {device}")

    try:
        # --- 1. 准备数据 ---
        num_spectra = y_batch_full.shape[0]  # N
        num_peaks = len(peak_anchors)  # M
        log_callback(f"[DL-Core] 准备 N={num_spectra} 个光谱, M={num_peaks} 个峰。")

        # 归一化 (我们必须在完整光谱上执行此操作)
        y_min_global = np.min(y_batch_full)
        y_max_global = np.max(y_batch_full)
        y_range_global = y_max_global - y_min_global
        if y_range_global < 1e-9: y_range_global = 1.0
        y_data_normalized_np = (y_batch_full - y_min_global) / y_range_global  # (N, P_full)

        log_callback(f"[DL-Core] 全局强度范围: Min={y_min_global:.2f}, Max={y_max_global:.2f}")

        # --- 2. 动态生成 (N, M, 5) 初始猜测值矩阵 ---
        p0_batch_numpy = np.zeros((num_spectra, num_peaks, 5), dtype=float)

        # 从 M4 UI 获取默认值
        default_sigma = config_dl.get('default_sigma', 5.0)
        default_gamma = config_dl.get('default_gamma', 5.0)
        default_eta = config_dl.get('default_eta', 0.5)

        p0_batch_numpy[:, :, 2] = default_sigma
        p0_batch_numpy[:, :, 3] = default_gamma
        p0_batch_numpy[:, :, 4] = default_eta

        for i in range(num_spectra):  # 遍历 N
            for j in range(num_peaks):  # 遍历 M
                anchor = peak_anchors[j]
                idx = np.abs(x_axis_full - anchor).argmin()
                height_norm = y_data_normalized_np[i, idx]

                p0_batch_numpy[i, j, 0] = height_norm if height_norm > 1e-6 else 1e-6  # 振幅
                p0_batch_numpy[i, j, 1] = anchor  # 中心

        log_callback(f"[DL-Core] 成功生成 (N, M, 5) = ({num_spectra}, {num_peaks}, 5) 初始猜测值矩阵。")

        # --- 3. 准备 PyTorch 张量 (仅限拟合区域) ---
        mask_np = np.zeros_like(x_axis_full, dtype=bool)
        for start, end in peak_regions:
            mask_np |= (x_axis_full >= start) & (x_axis_full <= end)

        x_axis_fit_region_np = x_axis_full[mask_np]  # (P_fit,)
        if len(x_axis_fit_region_np) == 0:
            raise ValueError("'peak_regions' 内没有数据点。")

        # (N, P_fit)
        x_batch_tensor = torch.tensor(
            np.tile(x_axis_fit_region_np, (num_spectra, 1)),
            dtype=torch.float32
        ).to(device)

        # (N, P_fit)
        y_true_batch_normalized = torch.tensor(
            y_data_normalized_np[:, mask_np],
            dtype=torch.float32
        ).to(device)

        log_callback(f"[DL-Core] 拟合点数 (P_fit) = {len(x_axis_fit_region_np)}")

        # --- 4. 初始化模型 ---
        model = BatchSpectralFitter(
            num_spectra=num_spectra,
            num_peaks=num_peaks,
            initial_params_batch=p0_batch_numpy,
            peak_shape=peak_shape,
            device=device,
            config_dl = config_dl
        ).to(device)

        # --- 5. 训练模型 ---
        _train_model(model, x_batch_tensor, y_true_batch_normalized, config_dl, log_callback)

        # --- 6. 提取结果 (反归一化) ---
        log_callback("[DL-Core] 训练完成。正在提取和反归一化参数...")
        model.eval()

        # 从模型中获取优化的参数 (N, M)
        opt_amps_norm = torch.exp(model.log_amplitudes).detach().cpu().numpy()
        opt_centers = model.centers.detach().cpu().numpy()
        opt_sigmas = torch.exp(model.log_sigmas).detach().cpu().numpy()

        # 反归一化振幅
        opt_amps_raw = opt_amps_norm * y_range_global  # 注意：这里不加 y_min，因为振幅是相对基线的

        if peak_shape == 'voigt':
            opt_gammas = torch.exp(model.log_gammas).detach().cpu().numpy()
            opt_etas = torch.sigmoid(model.eta_logits).detach().cpu().numpy()
            columns = ['Amplitude', 'Center (cm)', 'Sigma (cm)', 'Gamma (cm)', 'Eta']
            fit_function_np = multi_pseudo_voigt_np
            # params_per_peak = 5
            # # (导入 NumPy/SciPy 版本用于重建)
            # from .peak_fitting_core import reconstruct_fit_from_popt
        else:  # gaussian
            opt_gammas = np.zeros_like(opt_sigmas)
            opt_etas = np.zeros_like(opt_sigmas)
            columns = ['Amplitude', 'Center (cm)', 'Sigma (cm)']
            fit_function_np = multi_gaussian_np
            params_per_peak = 3
            from .peak_fitting_core import reconstruct_fit_from_popt

        # --- 7. 将结果格式化为 main.py 所需的字典 ---
        results_df_dict: Dict[int, pd.DataFrame] = {}
        fit_y_dict: Dict[int, np.ndarray] = {}

        for i in range(num_spectra):
            sample_index = indices_in_batch[i]  # 获取原始样本索引

            # 构建参数 DataFrame
            params_df = pd.DataFrame({
                'Amplitude': opt_amps_raw[i, :],
                'Center (cm)': opt_centers[i, :],
                'Sigma (cm)': opt_sigmas[i, :],
                'Gamma (cm)': opt_gammas[i, :],
                'Eta': opt_etas[i, :]
            })
            params_df = params_df[columns]
            results_df_dict[sample_index] = params_df

            # 重建拟合曲线 (使用完整的 x_axis)
            popt_flat = params_df.to_numpy().flatten()
            fit_y = fit_function_np(x_axis_full, *popt_flat)  # <--- 直接调用
            fit_y_dict[sample_index] = fit_y

        log_callback(f"[DL-Core] 成功格式化 {len(results_df_dict)} 个样本的结果。")
        return results_df_dict, fit_y_dict

    except Exception as e:
        log_callback(f"[DL-Core] 严重错误: {e}")
        import traceback
        log_callback(traceback.format_exc())
        return {}, {}