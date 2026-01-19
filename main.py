# -*- coding: utf-8 -*-
# 文件名: main.py
# 描述: 主应用程序入口
# (修改: M7/可视化页面逻辑重构，实现按需加载和解耦)

import sys
import os
import time
import traceback
import shutil
import re  # 确保 re 已导入 (M2 需要)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QSplitter, QTabWidget, QTextEdit, QStyle,
    QHBoxLayout, QComboBox, QPushButton, QDoubleSpinBox, QFormLayout,
    QSpinBox, QStackedWidget, QCheckBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QLineEdit,
    QHeaderView, QMessageBox, QListWidget, QListWidgetItem,
    QSizePolicy, QFileDialog, QToolBar,
    QTextEdit,QScrollArea,QInputDialog
)
from PyQt6.QtCore import (
    Qt, pyqtSlot as Slot, pyqtSignal,
    QObject, QRunnable, QThreadPool, QSize, QTimer,QMutex
)
from PyQt6.QtGui import QAction, QIcon, QColor, QActionGroup
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import json

try:
    from core.augmentation_utils import (
        prepare_target_concentrations,
        reconstruct_spectrum_from_params,
        load_residual_library,
        get_interpolated_residual,
        get_m4_input_data,
        find_adjacent_samples_for_params,
        load_m4_fit_results_from_csv  # <--- 新增导入
    )
except ImportError:
    # ... (处理导入错误) ...
    def load_m4_fit_results_from_csv(path): return {} # 占位符

# (靠近其他 core 导入)
try:
    from core.peak_fitting_dl_batch import run_dl_batch_fit

    peak_fitting_dl_available = True
except ImportError as e:
    print(f"警告: 无法导入 core.peak_fitting_dl_batch: {e}。DL拟合将不可用。")
    peak_fitting_dl_available = False


    # 占位符
    def run_dl_batch_fit(*args, **kwargs):
        log_cb = kwargs.get('log_callback', print)
        log_cb("错误: DL 拟合模块未加载。")
        return {}, {}

# --- 项目导入 ---
# (确保这些路径相对于 main.py 是正确的)
try:
    from gui.data_importer import DataImportWizard
    from gui.project_manager import ProjectManager
    from gui.plot_widget import PlotWidget, CustomPlotToolbar  # 导入自定义工具栏
    from core.data_model import SpectralProject

    from core.peak_fitting_core import reconstruct_fit_from_popt

    # M3
    from core.airpls_core import process_spectrum as process_airpls
    from core.msc_core import process_spectrum_msc
    from core.denoise_core import process_spectrum_denoise
    # M4
    from core.peak_fitting_core import perform_fit, find_auto_anchors, parse_popt_to_dataframe
    from core.peak_fitting_models import get_param_names

    imports_ok = True
except ImportError as e:
    print(f"错误: 导入 GUI/Core 模块失败: {e}。应用可能无法正常运行。")
    imports_ok = False



    # 定义虚拟类以便在导入失败时基础 UI 仍能启动
    class DataImportWizard:
        pass


    class ProjectManager(QWidget):
        sample_selected = pyqtSignal(int);
        get_checked_items_indices = lambda \
                self: [];
        get_current_selected_index = lambda self: -1;
        load_project_data = lambda s, p: None;
        table_view = QWidget();
        tree_view = QWidget()  # 添加虚拟 table/tree


    class PlotWidget(QWidget):
        anchor_added = pyqtSignal(float);
        region_defined = pyqtSignal(float, float);
        threshold_set = pyqtSignal(
            float);
        anchor_delete_requested = pyqtSignal(float);
        clear_plot = lambda self: None;
        plot_spectrum = lambda \
                s, p, i, **k: None;
        set_interaction_mode = lambda s, m: None;
        on_update_regions = lambda s, r: None;
        on_update_anchors = lambda \
                s, a: None;
        on_update_threshold_line = lambda s, v: None;
        plot_baseline_results = lambda s, *a, **k: None;
        plot_denoise_results = lambda \
                s, *a, **k: None;
        plot_peak_fitting_results = lambda s, *a, **k: None


    class SpectralProject:
        get_version_names = lambda s: ['original'];
        get_active_spectrum_by_index = lambda s, i: None;
        wavelengths = np.array(
            []);
        spectra_versions = {
            'original': np.array([])};
        labels_dataframe = pd.DataFrame();
        get_category_info = lambda s, i: (
            None, None);
        get_task_summary = lambda \
                s: 'unknown';
        active_spectra_version = 'original';
        get_active_spectra = lambda s: None;
        get_id_col = lambda \
                s: None;
        get_primary_target_col = lambda s: (None, None);
        add_spectra_version = lambda s, n, d, h: None;
        reset_to_original = lambda s: False


    def process_airpls(y, lambda_):
        return y, y


    def process_spectrum_msc(y, ref):
        return y, ref, None


    def process_spectrum_denoise(y, algo, **kw):
        return y, None

# 确保核心函数可访问
try:
    from core.peak_fitting_core import (
        perform_fit,
        find_auto_anchors,
        parse_popt_to_dataframe
    )

    peak_fitting_core_available = True
except ImportError as e:
    print(f"错误: 导入 peak_fitting_core 失败: {e}。拟合将失败。")
    peak_fitting_core_available = False


    def perform_fit(x, y, a, c):
        return None, None, np.inf, "peak_fitting_core missing"


    def find_auto_anchors(x, y, m, c):
        return [], "peak_fitting_core missing"


    def parse_popt_to_dataframe(p, s):
        return pd.DataFrame()

try:
    from core.peak_fitting_models import get_param_names

    peak_fitting_models_available = True
except ImportError as e:
    print(f"错误: 导入 peak_fitting_models 失败: {e}。拟合可能失败。")
    peak_fitting_models_available = False


    def get_param_names(s):
        return []

try:
    from core.linear_spectrum_aug_core import run_linear_spectrum_interpolation
except ImportError:
    print("警告: 无法导入 core.linear_spectrum_aug_core.run_linear_spectrum_interpolation")


    def run_linear_spectrum_interpolation(project, config):  # Placeholder
        print("!! Placeholder: run_linear_spectrum_interpolation called")
        time.sleep(1)  # Simulate work
        input_version = config.get("INPUT_VERSION", 'original')
        spec = project.spectra_versions.get(input_version)
        if spec is None: raise ValueError("Input version not found")
        n_samples = 5
        # 确保返回正确的结构
        labels_subset = project.labels_dataframe.head(n_samples).copy()
        # 模拟生成不同标签值
        interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)
        col_name = labels_subset.columns[interp_dim]
        labels_subset[col_name] = np.linspace(labels_subset[col_name].min() * 0.8, labels_subset[col_name].max() * 1.2,
                                              n_samples)
        return (spec[:n_samples].copy() * np.random.uniform(0.9, 1.1, size=(n_samples, 1)),
                labels_subset,
                project.wavelengths,
                project.task_info)

try:
    from core.linear_param_aug_core import run_linear_parameter_interpolation
except ImportError:
    print("警告: 无法导入 core.linear_param_aug_core.run_linear_parameter_interpolation")


    def run_linear_parameter_interpolation(project, m4_fit_results_dict, config):  # Placeholder
        print("!! Placeholder: run_linear_parameter_interpolation called")
        time.sleep(1)  # Simulate work
        m4_version = config.get("M4_RESULTS_VERSION", 'original')
        source_spectra = project.spectra_versions.get(m4_version)
        if source_spectra is None: source_spectra = project.spectra_versions['original']
        n_samples = 5
        labels_subset = project.labels_dataframe.head(n_samples).copy()
        interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)
        col_name = labels_subset.columns[interp_dim]
        labels_subset[col_name] = np.linspace(labels_subset[col_name].min() * 0.8, labels_subset[col_name].max() * 1.2,
                                              n_samples)
        return (source_spectra[:n_samples].copy() * np.random.uniform(0.9, 1.1, size=(n_samples, 1)),
                labels_subset,
                project.wavelengths,
                project.task_info,
                {})  # 返回 5 个值

try:
    from core.mogp_aug_core import train_and_run_mogp_generation
    # (注意: GPy 库可能需要额外处理)
except ImportError as e:
    print(f"警告: 无法导入 core.mogp_aug_core.train_and_run_mogp_generation: {e}")


    def train_and_run_mogp_generation(project, m4_fit_results_dict, config):  # Placeholder
        print("!! Placeholder: train_and_run_mogp_generation called")
        time.sleep(1)  # Simulate work
        m4_version = config.get("M4_RESULTS_VERSION", 'original')
        source_spectra = project.spectra_versions.get(m4_version)
        if source_spectra is None: source_spectra = project.spectra_versions['original']
        n_samples = 5
        labels_subset = project.labels_dataframe.head(n_samples).copy()
        interp_dim = config.get("INTERPOLATION_DIMENSION_INDEX", 0)
        col_name = labels_subset.columns[interp_dim]
        labels_subset[col_name] = np.linspace(labels_subset[col_name].min() * 0.8, labels_subset[col_name].max() * 1.2,
                                              n_samples)
        return (source_spectra[:n_samples].copy() * np.random.uniform(0.9, 1.1, size=(n_samples, 1)),
                labels_subset,
                project.wavelengths,
                project.task_info,
                {})  # 返回 5 个值


def _load_m5_params_from_csv(csv_path: str) -> Dict[int, pd.DataFrame]:
    """
    (新增) 辅助函数：从 M5_GENERATED_PARAMS.csv 加载峰参数
    并将其转换回 worker 所需的 Dict[int, pd.DataFrame] 格式。
    """
    params_dict = {}
    if not os.path.exists(csv_path):
        return params_dict

    try:
        combined_df = pd.read_csv(csv_path)
        if 'sample_index' not in combined_df.columns:
            print(f"警告: M5 参数 CSV {csv_path} 缺少 'sample_index' 列。")
            return params_dict

        # 确定哪些是标签列 (这些列需要被移除)
        # (这是一个启发式方法：假设非数字、非振幅/中心的列是标签)
        known_param_cols = ['Amplitude', 'Center (cm)', 'Sigma (cm)', 'Gamma (cm)', 'Eta']
        label_cols = [col for col in combined_df.columns if col not in known_param_cols and col != 'sample_index']

        # 按 sample_index 分组
        for sample_index, group in combined_df.groupby('sample_index'):
            # 移除标签列和 sample_index 列
            params_only_df = group.drop(columns=label_cols + ['sample_index'], errors='ignore')
            # 重建索引 (e.g., "Peak 1", "Peak 2", ...)
            params_only_df.index = [f"Peak {i + 1}" for i in range(len(params_only_df))]
            params_dict[int(sample_index)] = params_only_df

    except Exception as e:
        print(f"!!! 错误: 加载 M5 参数缓存 {csv_path} 失败: {e}")
        traceback.print_exc()

    return params_dict


# --- Worker 类 (保持不变) ---
class WorkerSignals(QObject):
    """Worker 线程信号"""
    finished = pyqtSignal(dict);
    error = pyqtSignal(str)
    log = pyqtSignal(str)


class AutoFindWorker(QRunnable):
    """自动寻峰 Worker"""

    def __init__(self, x, y, m, c):
        super().__init__();
        self.s = WorkerSignals();
        self.x, self.y, self.m, self.c = x, y, m, c

    @Slot()
    def run(self):
        try:
            if not peak_fitting_core_available: raise ImportError("peak_fitting_core missing")
            from core.peak_fitting_core import find_auto_anchors  # 线程内重新导入
            a, e = find_auto_anchors(self.x, self.y, self.m, self.c);
            self.s.finished.emit({'auto_anchors': a, 'error_msg': e})
        except Exception as ex:
            traceback.print_exc();
            self.s.error.emit(f"AF Thr Fail: {ex}")


class FitWorker(QRunnable):
    """单样本拟合 Worker (预览)"""

    def __init__(self, x, y, a, c):
        super().__init__();
        self.s = WorkerSignals();
        self.x, self.y, self.a, self.c = x, y, a, c;
        self.ps = c.get(
            'peak_shape', 'voigt')

    @Slot()
    def run(self):
        try:
            if not peak_fitting_core_available: raise ImportError("peak_fitting_core missing")
            from core.peak_fitting_core import perform_fit  # 线程内重新导入
            p, fy, s, e = perform_fit(self.x, self.y, self.a, self.c);
            self.s.finished.emit({'popt': p, 'fit_y': fy, 'sse': s, 'error_msg': e, 'peak_shape': self.ps})
        except Exception as ex:
            traceback.print_exc();
            self.s.error.emit(f"Fit Thr Fail: {ex}")


class BatchFitWorker(QRunnable):
    """批量拟合 Worker (寻峰 + 拟合)"""

    def __init__(self, i, x, y, final_anchors, cfit):  # 不再需要 cf (find_config)
        super().__init__();
        self.s = WorkerSignals();
        # 存储 final_anchors
        self.i, self.x, self.y, self.final_anchors, self.cfit = i, x, y, final_anchors, cfit;
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    @Slot()
    def run(self):
        try:
            if not peak_fitting_core_available: raise ImportError("peak_fitting_core missing")
            # --- VVVV 修改: 移除 find_auto_anchors VVVV ---
            # 只导入 perform_fit 和 parse_popt_to_dataframe
            from core.peak_fitting_core import perform_fit, parse_popt_to_dataframe
            # --- ^^^^ 修改结束 ^^^^ ---

            if self.stop_flag: raise InterruptedError("Stopped")

            # --- VVVV 修改: 直接调用 perform_fit VVVV ---
            # 使用传入的 self.final_anchors
            p, fy, s, e = perform_fit(self.x, self.y, self.final_anchors, self.cfit)
            # --- ^^^^ 修改结束 ^^^^ ---

            if self.stop_flag: raise InterruptedError("Stopped")
            if e: raise RuntimeError(f"FitFail: {e}")
            # --- VVVV 修正: 检查 popt (p) 是否为 None VVVV ---
            if p is None: raise RuntimeError("FitFail: popt is None")
            # --- ^^^^ 修正结束 ^^^^ ---

            df = parse_popt_to_dataframe(p, self.cfit.get('peak_shape', 'voigt'))
            if df.empty and p is not None: raise RuntimeError("Parsing popt failed")

            # --- (发出信号的代码保持不变, fit_y (fy) 仍然需要) ---
            self.s.finished.emit({'sample_index': self.i, 'results_df': df, 'fit_y': fy, 'error': None})

        except InterruptedError:
            self.s.finished.emit(
                {'sample_index': self.i, 'results_df': None, 'fit_y': None, 'error': "Stopped by user"})
        except Exception as ex:
            # 打印更详细的错误追踪信息
            print(f"!!! BatchFitWorker Error for sample {self.i} !!!")
            traceback.print_exc()
            self.s.finished.emit({'sample_index': self.i, 'results_df': None, 'fit_y': None, 'error': str(ex)})


# (在 BatchFitWorker 类的定义之后)

# 文件：main.py

class DLBatchFitWorker(QRunnable):
    """
    (新增) 批量拟合 Worker (用于 PyTorch DL 引擎)
    它一次性运行所有N个样本，然后一次性返回所有结果。
    """

    def __init__(self,
                 indices_in_batch: List[int],
                 x_axis_full: np.ndarray,
                 y_batch_full: np.ndarray,
                 final_anchors: List[float],
                 cfg_fit: Dict[str, Any],
                 cfg_dl: Dict[str, Any]): # <--- [修改] 移除了 log_callback_signal 参数

        super().__init__()
        self.s = WorkerSignals()  # finished, error, log

        # 存储所有数据
        self.indices_in_batch = indices_in_batch
        self.x_axis_full = x_axis_full
        self.y_batch_full = y_batch_full
        self.final_anchors = final_anchors
        self.cfg_fit = cfg_fit
        self.cfg_dl = cfg_dl

    def _log_emitter(self, message: str):
        """
        传递给 run_dl_batch_fit 的回调函数。
        它将通过信号将消息发送回主线程。
        """
        self.s.log.emit(message) # <--- [修改] 使用信号发射

    @Slot()
    def run(self):
        try:
            if not peak_fitting_dl_available:
                raise ImportError("core.peak_fitting_dl_batch 模块不可用。")

            self.s.log.emit(f"[DL-Worker] 开始为 {len(self.indices_in_batch)} 个样本进行 DL 批量拟合...")

            # --- 1. 执行核心 DL 拟合 ---
            # (这会阻塞线程，直到训练完成)
            results_df_dict, fit_y_dict = run_dl_batch_fit(
                x_axis_full=self.x_axis_full,
                y_batch_full=self.y_batch_full,
                indices_in_batch=self.indices_in_batch,
                peak_anchors=self.final_anchors,
                peak_shape=self.cfg_fit.get('peak_shape', 'voigt'),
                peak_regions=self.cfg_fit.get('peak_regions', []),
                config_dl=self.cfg_dl,
                log_callback=self._log_emitter  # <--- [修改] 传递内部发射器
            )

            if not results_df_dict:
                raise RuntimeError("DL 拟合核心未返回任何结果。")

            self.s.log.emit(f"[DL-Worker] 拟合完成。正在发射 {len(results_df_dict)} 个样本的结果...")

            # --- 2. 逐个发射信号 ---
            for sample_index, df in results_df_dict.items():
                if sample_index not in self.indices_in_batch:
                    continue

                fit_y = fit_y_dict.get(sample_index)
                if fit_y is None:
                    continue

                self.s.finished.emit({
                    'sample_index': sample_index,
                    'results_df': df,
                    'fit_y': fit_y,
                    'error': None
                })

            self.s.log.emit("[DL-Worker] 所有信号发射完毕。")

        except Exception as ex:
            print(f"!!! DLBatchFitWorker Error !!!")
            traceback.print_exc()
            self.s.error.emit(f"DL 批量拟合失败: {ex}")




# (在 main.py 文件中，AugmentationWorker 类的内部)
class AugmentationWorker(QRunnable):
    """
    (修改) 在后台线程中执行数据增强（线性光谱、线性参数或 MOGP）。
    (修改) 现在将结果保存为完整的项目缓存结构，以便 Page 1 加载。
    """

    def __init__(self, mode, project, config,
                 m4_fit_results_all=None,augmented_output_root_path="augmented_output",
                 project_cache_root_path="project_cache"):  # m4_fit_results_all is Dict[str, Dict[int, pd.DataFrame]]
        super().__init__()
        self.s = WorkerSignals()
        self.mode = mode  # 'LinearSpectrum', 'LinearParam', 'MOGP'
        self.project = project  # 传入的是主项目对象
        self.config = config
        self.m4_fit_results_all = m4_fit_results_all  # 完整的 M4 结果字典
        self.augmented_output_root_path = augmented_output_root_path

    @Slot()
    def run(self):
        start_time = time.time()
        # 初始化所有可能的返回值
        gen_spectra, gen_labels, x_axis, task_info = None, None, None, None
        temp_gen_params = {}  # 初始化为空字典

        # 确定输出文件名前缀
        output_name_prefix = self.config.get("OUTPUT_PREFIX")
        if not output_name_prefix:  # 如果用户未指定，则自动生成
            mode_str_map = {'LinearSpectrum': 'LinSpec', 'LinearParam': 'LinParam', 'MOGP': 'MOGP'}
            mode_str = mode_str_map.get(self.mode, 'Aug')
            # 尝试获取源版本名
            source_ver = "unknown"
            if self.mode == 'LinearSpectrum':
                source_ver = self.config.get('INPUT_VERSION', 'unknown')
            elif self.mode in ['LinearParam', 'MOGP']:
                source_ver = self.config.get('M4_RESULTS_VERSION', 'unknown')
            source_ver = source_ver.replace('_fitted_curves', '')  # 清理名称
            output_name_prefix = f"{source_ver}_{mode_str}_aug_{int(start_time)}"
        # 确保 config 中有最终的前缀，以便历史记录
        self.config["OUTPUT_PREFIX_FINAL"] = output_name_prefix  # 使用新键存储最终前缀

        try:
            print(f"[Worker] Starting augmentation mode: {self.mode}")

            # --- 调用核心逻辑 ---
            core_result = None
            if self.mode == 'LinearSpectrum':
                core_result = run_linear_spectrum_interpolation(
                    self.project, self.config
                )
                # 解包 (4 个值)
                gen_spectra, gen_labels, x_axis, task_info = core_result
                gen_spectra_noiseless = gen_spectra  # 线性光谱模式没有残差，两者相同
                temp_gen_params = {}
                  # 线性光谱模式无峰参数

            elif self.mode == 'LinearParam':
                m4_version = self.config.get("M4_RESULTS_VERSION")
                if not m4_version: raise ValueError("LinearParam: M4_RESULTS_VERSION not specified in config")
                if not self.m4_fit_results_all: raise ValueError("LinearParam: M4 results dictionary is missing.")
                core_result = run_linear_parameter_interpolation(
                    self.project, self.m4_fit_results_all, self.config
                )
                # --- VVVV 修改：解包 5 个值 VVVV ---
                gen_spectra, gen_spectra_noiseless, gen_labels, x_axis, task_info, temp_gen_params = core_result
                # --- ^^^^ 修改结束 ^^^^ ---

            elif self.mode == 'MOGP':
                m4_version = self.config.get("M4_RESULTS_VERSION")
                if not m4_version: raise ValueError("MOGP: M4_RESULTS_VERSION not specified in config")
                if not self.m4_fit_results_all: raise ValueError("MOGP: M4 results dictionary is missing.")
                # GPy 训练可能非常耗时
                core_result = train_and_run_mogp_generation(
                    self.project, self.m4_fit_results_all, self.config
                )
                # --- VVVV 修改：解包 5 个值 VVVV ---
                gen_spectra, gen_spectra_noiseless, gen_labels, x_axis, task_info, temp_gen_params = core_result
                # --- ^^^^ 修改结束 ^^^^ ---

            else:
                raise ValueError(f"Unknown augmentation mode: {self.mode}")

            # --- 验证结果 ---
            if gen_spectra is None or gen_labels is None or x_axis is None or task_info is None or temp_gen_params is None:
                raise RuntimeError(
                    f"Core augmentation function for mode '{self.mode}' did not return all expected values.")
            if len(gen_spectra) != len(gen_labels):
                raise RuntimeError(f"Generated spectra count ({len(gen_spectra)}) != labels count ({len(gen_labels)})")

            print(f"[Worker] Core logic finished. Generated {len(gen_spectra)} samples.")

            # --- VVVV (!!!) (修改) 保存到缓存文件夹 VVVV (!!!) ---
            save_paths = {}

            # 1. (新增) 定义此增广项目唯一的缓存目录
            # (self.AUGMENTED_OUTPUT_ROOT 是 "augmented_output")
            project_cache_dir = os.path.join(self.augmented_output_root_path, output_name_prefix)
            spectra_cache_dir = os.path.join(project_cache_dir, "spectra")
            fit_cache_dir = os.path.join(project_cache_dir, "fit_results")  # (为MOGP/LinParam准备)

            os.makedirs(spectra_cache_dir, exist_ok=True)
            os.makedirs(fit_cache_dir, exist_ok=True)  # (创建空 M4 目录)
            print(f"[Worker] Augmentation cache directory created at: {project_cache_dir}")

            # 2. (新增) 保存核心文件 (模仿 load_project)
            initial_history = []  # 在 try 块外部定义
            try:
                # wavelengths.npy
                np.save(os.path.join(project_cache_dir, "wavelengths.npy"), x_axis)
                # labels.csv
                labels_path_on_disk = os.path.join(project_cache_dir, "labels.csv")
                gen_labels.to_csv(labels_path_on_disk, index=False)
                # metadata.json (task_info)
                import json
                meta_path = os.path.join(project_cache_dir, "metadata.json")
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=4)
                # processing_history.json (创建包含 M5 步骤的)
                history_path = os.path.join(project_cache_dir, "processing_history.json")
                history_list = []
                base_history = {'step': f'Augmentation ({self.mode})', 'config': self.config,
                                'indices_processed': list(range(len(gen_labels)))}

                # 条目 1: 最终版本 (带残差)
                history_list.append({**base_history, 'output_version': 'original'})

                # 条目 2: 纯净版本 (无残差)
                history_list.append({**base_history, 'step': f'ReconstructionOnly ({self.mode})',
                                     'output_version': 'reconstructed_noiseless'})

                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(history_list, f, ensure_ascii=False, indent=4)

                # (记录用于 new_project 对象的路径)
                save_paths['labels'] = os.path.abspath(labels_path_on_disk)
                print(f"[Worker] Core cache files (labels, wav, meta, history) saved.")

            except Exception as e_core_save:
                raise RuntimeError(f"Failed to save core cache files: {e_core_save}")

            # 3. (新增) 保存 'original' 光谱 (如果存在)
            if gen_spectra is not None and len(gen_spectra) > 0:
                spectra_path_on_disk = os.path.join(spectra_cache_dir, "original.npy")
                try:
                    np.save(spectra_path_on_disk, gen_spectra)
                    # (新增) 保存 "纯净" 光谱
                    spectra_path_noiseless_on_disk = os.path.join(spectra_cache_dir, "reconstructed_noiseless.npy")
                    np.save(spectra_path_noiseless_on_disk, gen_spectra_noiseless)
                    print(f"[Worker] 'reconstructed_noiseless' spectra saved to {spectra_path_noiseless_on_disk}")
                    save_paths['spectra'] = os.path.abspath(spectra_path_on_disk)  # (用于 new_project)
                    print(f"[Worker] 'original' spectra saved to {spectra_path_on_disk}")
                except Exception as e_spec_save:
                    raise RuntimeError(f"Failed to save spectra file '{spectra_path_on_disk}': {e_spec_save}")
            else:
                print("[Worker] No samples generated, skipping spectra/original.npy saving.")
                save_paths['spectra'] = "N/A (No samples generated)"

            # 4. (新增) 保存 MOGP/LinParam 生成的峰参数 (如果存在)
            if temp_gen_params:
                print(f"[Worker] Saving {len(temp_gen_params)} generated peak parameters...")
                try:
                    # (我们将其保存为 'original.csv'，因为它们对应 'original' 光谱)
                    fit_csv_path = os.path.join(fit_cache_dir, "M5_GENERATED_PARAMS.csv")
                    all_dfs = []
                    for idx, df in temp_gen_params.items():
                        if df is not None and not df.empty:
                            df_copy = df.copy();
                            df_copy['sample_index'] = idx
                            # (合并标签)
                            if not gen_labels.empty and idx < len(gen_labels):
                                label_info = gen_labels.iloc[idx]
                                for col in label_info.index:
                                    if col not in df_copy.columns: df_copy[col] = label_info[col]
                            all_dfs.append(df_copy)

                    if all_dfs:
                        combined_df = pd.concat(all_dfs, ignore_index=True)
                        # (重新排序)
                        cols = list(combined_df.columns);
                        new_order = ['sample_index']
                        label_cols_to_add = [c for c in gen_labels.columns if c in cols and c not in new_order]
                        new_order.extend(label_cols_to_add)
                        remaining_cols_to_add = [c for c in cols if c not in new_order]
                        new_order.extend(remaining_cols_to_add);
                        combined_df = combined_df[new_order]
                        combined_df.to_csv(fit_csv_path, index=False, encoding='utf-8-sig')
                        print(f"[Worker] Generated peak parameters saved to: {fit_csv_path}")
                except Exception as e_param_save:
                    print(f"Warning: Failed to save generated peak parameters to CSV: {e_param_save}")
            # --- ^^^^ (修改) 保存逻辑结束 ^^^^ ---

            new_spectra_versions = {}
            if gen_spectra is not None and len(gen_spectra) > 0:
                new_spectra_versions['original'] = gen_spectra.copy()
            if gen_spectra_noiseless is not None and len(gen_spectra_noiseless) > 0:
                new_spectra_versions['reconstructed_noiseless'] = gen_spectra_noiseless.copy()

            # --- 创建新项目对象 ---
            # (即使生成 0 个样本，也创建一个空项目对象)
            new_project = SpectralProject(
                wavelengths=x_axis.copy() if x_axis is not None else np.array([]),
                labels_dataframe=gen_labels.copy() if gen_labels is not None else pd.DataFrame(),
                task_info=task_info.copy() if task_info is not None else {},
                data_file_path=save_paths.get('spectra', "N/A"),
                label_file_path=save_paths.get('labels', "N/A"),
                spectra_versions={'original': gen_spectra.copy()} if gen_spectra is not None and len(
                    gen_spectra) > 0 else {},
                active_spectra_version='original' if gen_spectra is not None and len(gen_spectra) > 0 else '',
                processing_history=initial_history,  # <-- (修改) 传入我们刚创建的 history
                generated_peak_params=temp_gen_params.copy() if temp_gen_params is not None else {}  # <-- 传入新字段
            )
            print(f"[Worker] New SpectralProject object created.")

            # --- 发送结果 ---
            self.s.finished.emit({
                'new_project': new_project,
                'save_paths': save_paths,
                'aug_project_name': output_name_prefix  # 使用最终的项目名
            })
            print(f"[Worker] Finished successfully in {time.time() - start_time:.2f}s.")

        except Exception as ex:
            print(f"[Worker] Error during augmentation:")
            traceback.print_exc()
            # 尝试传递更具体的错误信息
            error_details = f"{type(ex).__name__}: {ex}"
            self.s.error.emit(f"增广失败 ({self.mode}):\n{error_details}")

# --- 交互模式常量 ---
MODE_DISABLED = 0;
MODE_PICK_ANCHOR = 1;
MODE_SELECT_REGION = 2;
MODE_THRESHOLD_LINE = 3


# --- 常量结束 ---

class MainWindow(QMainWindow):
    """主应用窗口"""
    # --- 信号 ---
    request_plot_interaction_mode = pyqtSignal(int)
    request_plot_region_update = pyqtSignal(list)
    request_plot_anchors_update = pyqtSignal(list)
    request_plot_threshold_update = pyqtSignal(float)

    PROJECT_CACHE_ROOT = "project_cache"
    AUGMENTED_OUTPUT_ROOT = "augmented_output"
    # --- 状态变量 ---
    m4_current_regions = []
    fitting_stop_requested = False
    active_workers = {}
    m4_batch_fit_results: Dict[str, Dict[int, pd.DataFrame]] = defaultdict(dict)
    m4_batch_counter = 0;
    m4_batch_total = 0;
    m4_batch_start_time = 0
    m4_batch_input_version = 'original'
    m4_current_output_name = ""

    def _get_project_cache_dir(self, project: SpectralProject = None):
        """
        (新增) 获取当前主项目的唯一缓存目录路径。
        如果 project 为 None, 则使用 self.current_project。
        """
        if project is None:
            project = self.current_project

        if not project:
            return None  # 如果没有项目，则没有路径

        # 基于项目的数据文件路径创建一个安全、唯一的文件夹名称
        base_name = os.path.basename(project.data_file_path)
        project_hash = re.sub(r'[^\w\d_.-]', '_', base_name)  # 替换不安全字符

        return os.path.join(self.PROJECT_CACHE_ROOT, project_hash)

    def __init__(self):
        super().__init__()
        # 基础窗口设置
        self.setWindowTitle("SpectraX")
        self.setGeometry(100, 100, 1200, 800)
        self.current_project: SpectralProject = None
        self.m4_temp_fit_curves: Dict[int, np.ndarray] = {}  # 重命名 (旧: m4_batch_fit_curves)
        self.m4_temp_fit_results: Dict[int, pd.DataFrame] = {}  # (新增) 临时结果
        self.vis_m4_results_cache: Dict[str, pd.DataFrame] = {}

        self.m2_peak_config: Dict[str, Dict] = {}
        self.current_pick_mode_owner = None  # 'M2' or 'M4'

        self.m4_batch_final_anchors: List[float] = []  # 存储用于批量的最终锚点
        self.m4_batch_cfg_fit: Dict[str, Any] = {}  # 存储拟合配置
        self.m4_batch_cfg_find: Dict[str, Any] = {}  # 存储寻峰配置 (如果使用了自动寻峰)

        # 1. 创建主堆叠控件
        self.main_stack = QStackedWidget()
        self.setCentralWidget(self.main_stack)  # 设置为中央控件

        # 2. 创建页面占位符
        self.page_0_workflow = QWidget()  # 页面 0: M1-M6 工作流
        self.page_1_visualization = QWidget()  # 页面 1: 可视化视图

        # 3. 添加页面到堆叠控件
        self.main_stack.addWidget(self.page_0_workflow)
        self.main_stack.addWidget(self.page_1_visualization)

        # 4. (新增) 存储可视化页面的控件引用
        # (新代码)
        self.vis_project_manager: ProjectManager | None = None
        self.vis_data_source_combo: QComboBox | None = None
        self.vis_show_components_cb: QCheckBox | None = None
        self.vis_plot_button: QPushButton | None = None
        self.vis_clear_button: QPushButton | None = None
        self.vis_preview_plot: PlotWidget | None = None  # 右上角预览画板
        self.vis_main_plot: PlotWidget | None = None  # 右下角主画板

        # 5. (新增) 存储工具栏动作
        self.workflow_action: QAction | None = None
        self.vis_action: QAction | None = None
        # 线程池
        self.thread_pool = QThreadPool()
        self.m4_batch_mutex = QMutex()
        print(f"多线程启动: 最大线程数 {self.thread_pool.maxThreadCount()}")
        # 构建 UI
        self._setup_menu()
        self._setup_toolbar()  # <-- 新增：设置工具栏
        self._setup_ui_modules()


        self._connect_signals()
        self.log_console.append("欢迎...")
        # 检查导入
        if not imports_ok or not peak_fitting_core_available or not peak_fitting_models_available:
            QMessageBox.critical(self, "依赖错误", "缺少必要的 GUI 或 Core 模块。请检查控制台。")

        if hasattr(self, 'main_stack'):
            try:
                self.main_stack.currentChanged.connect(self.on_main_stack_page_changed)
                print("[DEBUG Init Fix] Successfully connected main_stack.currentChanged at the end of __init__.")
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to connect main_stack.currentChanged at end of __init__: {e}")
        print(f"[DEBUG Init END] ID of self in __init__: {id(self)}")


    def _setup_menu(self):  # 简化
        menu = self.menuBar();
        file_menu = menu.addMenu("&文件")
        style = self.style();

        icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        self.import_action = QAction(icon, "&导入...", self);
        self.import_action.triggered.connect(self.open_import_wizard);
        file_menu.addAction(self.import_action)

        icon_m4 = style.standardIcon(QStyle.StandardPixmap.SP_ArrowRight)  # 或其他图标
        self.import_m4_csv_action = QAction(icon_m4, "导入 M4 拟合结果 (CSV)...", self)
        self.import_m4_csv_action.setToolTip("加载之前保存的拟合参数文件 (.csv)，以便在 M5 中使用")
        self.import_m4_csv_action.triggered.connect(self.on_import_m4_results_csv)
        file_menu.addAction(self.import_m4_csv_action)

        # --- VVVV 新增重置动作 VVVV ---
        file_menu.addSeparator()
        icon_reset = style.standardIcon(QStyle.StandardPixmap.SP_DialogResetButton)
        self.reset_action = QAction(icon_reset, "&重置项目", self);
        self.reset_action.setToolTip("清除所有已处理的数据版本和M4结果，恢复到 'original' 状态")
        self.reset_action.triggered.connect(self.on_reset_project)
        self.reset_action.setEnabled(False)  # 默认禁用
        file_menu.addAction(self.reset_action)
        # --- ^^^^ 新增结束 ^^^^ ---

        # (可以放在 _setup_menu 之后)

    def _setup_toolbar(self):
        """(新增) 创建顶部工具栏用于视图切换。"""
        toolbar = QToolBar("主视图切换")
        toolbar.setIconSize(QSize(24, 24))  # 设置图标大小
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)  # 添加到顶部

        style = self.style()

        # 动作组，确保按钮互斥
        action_group = QActionGroup(self)
        action_group.setExclusive(True)

        # 按钮 1: 工作流视图
        self.workflow_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "工作流视图",
                                       self)
        self.workflow_action.setToolTip("切换到主数据处理和增强工作流页面 (M1-M6)")
        self.workflow_action.setCheckable(True)
        self.workflow_action.setChecked(True)  # 默认选中
        toolbar.addAction(self.workflow_action)
        action_group.addAction(self.workflow_action)

        # 按钮 2: 可视化视图
        self.vis_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon), "可视化视图", self)
        self.vis_action.setToolTip("切换到专用的数据对比和可视化页面")
        self.vis_action.setCheckable(True)
        self.vis_action.setChecked(False)
        self.vis_action.setEnabled(False)  # 默认禁用
        toolbar.addAction(self.vis_action)
        action_group.addAction(self.vis_action)

        # (连接信号将在 _connect_signals 中统一连接)

    def _setup_ui_modules(self):
        # 1. 创建 M1-M6 的核心组件 (保持不变)
        self.project_manager = ProjectManager()  # 用于页面 0
        self.plot_widget = PlotWidget()  # 用于页面 0
        self.log_console = QTextEdit();
        self.log_console.setReadOnly(True)  # 用于页面 0
        self.processing_tabs = QTabWidget();  # 用于页面 0
        self.processing_tabs.currentChanged.connect(self.on_main_tab_changed)

        # 2. 设置 M2-M6 选项卡 (移除 M7 设置调用)
        self._setup_m2_label_engineering_tab()
        self._setup_m3_preprocessing_tab()
        self._setup_m4_peak_fitting_tab()
        self._setup_m5_augmentation_tab()
        self._setup_m6_export_tab()
        # (移除 M7 选项卡设置调用)

        # 3. 添加 M2-M6 选项卡到 QTabWidget (移除 M7)
        style = self.style();
        icons = [style.standardIcon(p) for p in
                 [
                     QStyle.StandardPixmap.SP_FileLinkIcon,  # M2
                     QStyle.StandardPixmap.SP_CustomBase,  # M3
                     QStyle.StandardPixmap.SP_ComputerIcon,  # M4
                     QStyle.StandardPixmap.SP_ArrowForward,  # M5
                     QStyle.StandardPixmap.SP_DialogSaveButton  # M6
                     # (移除 M7 图标)
                 ]]
        self.processing_tabs.addTab(self.tab_label_eng, icons[0], "标签工程 (M2)")
        self.processing_tabs.addTab(self.tab_preprocess, icons[1], "预处理 (M3)")
        self.processing_tabs.addTab(self.tab_peak_fitting, icons[2], "峰拟合 (M4)")
        self.processing_tabs.addTab(self.tab_augment, icons[3], "数据增强 (M5)")
        self.processing_tabs.addTab(self.tab_export, icons[4], "导出 (M6)")
        # (移除 addTab for M7)

        # (禁用 M2 的逻辑保持不变)
        self.processing_tabs.setTabEnabled(self.processing_tabs.indexOf(self.tab_label_eng), False)

        # --- VVVV 修改: 调用页面布局函数 VVVV ---
        # 4. 布局 页面 0 (工作流视图)
        self._setup_page_0_workflow()

        # 5. 布局 页面 1 (可视化视图)
        self._setup_page_1_visualization()
        # --- ^^^^ 修改结束 ^^^^ ---
    def _setup_m2_label_engineering_tab(self):
            """Builds the UI for the Label Engineering Tab (M2)."""
            self.tab_label_eng = QWidget()
            layout = QVBoxLayout(self.tab_label_eng)

            info_label = QLabel("此模块用于为分类任务创建伪标签，以便在 M5 中使用回归模型进行增强。\n"
                                "请为每个类别指定一个特征峰。")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)

            # 1. 类别配置总表
            config_group = QGroupBox("类别伪标签配置")
            config_layout = QVBoxLayout(config_group)
            self.m2_config_table = QTableWidget()
            self.m2_config_table.setColumnCount(5)
            self.m2_config_table.setHorizontalHeaderLabels(
                ["类别", "特征峰 (cm⁻¹)", "窗口 (± cm⁻¹)", "伪标签列名", "状态"])
            self.m2_config_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.m2_config_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.m2_config_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.m2_config_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.m2_config_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self.m2_config_table.setFixedHeight(150)
            config_layout.addWidget(self.m2_config_table)
            layout.addWidget(config_group)

            # 2. 交互设置
            setup_group = QGroupBox("交互设置")
            setup_layout = QFormLayout(setup_group)

            self.m2_class_selector = QComboBox()
            self.m2_class_selector.currentTextChanged.connect(self.on_m2_class_selector_changed)
            setup_layout.addRow("选择配置类别:", self.m2_class_selector)

            pick_layout = QHBoxLayout()
            self.m2_pick_peak_btn = QPushButton("在图上选择特征峰")
            self.m2_pick_peak_btn.setCheckable(True)
            self.m2_pick_peak_btn.toggled.connect(self.on_m2_pick_peak_toggled)
            self.m2_peak_pos_display = QLineEdit("[ 未选择 ]")
            self.m2_peak_pos_display.setReadOnly(True)
            pick_layout.addWidget(self.m2_pick_peak_btn)
            pick_layout.addWidget(self.m2_peak_pos_display)
            setup_layout.addRow(pick_layout)

            self.m2_peak_window_input = QDoubleSpinBox()
            self.m2_peak_window_input.setRange(1.0, 100.0);
            self.m2_peak_window_input.setValue(10.0)
            setup_layout.addRow("峰搜索窗口 (± cm⁻¹):", self.m2_peak_window_input)

            self.m2_apply_peak_config_btn = QPushButton("设置/更新此类别配置")
            setup_layout.addRow(self.m2_apply_peak_config_btn)
            layout.addWidget(setup_group)

            # 2.5 缩放
            scale_group = QGroupBox("伪标签数值缩放 (可选)")
            scale_layout = QFormLayout(scale_group)

            self.m2_scale_checkbox = QCheckBox("将峰高缩放到指定范围")
            self.m2_scale_checkbox.toggled.connect(self.on_m2_scale_checkbox_toggled)  # 连接槽函数
            scale_layout.addRow(self.m2_scale_checkbox)

            self.m2_scale_min_input = QDoubleSpinBox()
            self.m2_scale_min_input.setDecimals(4);
            self.m2_scale_min_input.setRange(-1e6, 1e6);
            self.m2_scale_min_input.setValue(0.0)
            self.m2_scale_min_input.setEnabled(False)  # 默认禁用
            scale_layout.addRow("缩放后最小值:", self.m2_scale_min_input)

            self.m2_scale_max_input = QDoubleSpinBox()
            self.m2_scale_max_input.setDecimals(4);
            self.m2_scale_max_input.setRange(-1e6, 1e6);
            self.m2_scale_max_input.setValue(1.0)
            self.m2_scale_max_input.setEnabled(False)  # 默认禁用
            scale_layout.addRow("缩放后最大值:", self.m2_scale_max_input)

            layout.addWidget(scale_group)  # 将新 GroupBox 添加到主布局

            # 3. 执行
            self.m2_run_generation_btn = QPushButton("生成/更新所有伪标签")
            self.m2_run_generation_btn.setStyleSheet("background-color:#DAF7A6;")
            layout.addWidget(self.m2_run_generation_btn)

            layout.addStretch()

    # --- VVVV M3 UI 修改 VVVV ---
    def _setup_m3_preprocessing_tab(self):
        """Builds the UI for the Preprocessing Tab (M3) with Preview and Apply buttons."""
        self.tab_preprocess = QWidget();
        layout = QVBoxLayout(self.tab_preprocess);
        layout.setContentsMargins(0, 0, 0, 0);
        self.preprocess_sub_tabs = QTabWidget();
        layout.addWidget(self.preprocess_sub_tabs)

        # --- 1. 基线 (Baseline) ---
        base_w = QWidget();
        base_l = QVBoxLayout(base_w);
        form = QFormLayout();
        self.algo_combo = QComboBox();
        self.algo_combo.addItems(["AirPLS", "MSC"]);
        form.addRow("算法:", self.algo_combo);
        base_l.addLayout(form);
        self.param_stack = QStackedWidget();
        base_l.addWidget(self.param_stack);
        # AirPLS Params
        air_w = QWidget();
        air_f = QFormLayout(air_w);
        self.lambda_input = QDoubleSpinBox();
        self.lambda_input.setDecimals(0);
        self.lambda_input.setRange(1., 1e9);
        self.lambda_input.setValue(10000);
        air_f.addRow("Lambda:", self.lambda_input);
        self.param_stack.addWidget(air_w);
        # MSC Params
        msc_w = QWidget();
        msc_f = QFormLayout(msc_w);
        self.msc_ref_combo = QComboBox();
        self.msc_ref_combo.addItems(["类别平均", "指定索引"]);
        msc_f.addRow("参考:", self.msc_ref_combo);
        self.msc_ref_index_input = QSpinBox();
        self.msc_ref_index_input.setRange(0, 99999);
        self.msc_ref_index_label = QLabel("索引:");
        self.msc_ref_index_label.setVisible(False);
        msc_f.addRow(self.msc_ref_index_label, self.msc_ref_index_input);
        self.msc_ref_index_input.setVisible(False);
        self.msc_ref_combo.currentTextChanged.connect(self.on_msc_ref_choice_changed);
        self.param_stack.addWidget(msc_w);
        self.algo_combo.currentTextChanged.connect(self.on_algo_changed);

        # --- 按钮修改 (基线) ---
        self.baseline_preview_button = QPushButton("预览当前")
        self.baseline_apply_button = QPushButton("应用到勾选项")  # 重命名
        # self.baseline_preview_button.clicked.connect(self.on_run_baseline_preview)  # 连接预览
        # self.baseline_apply_button.clicked.connect(self.on_run_baseline_batch)  # 连接到新的 batch 函数
        base_l.addWidget(self.baseline_preview_button)  # 添加新按钮
        base_l.addWidget(self.baseline_apply_button)  # 添加重命名的按钮
        base_l.addStretch()

        # --- 2. 降噪 (Denoise) ---
        den_w = QWidget();
        den_l = QVBoxLayout(den_w);
        form = QFormLayout();
        self.denoise_algo_combo = QComboBox();
        self.denoise_algo_combo.addItems(["Savitzky-Golay"]);
        form.addRow("算法:", self.denoise_algo_combo);
        # SG Params
        self.sg_window_input = QSpinBox();
        self.sg_window_input.setRange(3, 999);
        self.sg_window_input.setValue(15);
        self.sg_window_input.setSingleStep(2);
        form.addRow("窗口(奇):", self.sg_window_input);
        self.sg_order_input = QSpinBox();
        self.sg_order_input.setRange(0, 20);
        self.sg_order_input.setValue(3);
        form.addRow("阶数:", self.sg_order_input);
        den_l.addLayout(form);

        # --- 按钮修改 (降噪) ---
        self.denoise_preview_button = QPushButton("预览当前")
        self.denoise_apply_button = QPushButton("应用到勾选项")  # 重命名
        # self.denoise_preview_button.clicked.connect(self.on_run_denoise_preview)  # 连接预览
        # self.denoise_apply_button.clicked.connect(self.on_run_denoise_batch)  # 连接到新的 batch 函数
        den_l.addWidget(self.denoise_preview_button)  # 添加新按钮
        den_l.addWidget(self.denoise_apply_button)  # 添加重命名的按钮
        den_l.addStretch()

        # --- 3. 添加子 tabs ---
        style = self.style();
        self.preprocess_sub_tabs.addTab(base_w, style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown), "基线");
        self.preprocess_sub_tabs.addTab(den_w, style.standardIcon(QStyle.StandardPixmap.SP_MediaVolume), "降噪")

    # --- ^^^^ M3 UI 修改结束 ^^^^ ---

    def _setup_m4_peak_fitting_tab(self):  # (保持不变)
        self.tab_peak_fitting = QWidget();
        # --- VVVV 关键修改 VVVV ---
        # 2. (新增) 创建一个 QScrollArea
        scroll_area = QScrollArea(self.tab_peak_fitting)
        scroll_area.setWidgetResizable(True)  # 关键：允许内容随宽度缩放
        # scroll_area.setFrameShape(QScrollArea.frameShape.NoFrame)  # (可选) 移除边框

        # --- (!!!) 新增修复：(!!!) ---
        # 1. 始终根据需要显示垂直滚动条
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # 2. (修正) 仅在需要时显示水平滚动条 (不再是 AlwaysOff)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # --- (!!!) 修复结束 (!!!) ---

        # 3. (新增) 创建一个“容器” QWidget，所有内容将放入其中
        scroll_container = QWidget()

        # --- (!!!) 新增修复：(!!!) ---
        # 4. 为容器设置一个最小宽度，防止它被过度挤压
        #    (这个值 400px 是一个估计值，您可以根据需要调整)
        scroll_container.setMinimumWidth(400)
        # --- (!!!) 修复结束 (!!!) ---

        scroll_area.setWidget(scroll_container)  # 将容器放入滚动区

        # 5. (修改) 将布局应用到“容器”上，而不是主选项卡
        m4_l = QVBoxLayout(scroll_container)

        # --- 1. 拟合引擎选择 ---
        grp_engine = QGroupBox("拟合引擎")
        lay_engine = QFormLayout(grp_engine)

        self.m4_fit_engine_combo = QComboBox()
        self.m4_fit_engine_combo.addItems(["SciPy (多线程)", "PyTorch (DL批量)"])
        self.m4_fit_engine_combo.setToolTip(
            "选择用于批量拟合的计算引擎:\n"
            "- SciPy: 传统 CPU 优化，逐个拟合，多线程并行。\n"
            "- PyTorch: GPU 加速 (推荐)，将所有光谱作为一批 (Batch) 同时训练。"
        )
        if not peak_fitting_dl_available:
            self.m4_fit_engine_combo.model().item(1).setEnabled(False)
            self.m4_fit_engine_combo.setToolTip("PyTorch (DL批量) - 模块加载失败，不可用")

        lay_engine.addRow("批量拟合引擎:", self.m4_fit_engine_combo)

        # --- 2. DL 引擎参数 ---
        self.m4_dl_params_group = QGroupBox("DL 批量拟合参数")
        lay_dl = QFormLayout(self.m4_dl_params_group)
        self.m4_dl_epochs_input = QSpinBox()
        self.m4_dl_epochs_input.setRange(100, 100000);
        self.m4_dl_epochs_input.setValue(10000);
        self.m4_dl_epochs_input.setSingleStep(100)
        lay_dl.addRow("训练周期 (Epochs):", self.m4_dl_epochs_input)

        self.m4_dl_lr_input = QDoubleSpinBox()
        self.m4_dl_lr_input.setDecimals(5);
        self.m4_dl_lr_input.setRange(1e-6, 1.0);
        self.m4_dl_lr_input.setValue(0.005);
        self.m4_dl_lr_input.setSingleStep(0.001)
        lay_dl.addRow("学习率 (LR):", self.m4_dl_lr_input)

        # Sigma 边界 (Min/Max)
        sigma_layout = QHBoxLayout()
        self.m4_dl_min_sigma_input = QDoubleSpinBox();
        self.m4_dl_min_sigma_input.setDecimals(1);
        self.m4_dl_min_sigma_input.setRange(0.1, 100.0);
        self.m4_dl_min_sigma_input.setValue(0.1)
        self.m4_dl_max_sigma_input = QDoubleSpinBox();
        self.m4_dl_max_sigma_input.setDecimals(1);
        self.m4_dl_max_sigma_input.setRange(1.0, 500.0);
        self.m4_dl_max_sigma_input.setValue(60.0)
        sigma_layout.addWidget(self.m4_dl_min_sigma_input);
        sigma_layout.addWidget(QLabel("到"));
        sigma_layout.addWidget(self.m4_dl_max_sigma_input)
        lay_dl.addRow("Sigma 边界 (Min/Max):", sigma_layout)

        # Gamma 边界 (Min/Max)
        gamma_layout = QHBoxLayout()
        self.m4_dl_min_gamma_input = QDoubleSpinBox();
        self.m4_dl_min_gamma_input.setDecimals(1);
        self.m4_dl_min_gamma_input.setRange(0.1, 100.0);
        self.m4_dl_min_gamma_input.setValue(0.1)
        self.m4_dl_max_gamma_input = QDoubleSpinBox();
        self.m4_dl_max_gamma_input.setDecimals(1);
        self.m4_dl_max_gamma_input.setRange(1.0, 500.0);
        self.m4_dl_max_gamma_input.setValue(60.0)
        gamma_layout.addWidget(self.m4_dl_min_gamma_input);
        gamma_layout.addWidget(QLabel("到"));
        gamma_layout.addWidget(self.m4_dl_max_gamma_input)
        lay_dl.addRow("Gamma 边界 (Min/Max):", gamma_layout)

        # --- 【新增】正则化控件 ---
        l2_layout = QHBoxLayout()
        self.m4_dl_use_l2_cb = QCheckBox("启用 L2 正则化")
        self.m4_dl_l2_lambda_input = QDoubleSpinBox()
        self.m4_dl_l2_lambda_input.setDecimals(9);
        self.m4_dl_l2_lambda_input.setRange(1e-10, 1.0);
        self.m4_dl_l2_lambda_input.setValue(1e-7);
        self.m4_dl_l2_lambda_input.setSingleStep(1e-7)
        self.m4_dl_l2_lambda_input.setEnabled(False)  # 默认禁用
        self.m4_dl_use_l2_cb.toggled.connect(self.m4_dl_l2_lambda_input.setEnabled)  # 连接复选框
        l2_layout.addWidget(self.m4_dl_use_l2_cb);
        l2_layout.addWidget(self.m4_dl_l2_lambda_input)
        lay_dl.addRow("正则化 (Lambda):", l2_layout)

        # 默认隐藏 DL 参数
        self.m4_dl_params_group.setVisible(self.m4_fit_engine_combo.currentIndex() == 1)
        # 连接信号
        self.m4_fit_engine_combo.currentIndexChanged.connect(
            lambda idx: self.m4_dl_params_group.setVisible(idx == 1)
        )

        m4_l.addWidget(grp_engine)
        m4_l.addWidget(self.m4_dl_params_group)

        # 基础
        form1 = QFormLayout();
        self.m4_shape_combo = QComboBox();
        self.m4_shape_combo.addItems(["Voigt", "Gaussian"]);
        form1.addRow("峰形:", self.m4_shape_combo);
        self.m4_center_shift_input = QDoubleSpinBox();
        self.m4_center_shift_input.setRange(0.1, 1000.);
        self.m4_center_shift_input.setValue(10.);
        form1.addRow("中心偏移:", self.m4_center_shift_input);
        m4_l.addLayout(form1)
        # 交互
        grp_int = QGroupBox("绘图交互");
        lay_int = QHBoxLayout(grp_int);
        self.m4_pick_mode_btn = QPushButton("選峰");
        self.m4_pick_mode_btn.setCheckable(True);
        self.m4_pick_mode_btn.toggled.connect(self.on_manual_pick_toggled);
        self.m4_region_mode_btn = QPushButton("選區域");
        self.m4_region_mode_btn.setCheckable(True);
        self.m4_region_mode_btn.toggled.connect(self.on_region_select_toggled);
        self.m4_threshold_mode_btn = QPushButton("阈值线");
        self.m4_threshold_mode_btn.setCheckable(True);
        self.m4_threshold_mode_btn.toggled.connect(self.on_threshold_line_toggled);
        lay_int.addWidget(self.m4_pick_mode_btn);
        lay_int.addWidget(self.m4_region_mode_btn);
        lay_int.addWidget(self.m4_threshold_mode_btn);
        m4_l.addWidget(grp_int)
        # 区域
        grp_reg = QGroupBox("拟合区域");
        lay_reg = QVBoxLayout(grp_reg);
        self.m4_region_list = QListWidget();
        self.m4_region_list.setFixedHeight(60);
        lay_reg.addWidget(self.m4_region_list);
        lay_reg_btn = QHBoxLayout();
        self.m4_region_delete_btn = QPushButton("刪除");
        self.m4_region_delete_btn.clicked.connect(self.on_delete_region);
        self.m4_region_clear_btn = QPushButton("清空");
        self.m4_region_clear_btn.clicked.connect(self.on_clear_regions);
        lay_reg_btn.addWidget(self.m4_region_delete_btn);
        lay_reg_btn.addWidget(self.m4_region_clear_btn);
        lay_reg.addLayout(lay_reg_btn);
        m4_l.addWidget(grp_reg)
        # 锚点 (预览)
        grp_anc = QGroupBox("锚点 [预览用]");
        lay_anc = QVBoxLayout(grp_anc);
        self.m4_anchor_table = QTableWidget();
        self.m4_anchor_table.setColumnCount(2);
        self.m4_anchor_table.setHorizontalHeaderLabels(["X", "类型"]);
        self.m4_anchor_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch);
        self.m4_anchor_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed);
        self.m4_anchor_table.setColumnWidth(1, 60);
        self.m4_anchor_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);
        self.m4_anchor_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);
        lay_anc.addWidget(self.m4_anchor_table);
        self.m4_delete_anchor_btn = QPushButton("删除选中");
        self.m4_delete_anchor_btn.clicked.connect(self.on_delete_selected_anchor);
        lay_anc.addWidget(self.m4_delete_anchor_btn);
        lay_anc_btn = QHBoxLayout();
        self.m4_find_auto_btn = QPushButton("自動查找");
        self.m4_find_auto_btn.clicked.connect(self.on_find_auto_peaks);
        self.m4_clear_auto_btn = QPushButton("清空自動");
        self.m4_clear_auto_btn.clicked.connect(self.on_clear_anchors_auto);
        self.m4_clear_all_btn = QPushButton("清空所有");
        self.m4_clear_all_btn.clicked.connect(self.on_clear_anchors_all);
        lay_anc_btn.addWidget(self.m4_find_auto_btn);
        lay_anc_btn.addWidget(self.m4_clear_auto_btn);
        lay_anc_btn.addWidget(self.m4_clear_all_btn);
        lay_anc.addLayout(lay_anc_btn);
        m4_l.addWidget(grp_anc)
        # 自动寻峰参数
        grp_auto = QGroupBox("自动寻峰参数");
        lay_auto = QFormLayout(grp_auto);

        self.m4_use_autofind_cb = QCheckBox("启用自动寻峰 (将与手动锚点合并)");
        self.m4_use_autofind_cb.setChecked(True);  # 默认启用
        lay_auto.addRow(self.m4_use_autofind_cb)

        self.m4_autofind_smooth_cb = QCheckBox("平滑");
        self.m4_autofind_smooth_cb.setChecked(False);
        lay_auto.addRow(self.m4_autofind_smooth_cb);
        lay_thr = QHBoxLayout();
        self.m4_height_thresh_input = QDoubleSpinBox();
        self.m4_height_thresh_input.setRange(0., 1e9);
        self.m4_height_thresh_input.setValue(10.);
        self.m4_height_thresh_input.valueChanged.connect(self.on_threshold_spinbox_changed);
        lay_thr.addWidget(self.m4_height_thresh_input);
        lay_auto.addRow("阈值:", lay_thr);
        self.m4_peak_dist_input = QSpinBox();
        self.m4_peak_dist_input.setRange(1, 1000);
        self.m4_peak_dist_input.setValue(15);
        lay_auto.addRow("峰距:", self.m4_peak_dist_input);
        self.m4_max_peaks_input = QSpinBox();
        self.m4_max_peaks_input.setRange(1, 1000);
        self.m4_max_peaks_input.setValue(50);
        lay_auto.addRow("峰数:", self.m4_max_peaks_input);
        self.m4_tolerance_input = QDoubleSpinBox();
        self.m4_tolerance_input.setRange(0.1, 1000.);
        self.m4_tolerance_input.setValue(10.);
        lay_auto.addRow("容差:", self.m4_tolerance_input);
        m4_l.addWidget(grp_auto)
        # 执行
        lay_run = QHBoxLayout();
        self.m4_output_name_input = QLineEdit()
        self.m4_output_name_input.setPlaceholderText("结果命名 (留空自动生成)")
        self.m4_output_name_input.setToolTip(
            "为本次拟合结果指定一个唯一名称。\n如果留空，将自动生成 (例如: original_Voigt_timestamp)。")

        # 将输入框添加到布局
        lay_run.addWidget(QLabel("结果命名:"))
        lay_run.addWidget(self.m4_output_name_input)

        self.m4_run_fit_button = QPushButton("预览拟合")
        self.m4_run_fit_button.clicked.connect(self.on_run_peak_fit_preview)

        self.m4_run_batch_button = QPushButton("应用到勾选")
        self.m4_run_batch_button.setStyleSheet("background-color:#DAF7A6;")
        self.m4_run_batch_button.clicked.connect(self.on_run_batch_apply)

        self.m4_stop_batch_button = QPushButton("停止批量")
        self.m4_stop_batch_button.setStyleSheet("background-color:#FFC300;")
        self.m4_stop_batch_button.clicked.connect(self.on_stop_fitting)
        self.m4_stop_batch_button.setEnabled(False)

        lay_run.addWidget(self.m4_run_fit_button)
        lay_run.addWidget(self.m4_run_batch_button)
        lay_run.addWidget(self.m4_stop_batch_button)

        m4_l.addLayout(lay_run)
        m4_l.addStretch()  # 保持底部填充

        # --- VVVV 关键修改 VVVV ---
        # 5. (新增) 最后，为“主选项卡”设置一个布局，并把滚动区放进去
        tab_main_layout = QVBoxLayout(self.tab_peak_fitting)
        tab_main_layout.addWidget(scroll_area)
        tab_main_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        # --- ^^^^ 修改结束 ^^^^ ---

    def _setup_m5_augmentation_tab(self):
        """Builds the UI for the Augmentation Tab (M5) with sub-tabs."""
        self.tab_augment = QWidget()
        layout = QVBoxLayout(self.tab_augment)
        layout.setContentsMargins(5, 5, 5, 5)

        # --- 子选项卡 ---
        self.m5_sub_tabs = QTabWidget()
        layout.addWidget(self.m5_sub_tabs)

        # =====================================================================
        # Tab 1: 线性光谱插值 (Linear Spectrum)
        # =====================================================================
        linear_spec_tab = QWidget()
        linear_spec_layout = QVBoxLayout(linear_spec_tab)
        linear_spec_layout.addWidget(QLabel("直接在光谱强度之间进行线性插值。\n输入: 左侧面板选择的光谱版本。"))

        linear_spec_config_group = QGroupBox("配置")
        linear_spec_form = QFormLayout(linear_spec_config_group)
        self.m5_spec_interp_dim = QSpinBox()
        self.m5_spec_interp_dim.setRange(0, 10)
        self.m5_spec_interp_dim.setValue(0)
        linear_spec_form.addRow("插值维度索引:", self.m5_spec_interp_dim)
        linear_spec_layout.addWidget(linear_spec_config_group)

        linear_spec_target_group = QGroupBox("生成目标设置")
        linear_spec_target_form = QFormLayout(linear_spec_target_group)
        self.m5_spec_mode = QComboBox()
        self.m5_spec_mode.addItems(["Interpolate (按步长)", "Specific (指定值)"])
        linear_spec_target_form.addRow("生成模式:", self.m5_spec_mode)
        self.m5_spec_step = QDoubleSpinBox()
        self.m5_spec_step.setDecimals(4)
        self.m5_spec_step.setRange(1e-6, 1.0)
        self.m5_spec_step.setValue(0.01)
        linear_spec_target_form.addRow("插值步长:", self.m5_spec_step)
        self.m5_spec_specific = QLineEdit()
        self.m5_spec_specific.setPlaceholderText("例如: 0.25, 0.35, 0.45")
        linear_spec_target_form.addRow("特定值 (逗号分隔):", self.m5_spec_specific)
        linear_spec_layout.addWidget(linear_spec_target_group)
        linear_spec_layout.addStretch()
        self.m5_sub_tabs.addTab(linear_spec_tab, "线性光谱插值")

        # 连接信号
        self.m5_spec_mode.currentIndexChanged.connect(
            lambda: self._update_m5_target_widgets(
                self.m5_spec_mode, self.m5_spec_step, self.m5_spec_specific
            )
        )
        self._update_m5_target_widgets(self.m5_spec_mode, self.m5_spec_step, self.m5_spec_specific)

        # =====================================================================
        # Tab 2: 峰参数插值 (Linear Param)
        # =====================================================================
        linear_param_tab = QWidget()
        linear_param_layout = QVBoxLayout(linear_param_tab)
        linear_param_layout.addWidget(QLabel("在峰参数之间进行线性插值，然后重建光谱。\n输入: M4 拟合结果。"))

        linear_param_input_group = QGroupBox("输入")
        linear_param_input_form = QFormLayout(linear_param_input_group)
        self.m5_param_source_combo = QComboBox()
        self.m5_param_source_combo.setObjectName("m5_param_source_combo")
        linear_param_input_form.addRow("源 M4 拟合结果:", self.m5_param_source_combo)
        linear_param_layout.addWidget(linear_param_input_group)

        linear_param_config_group = QGroupBox("配置")
        linear_param_config_form = QFormLayout(linear_param_config_group)
        self.m5_param_interp_dim = QSpinBox()
        self.m5_param_interp_dim.setRange(0, 10)
        self.m5_param_interp_dim.setValue(0)
        linear_param_config_form.addRow("插值维度索引:", self.m5_param_interp_dim)
        self.m5_param_add_residuals = QCheckBox("添加真实残差 (推荐)")
        self.m5_param_add_residuals.setChecked(True)
        linear_param_config_form.addRow(self.m5_param_add_residuals)
        linear_param_layout.addWidget(linear_param_config_group)

        linear_param_target_group = QGroupBox("生成目标设置")
        linear_param_target_form = QFormLayout(linear_param_target_group)
        self.m5_param_mode = QComboBox()
        self.m5_param_mode.addItems(["Interpolate (按步长)", "Specific (指定值)"])
        linear_param_target_form.addRow("生成模式:", self.m5_param_mode)
        self.m5_param_step = QDoubleSpinBox()
        self.m5_param_step.setDecimals(4)
        self.m5_param_step.setRange(1e-6, 1.0)
        self.m5_param_step.setValue(0.01)
        linear_param_target_form.addRow("插值步长:", self.m5_param_step)
        self.m5_param_specific = QTextEdit()
        self.m5_param_specific.setPlaceholderText("例如: 0.23, 0\n0.26, 0\n(每行一个多维值)")
        self.m5_param_specific.setFixedHeight(60)
        linear_param_target_form.addRow("特定值 (每行一个):", self.m5_param_specific)
        linear_param_layout.addWidget(linear_param_target_group)
        linear_param_layout.addStretch()
        self.m5_sub_tabs.addTab(linear_param_tab, "峰参数插值")

        # 连接信号
        self.m5_param_mode.currentIndexChanged.connect(
            lambda: self._update_m5_target_widgets(
                self.m5_param_mode, self.m5_param_step, self.m5_param_specific
            )
        )
        self._update_m5_target_widgets(self.m5_param_mode, self.m5_param_step, self.m5_param_specific)

        # =====================================================================
        # Tab 3: MOGP 参数化生成 (MOGP)
        # =====================================================================
        mogp_tab = QWidget()
        # 使用 ScrollArea 防止内容过多显示不全
        mogp_scroll = QScrollArea()
        mogp_scroll.setWidgetResizable(True)
        mogp_content = QWidget()
        mogp_layout = QVBoxLayout(mogp_content)

        mogp_layout.addWidget(QLabel("训练 MOGP+多项式 模型生成峰参数，重建并添加残差。\n输入: M4 拟合结果。"))

        # --- 输入 ---
        mogp_input_group = QGroupBox("输入")
        mogp_input_form = QFormLayout(mogp_input_group)
        self.m5_mogp_source_combo = QComboBox()
        self.m5_mogp_source_combo.setObjectName("m5_mogp_source_combo")
        mogp_input_form.addRow("源 M4 拟合结果:", self.m5_mogp_source_combo)
        mogp_layout.addWidget(mogp_input_group)

        # --- 基础配置 ---
        mogp_config_group = QGroupBox("基础配置")
        mogp_config_form = QFormLayout(mogp_config_group)
        self.m5_mogp_latent_q = QSpinBox()
        self.m5_mogp_latent_q.setRange(1, 20)
        self.m5_mogp_latent_q.setValue(10)
        mogp_config_form.addRow("MOGP 潜变量 (Q):", self.m5_mogp_latent_q)

        self.m5_mogp_poly_deg = QSpinBox()
        self.m5_mogp_poly_deg.setRange(1, 5)
        self.m5_mogp_poly_deg.setValue(3)
        mogp_config_form.addRow("峰位多项式阶数:", self.m5_mogp_poly_deg)

        self.m5_mogp_add_residuals = QCheckBox("添加真实残差 (推荐)")
        self.m5_mogp_add_residuals.setChecked(True)
        mogp_config_form.addRow(self.m5_mogp_add_residuals)
        mogp_layout.addWidget(mogp_config_group)

        # --- 高级优化设置 (新增) ---
        mogp_adv_group = QGroupBox("高级优化设置 (RANSAC & 积分面积约束)")
        adv_layout = QFormLayout(mogp_adv_group)

        # RANSAC 设置
        self.m5_mogp_use_ransac = QCheckBox("对面积拟合启用 RANSAC (去除异常值)")
        self.m5_mogp_use_ransac.setChecked(False)
        self.m5_mogp_use_ransac.setToolTip("使用 RANSAC 算法剔除积分面积数据中的离群点，提高面积模型的鲁棒性。")
        adv_layout.addRow(self.m5_mogp_use_ransac)

        self.m5_mogp_ransac_min_samples = QSpinBox()
        self.m5_mogp_ransac_min_samples.setRange(2, 1000)
        self.m5_mogp_ransac_min_samples.setValue(7)
        self.m5_mogp_ransac_min_samples.setToolTip("RANSAC 拟合所需的最小样本数 (建议 > 样本总数的一半)。")
        adv_layout.addRow("RANSAC 最小样本数:", self.m5_mogp_ransac_min_samples)

        # 面积约束设置
        self.m5_mogp_apply_area_constraint = QCheckBox("应用积分面积约束优化")
        self.m5_mogp_apply_area_constraint.setChecked(False)
        self.m5_mogp_apply_area_constraint.setToolTip(
            "训练第二个 MOGP 模型预测峰组总面积，并以此优化生成的峰宽参数 (Sigma/Gamma)。")
        adv_layout.addRow(self.m5_mogp_apply_area_constraint)

        # 峰分组输入
        self.m5_mogp_peak_groups = QTextEdit()
        self.m5_mogp_peak_groups.setPlaceholderText("例如: 944-1078, 1078-1136\n(每行一个范围或逗号分隔，单位 cm-1)")
        self.m5_mogp_peak_groups.setFixedHeight(60)
        self.m5_mogp_peak_groups.setToolTip("定义峰分组的波长范围。程序将计算这些范围内所有峰的积分面积总和用于约束。")
        adv_layout.addRow("峰分组范围 (Start-End):", self.m5_mogp_peak_groups)

        mogp_layout.addWidget(mogp_adv_group)

        # --- 生成目标设置 ---
        mogp_target_group = QGroupBox("生成目标设置")
        mogp_target_form = QFormLayout(mogp_target_group)
        self.m5_mogp_mode = QComboBox()
        self.m5_mogp_mode.addItems(["Interpolate (按步长)", "Specific (指定值)"])
        mogp_target_form.addRow("生成模式:", self.m5_mogp_mode)

        self.m5_mogp_step = QDoubleSpinBox()
        self.m5_mogp_step.setDecimals(4)
        self.m5_mogp_step.setRange(1e-6, 1.0)
        self.m5_mogp_step.setValue(0.01)
        mogp_target_form.addRow("插值步长:", self.m5_mogp_step)

        self.m5_mogp_specific = QTextEdit()
        self.m5_mogp_specific.setPlaceholderText("例如: 0.23, 0\n0.26, 0\n(每行一个多维值)")
        self.m5_mogp_specific.setFixedHeight(60)
        mogp_target_form.addRow("特定值 (每行一个):", self.m5_mogp_specific)

        mogp_layout.addWidget(mogp_target_group)
        mogp_layout.addStretch()

        mogp_scroll.setWidget(mogp_content)
        self.m5_sub_tabs.addTab(mogp_scroll, "MOGP 参数化生成")

        # 连接信号
        self.m5_mogp_mode.currentIndexChanged.connect(
            lambda: self._update_m5_target_widgets(
                self.m5_mogp_mode, self.m5_mogp_step, self.m5_mogp_specific
            )
        )
        self._update_m5_target_widgets(self.m5_mogp_mode, self.m5_mogp_step, self.m5_mogp_specific)

        # --- 通用输出设置 (子选项卡外部) ---
        output_group = QGroupBox("输出设置")
        output_layout = QFormLayout(output_group)
        self.m5_output_prefix = QLineEdit()
        self.m5_output_prefix.setPlaceholderText("例如: my_augmented_data (留空则自动生成)")
        output_layout.addRow("输出文件名前缀:", self.m5_output_prefix)
        layout.addWidget(output_group)

        # --- 执行按钮 ---
        self.m5_run_button = QPushButton("开始生成增广数据")
        self.m5_run_button.setStyleSheet("background-color:#DAF7A6;")
        layout.addWidget(self.m5_run_button)

        layout.addStretch()

    def _setup_m6_export_tab(self):  # (保持不变)
        """Builds the UI for the Export Tab (M6)."""
        self.tab_export = QWidget()
        layout = QVBoxLayout(self.tab_export)
        layout.addWidget(QLabel("模块 6: 导出处理结果"))

        # --- Peak Fit Result Export ---
        fit_export_group = QGroupBox("导出峰拟合参数 (CSV)")
        fit_export_layout = QFormLayout(fit_export_group)

        self.m6_fit_version_combo = QComboBox()  # Dropdown to select fit result version
        fit_export_layout.addRow("选择拟合结果来源 (基于输入光谱版本):", self.m6_fit_version_combo)

        self.m6_export_fit_button = QPushButton("导出选中版本的拟合结果")
        self.m6_export_fit_button.clicked.connect(self.on_export_fit_results)
        fit_export_layout.addRow(self.m6_export_fit_button)
        layout.addWidget(fit_export_group)

        # --- Spectra Export (Placeholder) ---
        spectra_export_group = QGroupBox("导出光谱数据")
        spectra_export_layout = QFormLayout(spectra_export_group)
        self.m6_spectra_version_combo = QComboBox()  # Dropdown to select spectra version
        spectra_export_layout.addRow("选择要导出的光谱版本:", self.m6_spectra_version_combo)
        self.m6_export_spectra_button = QPushButton("导出选中版本的光谱 (CSV/NPY)")
        self.m6_export_spectra_button.setEnabled(False)  # Enable later
        self.m6_export_spectra_button.clicked.connect(self.on_export_spectra) # Connect later
        spectra_export_layout.addRow(self.m6_export_spectra_button)
        layout.addWidget(spectra_export_group)

        layout.addStretch()

    def _setup_page_0_workflow(self):
        """(新增) 构建 M1-M6 工作流页面的布局 (页面 0)。"""
        # (这是 _setup_layout 的旧代码)
        ls = QSplitter(Qt.Orientation.Vertical);
        ls.addWidget(self.project_manager);  # 使用主 project_manager
        ls.addWidget(self.processing_tabs);  # 使用主 processing_tabs (M2-M6)
        ls.setSizes([300, 550])  # 可调整

        rp_l = QVBoxLayout();
        rp_l.setContentsMargins(0, 0, 0, 0);
        rp_l.addWidget(self.plot_widget);  # 使用主 plot_widget
        rp_w = QWidget();
        rp_w.setLayout(rp_l)

        rs = QSplitter(Qt.Orientation.Vertical);
        rs.addWidget(rp_w);
        rs.addWidget(self.log_console);  # 使用主 log_console
        rs.setSizes([600, 200])  # 可调整

        ms = QSplitter(Qt.Orientation.Horizontal);
        ms.addWidget(ls);
        ms.addWidget(rs);
        ms.setSizes([450, 850])  # 可调整

        # --- 关键修改: 将布局设置到 page_0 ---
        page_0_layout = QVBoxLayout(self.page_0_workflow)  # 应用到 self.page_0_workflow
        page_0_layout.addWidget(ms)
        page_0_layout.setContentsMargins(0, 0, 0, 0)  # 无边距

    # (在 main.py 的 MainWindow 类中)
    def _setup_page_1_visualization(self):
        """(重构) 构建专用可视化页面的布局 (Page 1)。
        - 左侧: 控制面板 (数据源, 版本, 样本列表, 按钮)
        - 右侧: 双画板 (预览区, 主绘图区)
        """
        layout = QVBoxLayout(self.page_1_visualization)
        layout.setContentsMargins(5, 5, 5, 5)

        # 1. 创建主分割器 (左右)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)

        # 2. 创建所有控件
        self.vis_data_source_combo = QComboBox()
        self.vis_data_source_combo.setObjectName("vis_data_source_combo")
        self.vis_data_source_combo.setToolTip("选择要分析的项目 (主项目或 M5 生成的项目)")

        self.vis_plot_button = QPushButton("绘制选中项到主画板")
        self.vis_plot_button.setStyleSheet("background-color: #DAF7A6;")  # 绿色
        self.vis_clear_button = QPushButton("清空主画板")

        self.vis_show_components_cb = QCheckBox("在预览区显示构成峰")
        self.vis_show_components_cb.setObjectName("vis_show_components_cb")

        self.vis_preview_plot = PlotWidget()
        self.vis_preview_plot.setObjectName("vis_preview_plot")

        self.vis_main_plot = PlotWidget()
        self.vis_main_plot.setObjectName("vis_main_plot")
        self.vis_main_plot.ax_main.set_title("主绘图区 (可叠加)")  # 设置初始标题

        # 3. 构建左侧面板
        # (重要: 假设 project_manager.py 已按要求修改)
        try:
            self.vis_project_manager = ProjectManager(mode='visualization')
            self.vis_project_manager.setObjectName("vis_project_manager")

            # 注入顶部控件 (数据源)
            self.vis_project_manager.add_control_widget("数据源:", self.vis_data_source_combo)

            # 注入底部控件 (按钮)
            self.vis_project_manager.add_bottom_buttons([self.vis_plot_button, self.vis_clear_button])

            main_splitter.addWidget(self.vis_project_manager)  # 添加到主分割器
            print("[DEBUG Setup Vis] 成功创建并配置 'visualization' 模式的 ProjectManager。")

        except TypeError:
            print("!!!!!!!!!!!!!! 严重错误: ProjectManager.__init__ 不支持 'mode' 参数 !!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!! 请先修改 gui/project_manager.py !!!!!!!!!!!!!!")
            # 创建一个后备
            fallback_left = QWidget()
            fallback_layout = QVBoxLayout(fallback_left)
            fallback_layout.addWidget(QLabel("错误: ProjectManager 加载失败"))
            fallback_layout.addWidget(self.vis_data_source_combo)
            fallback_layout.addWidget(self.vis_plot_button)
            fallback_layout.addWidget(self.vis_clear_button)
            main_splitter.addWidget(fallback_left)
        except AttributeError as e:
            print(f"!!!!!!!!!!!!!! 严重错误: ProjectManager 缺少方法 (例如 add_control_widget): {e} !!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!! 请先修改 gui/project_manager.py !!!!!!!!!!!!!!")

        # 4. 构建右侧面板 (双画板)
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # 4a. 预览区 (右上)
        preview_group = QGroupBox("预览当前 (单击样本或切换版本)")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.vis_preview_plot)  # 添加预览画板
        preview_layout.addWidget(self.vis_show_components_cb)  # 添加复选框
        right_splitter.addWidget(preview_group)

        # 4b. 主绘图区 (右下)
        main_plot_group = QGroupBox("主绘图区 (可叠加)")
        main_plot_layout = QVBoxLayout(main_plot_group)
        main_plot_layout.addWidget(self.vis_main_plot)  # 添加主画板
        right_splitter.addWidget(main_plot_group)

        main_splitter.addWidget(right_splitter)  # 添加到主分割器

        # 5. 设置初始尺寸
        main_splitter.setSizes([350, 650])
        right_splitter.setSizes([400, 400])

    def _connect_signals(self):
        """(修改) 连接所有信号和槽。
        - Page 1 的连接被完全重构以适应新设计。
        """
        print("[DEBUG] Connecting signals...")

        # --- 页面 0 (工作流) 信号 (保持不变) ---
        if hasattr(self, 'project_manager') and self.project_manager:
            try:
                self.project_manager.sample_selected.connect(self.on_sample_selected)
                if hasattr(self.project_manager, 'version_selector_combo'):
                    self.project_manager.version_selector_combo.currentTextChanged.connect(self.on_version_selected)
                if hasattr(self.project_manager, 'compare_version_changed'):
                    self.project_manager.compare_version_changed.connect(self.on_compare_version_selected)
            except Exception as e:
                print(f"Error connecting project_manager signals: {e}")

        if hasattr(self, 'plot_widget') and self.plot_widget:
            try:
                self.plot_widget.anchor_added.connect(self.on_anchor_added_from_plot)
                self.plot_widget.region_defined.connect(self.on_region_defined_from_plot)
                self.plot_widget.threshold_set.connect(self.on_threshold_set_from_plot)
                self.plot_widget.anchor_delete_requested.connect(self.on_anchor_deleted_from_plot)
                # Main -> Plot
                self.request_plot_interaction_mode.connect(self.plot_widget.set_interaction_mode)
                self.request_plot_region_update.connect(self.plot_widget.on_update_regions)
                self.request_plot_anchors_update.connect(self.plot_widget.on_update_anchors)
                self.request_plot_threshold_update.connect(self.plot_widget.on_update_threshold_line)
            except Exception as e:
                print(f"Error connecting plot_widget signals: {e}")

        # --- M2 信号 ---
        if hasattr(self, 'm2_apply_peak_config_btn'):
            self.m2_apply_peak_config_btn.clicked.connect(self.on_m2_apply_peak_config)
        if hasattr(self, 'm2_run_generation_btn'):
            self.m2_run_generation_btn.clicked.connect(self.on_m2_run_generation)
        if hasattr(self, 'm2_scale_checkbox'):
            self.m2_scale_checkbox.toggled.connect(self.on_m2_scale_checkbox_toggled)
        # (m2_pick_peak_btn 和 m2_class_selector 的连接在 _setup_m2... 中完成)

        # --- M3 信号 ---
        if hasattr(self, 'baseline_preview_button'):
            self.baseline_preview_button.clicked.connect(self.on_run_baseline_preview)
        if hasattr(self, 'baseline_apply_button'):
            self.baseline_apply_button.clicked.connect(self.on_run_baseline_batch)
        if hasattr(self, 'denoise_preview_button'):
            self.denoise_preview_button.clicked.connect(self.on_run_denoise_preview)
        if hasattr(self, 'denoise_apply_button'):
            self.denoise_apply_button.clicked.connect(self.on_run_denoise_batch)
        if hasattr(self, 'algo_combo'):
            self.algo_combo.currentTextChanged.connect(self.on_algo_changed)
        if hasattr(self, 'msc_ref_combo'):
            self.msc_ref_combo.currentTextChanged.connect(self.on_msc_ref_choice_changed)

        # --- M4 信号 ---
        if hasattr(self, 'm4_pick_mode_btn'):
            self.m4_pick_mode_btn.toggled.connect(self.on_manual_pick_toggled)
        if hasattr(self, 'm4_region_mode_btn'):
            self.m4_region_mode_btn.toggled.connect(self.on_region_select_toggled)
        if hasattr(self, 'm4_threshold_mode_btn'):
            self.m4_threshold_mode_btn.toggled.connect(self.on_threshold_line_toggled)
        if hasattr(self, 'm4_height_thresh_input'):
            self.m4_height_thresh_input.valueChanged.connect(self.on_threshold_spinbox_changed)
        if hasattr(self, 'm4_region_delete_btn'):
            self.m4_region_delete_btn.clicked.connect(self.on_delete_region)
        if hasattr(self, 'm4_region_clear_btn'):
            self.m4_region_clear_btn.clicked.connect(self.on_clear_regions)
        if hasattr(self, 'm4_delete_anchor_btn'):
            self.m4_delete_anchor_btn.clicked.connect(self.on_delete_selected_anchor)
        if hasattr(self, 'm4_find_auto_btn'):
            self.m4_find_auto_btn.clicked.connect(self.on_find_auto_peaks)
        if hasattr(self, 'm4_clear_auto_btn'):
            self.m4_clear_auto_btn.clicked.connect(self.on_clear_anchors_auto)
        if hasattr(self, 'm4_clear_all_btn'):
            self.m4_clear_all_btn.clicked.connect(self.on_clear_anchors_all)
        if hasattr(self, 'm4_run_fit_button'):
            self.m4_run_fit_button.clicked.connect(self.on_run_peak_fit_preview)

        # --- VVVV 关键修改 VVVV ---
        # 此连接已移至 _setup_m4_peak_fitting_tab 并指向 on_run_batch_apply
        # if hasattr(self, 'm4_run_batch_button'):
        #     self.m4_run_batch_button.clicked.connect(self.on_run_batch_fit) # <--- 移除此行
        # --- ^^^^ 修改结束 ^^^^ ---

        if hasattr(self, 'm4_stop_batch_button'):
            self.m4_stop_batch_button.clicked.connect(self.on_stop_fitting)
        if hasattr(self, 'm4_anchor_table'):
            self.m4_anchor_table.itemSelectionChanged.connect(self.on_m4_anchor_table_sync)

        # --- M5 信号 ---
        if hasattr(self, 'm5_run_button'):
            self.m5_run_button.clicked.connect(self.on_run_augmentation)
        # (M5 子选项卡的连接在 _setup_m5_augmentation_tab 内部完成)

        # --- M6 信号 ---
        if hasattr(self, 'm6_export_fit_button'):
            self.m6_export_fit_button.clicked.connect(self.on_export_fit_results)
        # (M6 导出光谱按钮的连接将来添加)

        # --- VVVV 重构: 视图切换和页面 1 (可视化) 信号 VVVV ---
        # 1. 连接工具栏按钮 (保持不变)
        if self.workflow_action:
            self.workflow_action.triggered.connect(self.on_toggle_view_workflow)
            print("[DEBUG Connect] Connected workflow_action")
        if self.vis_action:
            self.vis_action.triggered.connect(self.on_toggle_view_visualization)
            print("[DEBUG Connect] Connected vis_action")

        # 2. (保持) 延迟连接 main_stack.currentChanged (在 __init__ 末尾)

        # 3. 连接可视化页面 (Page 1) 的新控件
        try:
            self.vis_data_source_combo.currentTextChanged.connect(self.on_vis_source_changed)
            print("[DEBUG Connect TRY] Connected vis_data_source_combo to on_vis_source_changed")
        except AttributeError:
            print("[DEBUG Connect TRY] FAILED to connect vis_data_source_combo")

        try:
            # (注意: 访问 vis_project_manager 内部的下拉框)
            self.vis_project_manager.version_selector_combo.currentTextChanged.connect(self.on_vis_version_changed)
            print("[DEBUG Connect TRY] Connected vis_project_manager.version_selector_combo to on_vis_version_changed")
        except AttributeError:
            print("[DEBUG Connect TRY] FAILED to connect vis_project_manager.version_selector_combo")

        try:
            self.vis_project_manager.sample_selected.connect(self.on_vis_preview_triggered)
            print("[DEBUG Connect TRY] Connected vis_project_manager.sample_selected to on_vis_preview_triggered")
        except AttributeError:
            print("[DEBUG Connect TRY] FAILED to connect vis_project_manager.sample_selected")

        try:
            self.vis_show_components_cb.toggled.connect(self.on_vis_preview_triggered)
            print("[DEBUG Connect TRY] Connected vis_show_components_cb to on_vis_preview_triggered")
        except AttributeError:
            print("[DEBUG Connect TRY] FAILED to connect vis_show_components_cb")

        try:
            self.vis_plot_button.clicked.connect(self.on_vis_plot_selected)
            print("[DEBUG Connect TRY] Connected vis_plot_button to on_vis_plot_selected")
        except AttributeError:
            print("[DEBUG Connect TRY] FAILED to connect vis_plot_button")

        try:
            self.vis_clear_button.clicked.connect(self.on_vis_clear_plot)
            print("[DEBUG Connect TRY] Connected vis_clear_button to on_vis_clear_plot")
        except AttributeError:
            print("[DEBUG Connect TRY] FAILED to connect vis_clear_button")
        # --- ^^^^ 重构结束 ^^^^ ---

        print("[DEBUG] Finished connecting signals.")

    # (可以放在 M2 相关的槽函数区域)
    @Slot(bool)
    def on_m2_scale_checkbox_toggled(self, checked):
        """
        (新增) 当 M2 的 "缩放伪标签" 复选框状态改变时，启用/禁用范围输入框。
        """
        if hasattr(self, 'm2_scale_min_input'):
            self.m2_scale_min_input.setEnabled(checked)
        if hasattr(self, 'm2_scale_max_input'):
            self.m2_scale_max_input.setEnabled(checked)

    # --- M3 Slots ---
    @Slot(str)
    def on_algo_changed(self, name):  # (保持不变)
        if hasattr(self, 'param_stack'): self.param_stack.setCurrentIndex(0 if name == "AirPLS" else 1)

    @Slot(str)
    def on_msc_ref_choice_changed(self, choice):  # (保持不变)
        if hasattr(self, 'msc_ref_index_label'):
            # Corrected comparison text
            v = (choice == "指定索引");
            self.msc_ref_index_label.setVisible(v);
            self.msc_ref_index_input.setVisible(v)

    # --- VVVV M3 逻辑修改: 拆分为 预览 和 批量 VVVV ---

    def _get_m3_buttons(self):
        """Helper to get list of M3 button widgets, if they exist."""
        return [getattr(self, name, None) for name in [
            "baseline_preview_button", "baseline_apply_button",
            "denoise_preview_button", "denoise_apply_button"]]

    @Slot()
    def on_run_baseline_preview(self):  # (重构: 仅预览)
        if not self.current_project: self.add_to_log("错误: 无项目"); return
        idx = self.project_manager.get_current_selected_index();
        if idx == -1: self.add_to_log("错误: 未选样本"); return

        # 获取预览所需数据
        try:
            x = self.current_project.wavelengths
            y_old = self.current_project.get_active_spectrum_by_index(idx)
            if y_old is None: raise ValueError(f"无法获取样本 {idx} 的激活数据")
            input_version = self.current_project.active_spectra_version
            title = f"样本 {idx} (预览, 基于: {input_version})"
        except Exception as e:
            self.add_to_log(f"错误: 预览失败: {e}");
            return

        algo = getattr(self, 'algo_combo', None) and self.algo_combo.currentText()
        if not algo: self.add_to_log("错误: 找不到基线算法控件"); return

        # 禁用所有M3/M4按钮
        buttons_to_disable = self._get_m3_buttons() + self._get_m4_buttons()
        for btn in buttons_to_disable:
            if btn: btn.setEnabled(False)
        QApplication.processEvents()

        try:
            if algo == "AirPLS" and hasattr(self, 'lambda_input'):
                lambda_val = self.lambda_input.value()
                # 直接调用核心函数
                baseline, y_new = process_airpls(y_old, lambda_=lambda_val)
                self.plot_widget.plot_baseline_results(algo, x, y_old, y_new, bl=baseline, title=title)

            elif algo == "MSC" and hasattr(self, 'msc_ref_combo'):
                ref_choice = self.msc_ref_combo.currentText()
                ref_idx = self.msc_ref_index_input.value() if ref_choice == "指定索引" else None

                # 获取参考光谱
                ref_spec = None
                try:
                    active_spectra = self.current_project.get_active_spectra()
                    if active_spectra is None: raise ValueError("无法获取激活光谱数据集")
                    if ref_choice == "类别平均":
                        ref_spec = np.mean(active_spectra, axis=0)
                    elif ref_choice == "指定索引":
                        if ref_idx is None or not (0 <= ref_idx < active_spectra.shape[0]):
                            raise ValueError(f"无效的参考索引: {ref_idx}")
                        ref_spec = active_spectra[ref_idx]
                    else:
                        raise ValueError(f"未知的 MSC 参考选项: {ref_choice}")
                except Exception as e:
                    self.add_to_log(f"错误: 无法获取 MSC 参考光谱: {e}");
                    return

                # 直接调用核心函数
                y_new, _, err = process_spectrum_msc(y_old, reference_spectrum=ref_spec)
                if err:
                    self.add_to_log(f"错误: MSC 预览失败: {err}");
                    return
                self.plot_widget.plot_baseline_results(algo, x, y_old, y_new, ref=ref_spec, title=title)

            else:
                self.add_to_log(f"错误: 基线算法 {algo} 或其控件未找到")

        except Exception as e:
            self.add_to_log(f"错误: {algo} 预览执行失败: {e}");
            traceback.print_exc()
        finally:
            for btn in buttons_to_disable:  # Re-enable buttons
                if btn: btn.setEnabled(True)

    @Slot()
    def on_run_denoise_preview(self):  # (重构: 仅预览)
        if not self.current_project: self.add_to_log("错误: 无项目"); return
        idx = self.project_manager.get_current_selected_index();
        if idx == -1: self.add_to_log("错误: 未选样本"); return

        # 获取预览所需数据
        try:
            x = self.current_project.wavelengths
            y_old = self.current_project.get_active_spectrum_by_index(idx)
            if y_old is None: raise ValueError(f"无法获取样本 {idx} 的激活数据")
            input_version = self.current_project.active_spectra_version
            title = f"样本 {idx} (预览, 基于: {input_version})"
        except Exception as e:
            self.add_to_log(f"错误: 预览失败: {e}");
            return

        algo = getattr(self, 'denoise_algo_combo', None) and self.denoise_algo_combo.currentText()
        if not algo: self.add_to_log("错误: 找不到降噪算法控件"); return

        # 禁用所有M3/M4按钮
        buttons_to_disable = self._get_m3_buttons() + self._get_m4_buttons()
        for btn in buttons_to_disable:
            if btn: btn.setEnabled(False)
        QApplication.processEvents()

        try:
            params = {}
            if algo == "Savitzky-Golay" and hasattr(self, 'sg_window_input'):
                win = self.sg_window_input.value();
                order = self.sg_order_input.value()
                params = {'window_length': win, 'polyorder': order}

                # 直接调用核心函数
                y_new, err = process_spectrum_denoise(y_old, algorithm=algo, **params)
                if err:
                    self.add_to_log(f"错误: {algo} 预览失败: {err}");
                    return

                self.plot_widget.plot_denoise_results(x, y_old, y_new, title=title)
            else:
                self.add_to_log(f"错误: 降噪算法 {algo} 或其控件未找到")

        except Exception as e:
            self.add_to_log(f"错误: {algo} 预览执行失败: {e}");
            traceback.print_exc()
        finally:
            for btn in buttons_to_disable:  # Re-enable buttons
                if btn: btn.setEnabled(True)

    @Slot()
    def on_run_baseline_batch(self):
        """ (新增) 应用基线校正到 *勾选的* 样本。"""
        print("--- 调试：进入 on_run_baseline_batch ---")
        if not self.current_project: self.add_to_log("错误: 无项目"); return

        indices = self.project_manager.get_checked_items_indices()
        if not indices:
            self.add_to_log("错误: 未勾选任何样本。请在左侧列表中勾选要处理的样本。");
            return

        input_version = self.current_project.active_spectra_version
        algo = getattr(self, 'algo_combo', None) and self.algo_combo.currentText()
        if not algo: self.add_to_log("错误: 找不到基线算法控件"); return

        self.add_to_log(f"--- 开始批量基线校正 ({algo}) {len(indices)} 个样本 (基于: {input_version}) ---")

        buttons_to_disable = self._get_m3_buttons() + self._get_m4_buttons()
        for btn in buttons_to_disable:
            if btn: btn.setEnabled(False)
        QApplication.processEvents()

        try:
            if algo == "AirPLS" and hasattr(self, 'lambda_input'):
                lambda_val = self.lambda_input.value()
                # 调用修改后的辅助函数
                _ = self._apply_airpls_to_version(input_version, lambda_val, indices_to_process=indices)
            elif algo == "MSC" and hasattr(self, 'msc_ref_combo'):
                ref_choice = self.msc_ref_combo.currentText()
                ref_idx = self.msc_ref_index_input.value() if ref_choice == "指定索引" else None
                # --- VVVV 修复函数调用 VVVV ---
                # 调用修改后的辅助函数
                _ = self._apply_msc_to_version(input_version, ref_choice, indices_to_process=indices, ref_index=ref_idx)
                # --- ^^^^ 修复函数调用 ^^^^ ---
            else:
                self.add_to_log(f"错误: 基线算法 {algo} 或其控件未找到")

        except Exception as e:
            self.add_to_log(f"错误: {algo} 批量处理失败: {e}");
            traceback.print_exc()
        finally:
            self.add_to_log(f"--- 批量基线校正完成 ---")
            for btn in buttons_to_disable:  # Re-enable buttons
                if btn: btn.setEnabled(True)

    @Slot()
    def on_run_denoise_batch(self):
        """ (新增) 应用降噪到 *勾选的* 样本。"""
        print("--- 调试：进入 on_run_denoise_batch ---")  #
        if not self.current_project: self.add_to_log("错误: 无项目"); return

        indices = self.project_manager.get_checked_items_indices()
        if not indices:
            self.add_to_log("错误: 未勾选任何样本。请在左侧列表中勾选要处理的样本。");
            return

        input_version = self.current_project.active_spectra_version
        algo = getattr(self, 'denoise_algo_combo', None) and self.denoise_algo_combo.currentText()
        if not algo: self.add_to_log("错误: 找不到降噪算法控件"); return

        self.add_to_log(f"--- 开始批量降噪 ({algo}) {len(indices)} 个样本 (基于: {input_version}) ---")

        buttons_to_disable = self._get_m3_buttons() + self._get_m4_buttons()
        for btn in buttons_to_disable:
            if btn: btn.setEnabled(False)
        QApplication.processEvents()

        try:
            params = {}
            if algo == "Savitzky-Golay" and hasattr(self, 'sg_window_input'):
                win = self.sg_window_input.value();
                order = self.sg_order_input.value()
                params = {'window_length': win, 'polyorder': order}
                # 调用修改后的辅助函数
                _ = self._apply_denoise_to_version(input_version, algo, params, indices_to_process=indices)
            else:
                self.add_to_log(f"错误: 降噪算法 {algo} 或其控件未找到")

        except Exception as e:
            self.add_to_log(f"错误: {algo} 批量处理失败: {e}");
            traceback.print_exc()
        finally:
            self.add_to_log(f"--- 批量降噪完成 ---")
            for btn in buttons_to_disable:  # Re-enable buttons
                if btn: btn.setEnabled(True)

        # --- M3 Version Creation Helpers ---

        # --- VVVV M3 辅助函数修改 VVVV ---
    def _apply_airpls_to_version(self, input_version_name, lambda_val, indices_to_process: List[int]):
            """
            (修改) 对 *指定的索引* 应用 AirPLS 并创建新版本。
            未指定的索引将保留原样。
            """
            if not self.current_project: return None

            input_spectra = self.current_project.spectra_versions.get(input_version_name)
            if input_spectra is None:
                self.add_to_log(f"错误: 找不到输入版本 '{input_version_name}'")
                return None

            n_total = input_spectra.shape[0]
            n_process = len(indices_to_process)

            # 关键修改: 复制所有数据，我们只修改需要处理的部分
            corrected = input_spectra.copy()
            success = 0

            self.add_to_log(f"应用 AirPLS (L={lambda_val}) to {n_process} samples (from '{input_version_name}')...");
            QApplication.processEvents()

            for i in indices_to_process:  # 关键修改: 仅循环勾选的索引
                if not (0 <= i < n_total):
                    self.add_to_log(f"  警告: 索引 {i} 超出范围，跳过。")
                    continue
                try:
                    if 'process_airpls' not in globals(): raise NameError("process_airpls not imported")
                    # 处理 input_spectra[i]，存入 corrected[i]
                    _, corr = process_airpls(input_spectra[i], lambda_=lambda_val);
                    corrected[i] = corr;
                    success += 1
                except Exception as e:
                    self.add_to_log(f"  样本 {i} AirPLS 失败: {e}")
                    # traceback.print_exc()
                if success > 0 and success % 50 == 0:  # 每处理 50 个样本更新一次日志
                    self.add_to_log(f"  ...已处理 {success}/{n_process}");
                    QApplication.processEvents()

            if success == 0:
                self.add_to_log("错误: AirPLS 未能成功处理任何指定样本");
                return None

            self.add_to_log(f"  ...总共 {success}/{n_process} 个样本处理成功。")

            # 生成版本名 (确保唯一性)
            out_name = f"{input_version_name}_airpls_L{int(lambda_val)}";
            cnt = 1;
            base = out_name;
            while out_name in self.current_project.spectra_versions: out_name = f"{base}_{cnt}"; cnt += 1

            # 创建历史记录
            hist = {'step': 'AirPLS',
                    'params': {'lambda_': lambda_val},
                    'input_version': input_version_name,
                    'output_version': out_name,
                    'indices_processed': indices_to_process}  # 记录被处理的索引
            try:
                self.current_project.add_spectra_version(out_name, corrected, hist)
                self.add_to_log(f"创建版本: '{out_name}'")

                  # --- ^^^^ 新增结束 ^^^^ ---

                try:
                    cache_dir = self._get_project_cache_dir()
                    if cache_dir:
                        spectra_cache_dir = os.path.join(cache_dir, "spectra")
                        os.makedirs(spectra_cache_dir, exist_ok=True)  # 确保目录存在

                        file_path = os.path.join(spectra_cache_dir, f"{out_name}.npy")
                        np.save(file_path, corrected)
                        self.add_to_log(f"  > 已将版本 {out_name} 写入缓存: {file_path}")
                    else:
                        self.add_to_log(f"  ! 警告: 无法获取缓存目录，跳过磁盘写入。")
                except Exception as e_save:
                    self.add_to_log(f"  ! 错误: 写入 M3 缓存失败: {e_save}")

                self._update_version_selector()
                return out_name
            except Exception as e:
                self.add_to_log(f"添加版本失败: {e}");
                traceback.print_exc();
                return None

        # --- VVVV 修复函数定义 VVVV ---
    def _apply_msc_to_version(self, input_version_name, ref_choice, indices_to_process: List[int], ref_index=None):
            # --- ^^^^ 修复函数定义 ^^^^ ---
            """
            (修改) 对 *指定的索引* 应用 MSC 并创建新版本。
            未指定的索引将保留原样。
            (修复) 修正了参数顺序 (indices_to_process 移到 ref_index=None 之前)
            """
            if not self.current_project: return None;
            input_spectra = self.current_project.spectra_versions.get(input_version_name);
            if input_spectra is None: self.add_to_log(f"Err: No input ver '{input_version_name}'"); return None

            n_total, nf = input_spectra.shape;
            n_process = len(indices_to_process)

            # 关键修改: 复制所有数据
            corrected = input_spectra.copy();
            success = 0;
            ref_spec = None;
            desc = ""

            self.add_to_log(f"Apply MSC to {n_process} samples (from '{input_version_name}')...");
            QApplication.processEvents()

            try:  # 确定参考光谱 (使用 *所有* 输入光谱来计算平均值)
                if ref_choice == "类别平均":
                    ref_spec = np.mean(input_spectra, axis=0);
                    desc = "全体平均";
                    self.add_to_log(" Ref: Mean")
                elif ref_choice == "指定索引":
                    if ref_index is None or not (0 <= ref_index < n_total): raise ValueError(
                        f"Invalid index {ref_index}");
                    ref_spec = input_spectra[ref_index];
                    desc = f"Smp {ref_index}";
                    self.add_to_log(f" Ref: Smp {ref_index}")
                else:
                    raise ValueError(f"Unknown ref: {ref_choice}")
                if ref_spec.shape != (nf,): raise ValueError(f"Ref shape {ref_spec.shape}!={nf}")
            except Exception as e:
                self.add_to_log(f"Err get ref:{e}");
                return None

            # 应用 MSC
            for i in indices_to_process:  # 关键修改: 仅循环勾选的索引
                if not (0 <= i < n_total):
                    self.add_to_log(f"  警告: 索引 {i} 超出范围，跳过。")
                    continue
                try:
                    corr, _, err = process_spectrum_msc(input_spectra[i], ref_spec);
                except Exception as e_msc:
                    err = str(e_msc);
                if err:
                    self.add_to_log(f" Smp{i} MSC Fail:{err}")
                else:
                    corrected[i] = corr;
                    success += 1
                if success > 0 and success % 50 == 0:
                    self.add_to_log(f" ...{success}/{n_process}");
                    QApplication.processEvents()

            if success == 0: self.add_to_log("Err: MSC no success"); return None

            self.add_to_log(f"  ...总共 {success}/{n_process} 个样本处理成功。")

            # 版本命名和保存
            sfx = f"ref{ref_index}" if ref_choice == "指定索引" else "refMean";
            out = f"{input_version_name}_msc_{sfx}";
            cnt = 1;
            base = out;
            while out in self.current_project.spectra_versions:
                out = f"{base}_{cnt}";
                cnt += 1
            hist = {'step': 'MSC', 'params': {'reference': desc}, 'input_version': input_version_name,
                    'output_version': out, 'indices_processed': indices_to_process}
            try:
                self.current_project.add_spectra_version(out, corrected, hist);
                try:
                    cache_dir = self._get_project_cache_dir()
                    if cache_dir:
                        spectra_cache_dir = os.path.join(cache_dir, "spectra")
                        os.makedirs(spectra_cache_dir, exist_ok=True)  # 确保目录存在

                        file_path = os.path.join(spectra_cache_dir, f"{out}.npy")
                        np.save(file_path, corrected)
                        self.add_to_log(f"  > 已将版本 {out} 写入缓存: {file_path}")
                    else:
                        self.add_to_log(f"  ! 警告: 无法获取缓存目录，跳过磁盘写入。")
                except Exception as e_save:
                    self.add_to_log(f"  ! 错误: 写入 M3 缓存失败: {e_save}")
                self.add_to_log(
                    f"Version created: '{out}'");


                self._update_version_selector();
                return out
            except Exception as e:
                self.add_to_log(f"Err add version:{e}");
                traceback.print_exc();
                return None

    def _apply_denoise_to_version(self, input_version_name, algo, params, indices_to_process: List[int]):
            """
            (修改) 对 *指定的索引* 应用 Denoise 并创建新版本。
            未指定的索引将保留原样。
            """
            if not self.current_project: return None;
            input_spectra = self.current_project.spectra_versions.get(input_version_name);
            if input_spectra is None: self.add_to_log(f"Err: No input ver '{input_version_name}'"); return None

            n_total = input_spectra.shape[0];
            n_process = len(indices_to_process)

            # 关键修改: 复制所有数据
            denoised = input_spectra.copy();
            success = 0;
            p_str = "_".join(f"{k}{v}" for k, v in params.items())

            self.add_to_log(f"Apply {algo}({p_str}) to {n_process} samples (from '{input_version_name}')...");
            QApplication.processEvents()

            for i in indices_to_process:  # 关键修改: 仅循环勾选的索引
                if not (0 <= i < n_total):
                    self.add_to_log(f"  警告: 索引 {i} 超出范围，跳过。")
                    continue
                try:
                    den, err = process_spectrum_denoise(input_spectra[i], algorithm=algo, **params);
                except Exception as e_den:
                    err = str(e_den);
                if err:
                    self.add_to_log(f" Smp{i} {algo} Fail:{err}")
                else:
                    denoised[i] = den;
                    success += 1
                if success > 0 and success % 50 == 0:
                    self.add_to_log(f" ...{success}/{n_process}");
                    QApplication.processEvents()

            if success == 0: self.add_to_log(f"Err: {algo} no success"); return None

            self.add_to_log(f"  ...总共 {success}/{n_process} 个样本处理成功。")

            # 版本命名和保存
            algo_s = "sg" if "Savitzky" in algo else algo.lower();
            out = f"{input_version_name}_{algo_s}_{p_str}";
            cnt = 1;
            base = out;
            while out in self.current_project.spectra_versions:
                out = f"{base}_{cnt}";
                cnt += 1
            hist = {'step': algo, 'params': params, 'input_version': input_version_name, 'output_version': out,
                    'indices_processed': indices_to_process}
            try:
                self.current_project.add_spectra_version(out, denoised, hist);

                try:
                    cache_dir = self._get_project_cache_dir()
                    if cache_dir:
                        spectra_cache_dir = os.path.join(cache_dir, "spectra")
                        os.makedirs(spectra_cache_dir, exist_ok=True)  # 确保目录存在

                        file_path = os.path.join(spectra_cache_dir, f"{out}.npy")
                        np.save(file_path, denoised)
                        self.add_to_log(f"  > 已将版本 {out} 写入缓存: {file_path}")
                    else:
                        self.add_to_log(f"  ! 警告: 无法获取缓存目录，跳过磁盘写入。")
                except Exception as e_save:
                    self.add_to_log(f"  ! 错误: 写入 M3 缓存失败: {e_save}")

                self.add_to_log(
                    f"Version created: '{out}'");
                self._update_version_selector();
                return out
            except Exception as e:
                self.add_to_log(f"Err add version:{e}");
                traceback.print_exc();
                return None

        # --- ^^^^ M3 辅助函数修改结束 ^^^^ ---

        # --- M4 Slots (保持不变) ---
    def _get_m4_buttons(self):
            """Helper to get list of M4 button widgets, if they exist."""
            return [getattr(self, name, None) for name in [
                "m4_find_auto_btn", "m4_run_fit_button", "m4_run_batch_button",
                "m4_clear_auto_btn", "m4_clear_all_btn", "m4_delete_anchor_btn",
                "m4_stop_batch_button"]]  # Include stop button

    def _set_m4_buttons_enabled(self, enabled):  # Updated
            """Enable/disable M4 buttons (excluding stop)."""
            buttons = [getattr(self, name, None) for name in
                       ["m4_find_auto_btn", "m4_run_fit_button", "m4_run_batch_button", "m4_clear_auto_btn",
                        "m4_clear_all_btn", "m4_delete_anchor_btn"]]
            is_running = not enabled
            stop_btn = getattr(self, 'm4_stop_batch_button', None)
            if stop_btn: stop_btn.setEnabled(
                is_running and self.m4_batch_total > 0 and self.m4_batch_counter < self.m4_batch_total)  # More precise condition
            for btn in buttons:
                if btn: btn.setEnabled(enabled)

    @Slot()
    def on_import_m4_results_csv(self):
        """
        从 CSV 文件导入 M4 拟合结果并添加到当前会话中。
        """
        if not self.current_project:
            QMessageBox.warning(self, "无项目", "请先加载一个主项目，然后再导入拟合结果。")
            return

        # 1. 选择文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入 M4 拟合结果 CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            # 2. 调用核心函数读取
            results_dict = load_m4_fit_results_from_csv(file_path)

            if not results_dict:
                QMessageBox.warning(self, "导入失败", "文件中未包含有效的样本数据 (或 sample_index 列缺失)。")
                return

            # 3. 询问版本名称
            default_name = os.path.splitext(os.path.basename(file_path))[0]
            # 移除可能的时间戳后缀以保持整洁 (可选)
            # default_name = default_name.split('_')[0]

            version_name, ok = QInputDialog.getText(
                self, "命名拟合结果",
                "为导入的拟合结果指定一个唯一名称 (Version Name):",
                text=default_name
            )

            if not ok or not version_name:
                return

            if version_name in self.m4_batch_fit_results:
                reply = QMessageBox.question(
                    self, "覆盖确认",
                    f"版本 '{version_name}' 已存在。是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            # 4. 存储结果
            self.m4_batch_fit_results[version_name] = results_dict
            self.add_to_log(f"成功导入 M4 结果: '{version_name}' ({len(results_dict)} 个样本)")

            # 5. 刷新 UI 选择器 (M5 的下拉框现在应该能看到这个新版本了)
            self._update_version_selector()

            QMessageBox.information(self, "导入成功",
                                    f"已加载 '{version_name}'。\n您现在可以在 M5 模块的输入下拉框中选择它。")

        except Exception as e:
            self.add_to_log(f"导入 M4 CSV 失败: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "导入错误", f"读取文件时出错:\n{e}")

    # --- ^^^^ 新增结束 ^^^^ ---

    @Slot(int)
    def on_main_tab_changed(self, index):  # Updated Check
            is_m4_tab = (hasattr(self, 'tab_peak_fitting') and self.processing_tabs.widget(
                index) == self.tab_peak_fitting)
            if not is_m4_tab: self._disable_all_plot_interactions()

    def _disable_all_plot_interactions(self):  # Updated Check
            btns = [getattr(self, name, None) for name in
                    ["m4_pick_mode_btn", "m4_region_mode_btn", "m4_threshold_mode_btn", "m2_pick_peak_btn"]]  # 添加 M2 按钮
            for btn in btns:
                if btn: btn.blockSignals(True); btn.setChecked(False); btn.blockSignals(False)
            self.request_plot_interaction_mode.emit(MODE_DISABLED)
            self.current_pick_mode_owner = None  # 确保清除所有者

    @Slot(bool)
    def on_manual_pick_toggled(self, checked):  # Updated logic
            btn_region = getattr(self, 'm4_region_mode_btn', None)
            btn_thresh = getattr(self, 'm4_threshold_mode_btn', None)
            btn_self = getattr(self, 'm4_pick_mode_btn', None)
            btn_m2_pick = getattr(self, 'm2_pick_peak_btn', None)
            if checked:
                # --- VVVV 新增: M2 互斥 VVVV ---
                self.current_pick_mode_owner = 'M4'
                if btn_m2_pick and btn_m2_pick.isChecked():
                    btn_m2_pick.blockSignals(True);
                    btn_m2_pick.setChecked(False);
                    btn_m2_pick.blockSignals(False)
                # --- ^^^^ 新增结束 ^^^^ ---

                if btn_region and btn_region.isChecked(): btn_region.setChecked(False)
                if btn_thresh and btn_thresh.isChecked(): btn_thresh.setChecked(False)
                if btn_self and not btn_self.isChecked(): btn_self.setChecked(True)  # Ensure self is checked
                self.request_plot_interaction_mode.emit(MODE_PICK_ANCHOR);
                self.add_to_log("交互: [M4 选峰] 启用")
            elif (not btn_region or not btn_region.isChecked()) and \
                    (not btn_thresh or not btn_thresh.isChecked()) and \
                    (not btn_m2_pick or not btn_m2_pick.isChecked()):  # 确保 M2 也没被选中

                self._disable_all_plot_interactions()  # 只有在 M2 也没选中的情况下才禁用

            if not checked and self.current_pick_mode_owner == 'M4':
                self.current_pick_mode_owner = None

    @Slot(bool)
    def on_region_select_toggled(self, checked):  # Updated logic
            btn_pick = getattr(self, 'm4_pick_mode_btn', None)
            btn_thresh = getattr(self, 'm4_threshold_mode_btn', None)
            btn_self = getattr(self, 'm4_region_mode_btn', None)
            btn_m2_pick = getattr(self, 'm2_pick_peak_btn', None)  # M2 互斥
            if checked:
                if btn_pick and btn_pick.isChecked(): btn_pick.setChecked(False)
                if btn_thresh and btn_thresh.isChecked(): btn_thresh.setChecked(False)
                if btn_m2_pick and btn_m2_pick.isChecked(): btn_m2_pick.setChecked(False)  # M2 互斥
                if btn_self and not btn_self.isChecked(): btn_self.setChecked(True)
                self.request_plot_interaction_mode.emit(MODE_SELECT_REGION);
                self.add_to_log("交互: [选区域] 启用")
            elif (not btn_pick or not btn_pick.isChecked()) and \
                    (not btn_thresh or not btn_thresh.isChecked()) and \
                    (not btn_m2_pick or not btn_m2_pick.isChecked()):  # M2 互斥
                self._disable_all_plot_interactions()

    @Slot(bool)
    def on_threshold_line_toggled(self, checked):  # Updated logic
            btn_pick = getattr(self, 'm4_pick_mode_btn', None)
            btn_region = getattr(self, 'm4_region_mode_btn', None)
            btn_self = getattr(self, 'm4_threshold_mode_btn', None)
            btn_m2_pick = getattr(self, 'm2_pick_peak_btn', None)  # M2 互斥
            spin_box = getattr(self, 'm4_height_thresh_input', None)
            if checked:
                if btn_pick and btn_pick.isChecked(): btn_pick.setChecked(False)
                if btn_region and btn_region.isChecked(): btn_region.setChecked(False)
                if btn_m2_pick and btn_m2_pick.isChecked(): btn_m2_pick.setChecked(False)  # M2 互斥
                if btn_self and not btn_self.isChecked(): btn_self.setChecked(True)
                self.request_plot_interaction_mode.emit(MODE_THRESHOLD_LINE)  # 1. Set mode
                current = spin_box.value() if spin_box else 0
                self.request_plot_threshold_update.emit(current)  # 2. Send value
                self.add_to_log(f"交互: [阈值线] 启用 (值: {current:.2f})")
            elif (not btn_pick or not btn_pick.isChecked()) and \
                    (not btn_region or not btn_region.isChecked()) and \
                    (not btn_m2_pick or not btn_m2_pick.isChecked()):  # M2 互斥
                self._disable_all_plot_interactions()

    @Slot(float)
    def on_threshold_set_from_plot(self, threshold):  # Updated logic
            spin_box = getattr(self, 'm4_height_thresh_input', None)
            if spin_box:
                spin_box.blockSignals(True)
                min_v, max_v = spin_box.minimum(), spin_box.maximum();
                clamped = max(min_v, min(threshold, max_v))
                spin_box.setValue(clamped);
                spin_box.blockSignals(False)
                msg = f"阈值更新: {clamped:.2f}";
                if not np.isclose(threshold, clamped): msg += f" (限制在 [{min_v:.1f}, {max_v:.1f}])"
                self.add_to_log(msg)

    @Slot(float)
    def on_threshold_spinbox_changed(self, value):  # Updated logic
            thresh_btn = getattr(self, 'm4_threshold_mode_btn', None)
            if thresh_btn and thresh_btn.isChecked(): self.request_plot_threshold_update.emit(value)

    @Slot(float)
    def on_anchor_added_from_plot(self, x_pos):
            """
            (已修改) 根据 self.current_pick_mode_owner 路由信号。
            """
            owner = self.current_pick_mode_owner

            if owner == 'M4':
                # 路由到 M4: 在表格中添加锚点
                if hasattr(self, 'm4_anchor_table'):
                    self._add_anchor_to_table(x_pos, "Manual", QColor("lightgreen"))
                    self._update_plot_anchors()
                # (M4 保持多选模式，不自动禁用)

            elif owner == 'M2':
                # 路由到 M2: 更新 M2 UI 上的 QLineEdit
                self.on_m2_peak_received(x_pos)

            else:
                # 意外的选峰，禁用模式
                self.add_to_log(f"警告: 在未知模式下接收到选峰: {x_pos:.2f}")
                self._disable_all_plot_interactions()

    @Slot(float, float)
    def on_region_defined_from_plot(self, x1, x2):  # Updated logic
            if hasattr(self, 'm4_region_list'):
                start, end = min(x1, x2), max(x1, x2);
                region_str = f"{start:.2f} - {end:.2f}"
                items = [self.m4_region_list.item(i).text() for i in range(self.m4_region_list.count())]
                if region_str not in items: self.m4_region_list.addItem(
                    region_str); self._update_regions_from_list(); self.add_to_log(f"区域添加: {region_str}")

    @Slot()
    def on_delete_region(self):  # Updated logic
            if hasattr(self, 'm4_region_list'):
                sel = self.m4_region_list.selectedItems();
                if not sel: return
                for item in sel: self.m4_region_list.takeItem(self.m4_region_list.row(item))
                self._update_regions_from_list();
                self.add_to_log(f"删除 {len(sel)} 区域")

    @Slot()
    def on_clear_regions(self):  # Updated logic
            if hasattr(self,
                       'm4_region_list'): self.m4_region_list.clear(); self._update_regions_from_list(); self.add_to_log(
                "清空区域")

    def _update_regions_from_list(self):  # Updated logic (Robust parsing)
            new_regions = []
            if hasattr(self, 'm4_region_list'):
                for i in range(self.m4_region_list.count()):
                    item_text = self.m4_region_list.item(i).text()
                    try:
                        s, e = map(float, item_text.split(" - "));
                        new_regions.append((min(s, e), max(s, e)))
                    except (ValueError, IndexError, AttributeError) as e:
                        print(f"解析区域字符串失败: '{item_text}' - {e}")
            self.m4_current_regions = new_regions  # Update state
            self.request_plot_region_update.emit(self.m4_current_regions)  # Update plot

    def _add_anchor_to_table(self, x_pos, type_str, bg_color=None):  # Updated logic (Tolerance check)
            if not hasattr(self, 'm4_anchor_table'): return
            tol = 1e-3;
            existing = [a[0] for a in self._get_anchors_from_table()]
            if any(np.isclose(x_pos, ex, atol=tol) for ex in existing): print(f"锚点 {x_pos:.2f} 已存在."); return
            row = self.m4_anchor_table.rowCount();
            self.m4_anchor_table.insertRow(row)
            item_p = QTableWidgetItem(f"{x_pos:.3f}");
            item_p.setData(Qt.ItemDataRole.UserRole, float(x_pos))
            item_t = QTableWidgetItem(type_str);
            item_t.setData(Qt.ItemDataRole.UserRole, type_str)
            if bg_color: item_p.setBackground(bg_color); item_t.setBackground(bg_color)
            self.m4_anchor_table.setItem(row, 0, item_p);
            self.m4_anchor_table.setItem(row, 1, item_t)
            self.m4_anchor_table.sortItems(0, Qt.SortOrder.AscendingOrder)

    def _get_anchors_from_table(self, type_filter=None):  # Updated logic (Error handling)
            anchors = [];
            if not hasattr(self, 'm4_anchor_table'): return anchors
            for r in range(self.m4_anchor_table.rowCount()):
                try:
                    item_t = self.m4_anchor_table.item(r, 1);
                    item_p = self.m4_anchor_table.item(r, 0)
                    if item_t and item_p:
                        t = item_t.data(Qt.ItemDataRole.UserRole);
                        p = item_p.data(Qt.ItemDataRole.UserRole)
                        if isinstance(p, (int, float)) and (type_filter is None or t == type_filter): anchors.append(
                            (float(p), t))
                except Exception as e:
                    print(f"读取锚点行 {r} 失败: {e}")
            return anchors

    def _update_plot_anchors(self):  # Updated logic
            if hasattr(self, 'm4_anchor_table'):
                anchors = [a[0] for a in self._get_anchors_from_table()]
                self.request_plot_anchors_update.emit(anchors)

    @Slot()
    def on_m4_anchor_table_sync(self):
            self._update_plot_anchors()  # Unchanged

    @Slot()
    def on_delete_selected_anchor(self):  # Updated logic
            if not hasattr(self, 'm4_anchor_table'): return
            rows = sorted([idx.row() for idx in self.m4_anchor_table.selectionModel().selectedRows()], reverse=True)
            if not rows: self.add_to_log("提示: 请先选中要删除的行。"); return
            self.m4_anchor_table.blockSignals(True);  # Block signals
            for r in rows: self.m4_anchor_table.removeRow(r)
            self.m4_anchor_table.blockSignals(False);  # Unblock
            self.add_to_log(f"删除 {len(rows)} 个锚点。");
            self._update_plot_anchors()

    @Slot(float)
    def on_anchor_deleted_from_plot(self, x_pos_to_delete):  # Updated logic (Signal blocking)
            if not hasattr(self, 'm4_anchor_table'): return
            self.add_to_log(f"请求从图上删除锚点 {x_pos_to_delete:.2f}...")
            deleted = False;
            tol = 1e-3;
            row_to_delete = -1
            # Find row
            for r in range(self.m4_anchor_table.rowCount()):
                try:
                    item_p = self.m4_anchor_table.item(r, 0);
                    stored_pos = item_p.data(Qt.ItemDataRole.UserRole);
                except:
                    continue
                if isinstance(stored_pos, float) and np.isclose(stored_pos, x_pos_to_delete,
                                                                atol=tol): row_to_delete = r; break
            # Delete if found
            if row_to_delete != -1:
                try:
                    self.m4_anchor_table.blockSignals(True)
                    pos_val = self.m4_anchor_table.item(row_to_delete, 0).data(Qt.ItemDataRole.UserRole)
                    self.m4_anchor_table.removeRow(row_to_delete);
                    self.m4_anchor_table.blockSignals(False)
                    self.add_to_log(f"已删除锚点: {pos_val:.3f}");
                    deleted = True
                except Exception as e:
                    self.add_to_log(
                        f"删除行 {row_to_delete} 出错: {e}");
                    traceback.print_exc();
                    self.m4_anchor_table.blockSignals(
                        False)
            if deleted:
                self._update_plot_anchors()
            else:
                self.add_to_log("未找到匹配锚点。")

    @Slot()
    def on_clear_anchors_auto(self):  # Updated logic
            if not hasattr(self, 'm4_anchor_table'): return
            rows = [r for r in range(self.m4_anchor_table.rowCount()) if
                    getattr(self.m4_anchor_table.item(r, 1), 'data', lambda r: None)(
                        Qt.ItemDataRole.UserRole) == "Auto"]
            self.m4_anchor_table.blockSignals(True)
            for r in sorted(rows, reverse=True): self.m4_anchor_table.removeRow(r)
            self.m4_anchor_table.blockSignals(False)
            self._update_plot_anchors();
            self.add_to_log("清空自动峰")

    @Slot()
    def on_clear_anchors_all(self):  # Updated logic
            if hasattr(self, 'm4_anchor_table'): self.m4_anchor_table.setRowCount(
                0); self._update_plot_anchors(); self.add_to_log("清空所有锚点")

        # --- Config Getters (保持不变) ---
    def _get_m4_find_config(self, manual_anchors=[]):  # Fixed
            if not self.current_project: self.add_to_log("Err: No project"); return None
            try:
                xd = self.current_project.wavelengths;
                pr = self.m4_current_regions
                if not pr: pr = [(xd.min(), xd.max())]
                # Add checks for control existence
                smooth = getattr(self, 'm4_autofind_smooth_cb') and self.m4_autofind_smooth_cb.isChecked() or False
                thresh = getattr(self, 'm4_height_thresh_input') and self.m4_height_thresh_input.value() or 10.0
                dist = getattr(self, 'm4_peak_dist_input') and self.m4_peak_dist_input.value() or 15
                tol = getattr(self, 'm4_tolerance_input') and self.m4_tolerance_input.value() or 10.0
                max_p = getattr(self, 'm4_max_peaks_input') and self.m4_max_peaks_input.value() or None
                return {'manual_anchors': manual_anchors, 'use_smoothing': smooth, 'peak_height_threshold': thresh,
                        'peak_distance': dist, 'peak_tolerance': tol, 'peak_regions': pr, 'max_peaks_to_find': max_p}
            except Exception as e:
                self.add_to_log(f"Err get find config:{e}");
                traceback.print_exc();
                return None

    def _get_m4_fit_config(self):  # Fixed
            if not self.current_project: self.add_to_log("Err: No project"); return None
            try:
                xd = self.current_project.wavelengths;
                pr = self.m4_current_regions
                if not pr: pr = [(xd.min(), xd.max())]
                ps = getattr(self, 'm4_shape_combo') and self.m4_shape_combo.currentText().lower() or 'voigt'
                shift = getattr(self, 'm4_center_shift_input') and self.m4_center_shift_input.value() or 10.0
                return {'peak_shape': ps, 'center_shift_limit': shift, 'peak_regions': pr, 'bounds_multiplier': 2.5,
                        'max_sigma': 50., 'max_gamma': 50., 'maxfev': 20000}
            except Exception as e:
                self.add_to_log(f"Err get fit config:{e}");
                traceback.print_exc();
                return None

        # --- M4 Worker Launchers & Handlers (保持不变) ---
    @Slot()
    def on_find_auto_peaks(self):
            if not self.current_project:
                return;
            idx = self.project_manager.get_current_selected_index();
            if idx == -1:
                return
            y_active = self.current_project.get_active_spectrum_by_index(idx)
            if y_active is None:
                self.add_to_log("错误: 无法获取激活数据");
                return
            x = self.current_project.wavelengths;
            manual = [a[0] for a in self._get_anchors_from_table("Manual")]
            config = self._get_m4_find_config(manual_anchors=manual);
            if config is None:
                return
            self._set_m4_buttons_enabled(False);
            self.add_to_log(f"后台自动寻峰...")
            worker = AutoFindWorker(x, y_active, manual, config);
            worker.s.finished.connect(self.on_auto_find_complete);
            worker.s.error.connect(self.on_worker_error);
            self.thread_pool.start(worker)

    @Slot(dict)
    def on_auto_find_complete(self, results):  # Simplified
            self._set_m4_buttons_enabled(True);
            a = results.get('auto_anchors');
            e = results.get('error_msg')
            if e: self.add_to_log(f"AutoFind Fail: {e}"); return
            if not a: self.add_to_log("未找到额外峰."); return
            for x_pos in a: self._add_anchor_to_table(x_pos, "Auto", QColor("lightblue"))
            self._update_plot_anchors()

    @Slot()
    def on_run_peak_fit_preview(self):  # 修正后的版本
            # 检查 1: 项目是否存在
            if not self.current_project:
                self.add_to_log("错误: 未加载项目")  # 添加日志更友好
                return

            # 检查 2: 是否选中了样本
            idx = self.project_manager.get_current_selected_index()
            if idx == -1:
                self.add_to_log("错误: 请先选择一个样本进行预览")  # 添加日志
                return

            # --- 原有代码继续 ---
            y_active = self.current_project.get_active_spectrum_by_index(idx)
            if y_active is None: self.add_to_log("错误: 无法获取激活数据"); return
            x = self.current_project.wavelengths;
            anchors = sorted([a[0] for a in self._get_anchors_from_table()])
            if not anchors: self.add_to_log("错误: 锚点列表为空."); return
            config = self._get_m4_fit_config();
            if config is None: return
            self._set_m4_buttons_enabled(False);
            self.add_to_log(f"后台预览拟合...")
            worker = FitWorker(x, y_active, anchors, config);
            worker.s.finished.connect(self.on_fit_complete);
            worker.s.error.connect(self.on_worker_error);
            self.thread_pool.start(worker)

    @Slot(dict)
    def on_fit_complete(self, results):  # Simplified
            self._set_m4_buttons_enabled(True);
            p = results.get('popt');
            fy = results.get('fit_y');
            s = results.get('sse');
            e = results.get('error_msg');
            ps = results.get('peak_shape')
            if e: self.add_to_log(f"Fit Fail: {e}"); return
            if p is None or fy is None: self.add_to_log("Fit Fail: 无结果"); return
            self.add_to_log(f"Fit OK. SSE: {s:.4e}")
            try:
                df = parse_popt_to_dataframe(p, ps);
                self.add_to_log("--- Fit Params ---");
                self.add_to_log(
                    df.to_string(float_format="%.3f"));
                self.add_to_log("---")
            except Exception as ex:
                self.add_to_log(f"Error parsing popt: {ex}")
            idx = self.project_manager.get_current_selected_index();
            title = f"Smp {idx} ({ps} Fit)"
            if hasattr(self.plot_widget, 'plot_peak_fitting_results') and self.current_project:
                x_data = self.current_project.wavelengths;
                y_data_active = self.current_project.get_active_spectrum_by_index(idx)
                if y_data_active is not None:
                    self.plot_widget.plot_peak_fitting_results(
                        x_data, y_data_active, fy,
                        params_df=df,  # <--- 传递 DataFrame
                        peak_shape=ps,  # <--- 传递 peak_shape
                        title=title
                    )
                else:
                    self.add_to_log("错误: 无法获取激活数据绘图")
            else:
                self.add_to_log("Error: plot func/project missing.")
            # (在 on_run_batch_fit 函数之前)

    @Slot()
    def on_run_batch_apply(self):
        """
        (新增) "应用到勾选" 按钮的唯一入口。
        它检查选择的引擎并调用适当的拟合函数。
        """
        engine_index = self.m4_fit_engine_combo.currentIndex()

        if engine_index == 0:
            # 0 = SciPy (多线程)
            self.add_to_log("--- 启动 [SciPy (多线程)] 批量拟合 ---")
            self.on_run_batch_fit()  # 调用原始的拟合函数
        elif engine_index == 1:
            # 1 = PyTorch (DL批量)
            self.add_to_log("--- 启动 [PyTorch (DL批量)] 批量拟合 ---")
            self.on_run_batch_fit_dl()  # 调用新的 DL 拟合函数
        else:
            self.add_to_log(f"错误: 未知的拟合引擎索引 {engine_index}")

    @Slot()
    def on_run_batch_fit_dl(self):
        """
        (新增) 启动 PyTorch DL 批量拟合任务。
        """
        # --- 1. 禁用按钮 (防止双击) ---
        self._set_m4_buttons_enabled(False)

        # --- 2. 获取所有通用数据 ---
        try:
            if not self.current_project:
                raise ValueError("未加载项目。")
            indices = self.project_manager.get_checked_items_indices()
            if not indices:
                raise ValueError("未勾选样本。")

            cfg_fit = self._get_m4_fit_config()
            if cfg_fit is None:
                raise ValueError("无法获取 M4 拟合配置。")

            self.m4_batch_input_version = self.current_project.active_spectra_version
            active_data = self.current_project.get_active_spectra()
            if active_data is None:
                raise ValueError(f"无法获取版本 '{self.m4_batch_input_version}' 数据")

            # --- [新增/修改] 确定输出名称 ---
            user_name = self.m4_output_name_input.text().strip()
            if user_name:
                self.m4_current_output_name = user_name
            else:
                timestamp = int(time.time())
                self.m4_current_output_name = f"{self.m4_batch_input_version}_DL_{timestamp}"

            if self.m4_current_output_name in self.m4_batch_fit_results:
                reply = QMessageBox.question(self, "覆盖确认",
                                             f"结果名称 '{self.m4_current_output_name}' 已存在。\n是否覆盖？",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    self._set_m4_buttons_enabled(True)
                    return

            self.add_to_log(f"--- 准备 DL 批量拟合 {len(indices)} 样本 (输出名称: '{self.m4_current_output_name}') ---")

            # (与 on_run_batch_fit 相同的锚点确定逻辑)
            final_anchors = []
            manual_anchors = [a[0] for a in self._get_anchors_from_table(type_filter='Manual')]
            use_autofind = getattr(self, 'm4_use_autofind_cb', None) and self.m4_use_autofind_cb.isChecked()
            self.m4_batch_cfg_fit = cfg_fit
            if use_autofind:
                cfg_f = self._get_m4_find_config(manual_anchors=manual_anchors)
                if cfg_f is None: raise ValueError("无法获取自动寻峰配置。")
                self.m4_batch_cfg_find = cfg_f
                first_idx = indices[0]
                x_repr, y_repr = self.current_project.wavelengths, active_data[first_idx]

                from core.peak_fitting_core import find_auto_anchors
                auto_anchors, find_err = find_auto_anchors(x_repr, y_repr, manual_anchors, cfg_f)
                if find_err: raise RuntimeError(f"自动寻峰失败: {find_err}")
                combined_anchors_set = set(manual_anchors) | set(auto_anchors)
                final_anchors = sorted(list(combined_anchors_set))
            else:
                final_anchors = sorted(manual_anchors)

            if not final_anchors:
                raise ValueError("最终锚点列表为空，无法进行拟合。")
            self.m4_batch_final_anchors = final_anchors

        except Exception as e_prep:
            self.add_to_log(f"错误: 准备 DL 拟合失败: {e_prep}")
            traceback.print_exc()
            self._set_m4_buttons_enabled(True)
            return

        # --- 3. 准备批量数据 ---
        try:
            x_axis_full_np = self.current_project.wavelengths
            y_batch_full_np = active_data[indices, :]
        except Exception as e_data:
            self.add_to_log(f"错误: 提取批量数据失败: {e_data}")
            self._set_m4_buttons_enabled(True)
            return

        # --- 4. 获取 DL 特定配置 ---
        cfg_dl = {
            'dl_epochs': self.m4_dl_epochs_input.value(),
            'dl_learning_rate': self.m4_dl_lr_input.value(),
            'dl_min_sigma': self.m4_dl_min_sigma_input.value(),
            'dl_max_sigma': self.m4_dl_max_sigma_input.value(),
            'dl_min_gamma': self.m4_dl_min_gamma_input.value(),
            'dl_max_gamma': self.m4_dl_max_gamma_input.value(),
            'dl_use_l2_reg': self.m4_dl_use_l2_cb.isChecked(),
            'dl_l2_lambda': self.m4_dl_l2_lambda_input.value(),
            'dl_print_interval': 100,
            'default_sigma': 5.0,
            'default_gamma': 5.0,
            'default_eta': 0.5
        }

        # --- 5. 重置状态并启动 Worker ---
        self.add_to_log(f"步骤 2: 开始为 {len(indices)} 个样本启动 [单个] DL 拟合线程...")
        self.fitting_stop_requested = False
        self.m4_stop_batch_button.setEnabled(True)
        self.m4_temp_fit_results.clear()
        self.m4_temp_fit_curves.clear()
        self.m4_batch_counter = 0
        self.active_workers.clear()
        self.m4_batch_total = len(indices)
        self.m4_batch_start_time = time.time()

        # [修改] 实例化 Worker (不再传递 log_callback_signal)
        worker = DLBatchFitWorker(
            indices_in_batch=indices,
            x_axis_full=x_axis_full_np,
            y_batch_full=y_batch_full_np,
            final_anchors=self.m4_batch_final_anchors,
            cfg_fit=self.m4_batch_cfg_fit,
            cfg_dl=cfg_dl
        )

        # [新增] 连接日志信号
        worker.s.log.connect(self.add_to_log)

        # (保持不变) 连接完成和错误信号
        worker.s.finished.connect(self.on_batch_fit_complete)
        worker.s.error.connect(self.on_worker_error)

        self.active_workers[0] = worker
        self.thread_pool.start(worker)

    @Slot()
    def on_run_batch_fit(self):
        if not self.current_project:
            self.add_to_log("错误: 未加载项目。")
            return
        indices = self.project_manager.get_checked_items_indices()
        if not indices:
            self.add_to_log("错误: 未勾选样本。")
            return

        # --- 获取配置 ---
        cfg_fit = self._get_m4_fit_config()
        if cfg_fit is None: return
        # (find_config 现在只在启用自动寻峰时需要)
        cfg_f = self._get_m4_find_config(manual_anchors=[])

        # 获取当前激活版本的数据
        self.m4_batch_input_version = self.current_project.active_spectra_version
        active_data = self.current_project.get_active_spectra()
        if active_data is None:
            self.add_to_log(f"错误: 无法获取版本 '{self.m4_batch_input_version}' 数据")
            return

        # --- [新增/修改] 确定输出名称 ---
        user_name = self.m4_output_name_input.text().strip()
        if user_name:
            self.m4_current_output_name = user_name
        else:
            # 自动生成: 输入版本_峰形_时间戳
            shape = self.m4_shape_combo.currentText()
            timestamp = int(time.time())
            self.m4_current_output_name = f"{self.m4_batch_input_version}_{shape}_{timestamp}"

        # 简单检查名称是否已存在 (防止意外覆盖)
        if self.m4_current_output_name in self.m4_batch_fit_results:
            reply = QMessageBox.question(self, "覆盖确认",
                                         f"结果名称 '{self.m4_current_output_name}' 已存在。\n是否覆盖？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        self.add_to_log(f"--- 准备批量拟合 {len(indices)} 样本 (输出名称: '{self.m4_current_output_name}') ---")

        try:
            # --- VVVV 新逻辑: 确定最终锚点 (循环外) VVVV ---
            final_anchors = []
            manual_anchors = [a[0] for a in self._get_anchors_from_table(type_filter='Manual')]
            use_autofind = getattr(self, 'm4_use_autofind_cb', None) and self.m4_use_autofind_cb.isChecked()

            self.m4_batch_cfg_fit = cfg_fit

            if use_autofind:
                self.add_to_log("步骤 1: 启用自动寻峰...")
                if cfg_f is None: raise ValueError("无法获取自动寻峰配置。")

                cfg_f['manual_anchors'] = manual_anchors
                self.m4_batch_cfg_find = cfg_f

                first_idx = indices[0]
                try:
                    x_repr, y_repr = self.current_project.wavelengths, active_data[first_idx]
                except IndexError:
                    raise ValueError(f"无法获取第一个选中样本 (索引 {first_idx}) 的数据。")

                if not peak_fitting_core_available: raise ImportError("peak_fitting_core missing")
                from core.peak_fitting_core import find_auto_anchors
                auto_anchors, find_err = find_auto_anchors(x_repr, y_repr, manual_anchors, cfg_f)

                if find_err:
                    raise RuntimeError(f"自动寻峰失败: {find_err}")
                self.add_to_log(f"自动寻峰找到 {len(auto_anchors)} 个新锚点 (已排除与手动重叠)。")

                combined_anchors_set = set(manual_anchors) | set(auto_anchors)
                final_anchors = sorted(list(combined_anchors_set))
                self.add_to_log(f"合并后最终锚点数: {len(final_anchors)}")

            else:
                self.add_to_log("步骤 1: 禁用自动寻峰，仅使用手动锚点。")
                final_anchors = sorted(manual_anchors)
                self.add_to_log(f"手动锚点数: {len(final_anchors)}")

            if not final_anchors:
                raise ValueError("最终锚点列表为空，无法进行拟合。请添加手动锚点或启用自动寻峰。")
            self.add_to_log(f"最终锚点位置: {[f'{a:.2f}' for a in final_anchors]}")
            self.m4_batch_final_anchors = final_anchors

        except Exception as e_anchor:
            self.add_to_log(f"错误: 确定最终锚点失败: {e_anchor}")
            traceback.print_exc()
            return

        # --- VVVV 启动 Worker (循环内) VVVV ---
        self.add_to_log(f"步骤 2: 开始为 {len(indices)} 个样本启动拟合线程...")
        self.fitting_stop_requested = False
        self.m4_stop_batch_button.setEnabled(True)
        self.m4_temp_fit_results.clear()
        self.m4_temp_fit_curves.clear()
        self.m4_batch_counter = 0
        self.active_workers.clear()
        self.m4_batch_total = len(indices)
        self.m4_batch_start_time = time.time()
        self._set_m4_buttons_enabled(False)

        for idx in indices:
            if self.fitting_stop_requested: self.add_to_log("批量中止."); break
            try:
                x, y = self.current_project.wavelengths, active_data[idx]
            except Exception as e:
                self.add_to_log(f"启动 Smp {idx} Worker 失败 (获取数据): {e}")
                self.m4_batch_counter += 1
                continue

            worker = BatchFitWorker(idx, x, y, self.m4_batch_final_anchors, self.m4_batch_cfg_fit)
            worker.s.finished.connect(self.on_batch_fit_complete)
            worker.s.error.connect(self.on_worker_error)
            self.active_workers[idx] = worker
            self.thread_pool.start(worker)

        if self.fitting_stop_requested and self.m4_batch_counter < self.m4_batch_total:
            self.add_to_log("等待任务完成或中止...")
        elif not self.active_workers and self.m4_batch_counter == self.m4_batch_total:
            self.add_to_log("未启动任何有效拟合任务。")
            self.on_batch_finish()

    @Slot(dict)
    def on_batch_fit_complete(self, results):
        """
        处理单个样本拟合完成的信号。
        包含修复：处理手动停止后的状态同步问题。
        """
        self.m4_batch_mutex.lock()
        try:
            # [关键修复] 如果 active_workers 为空且处于停止请求状态，
            # 说明在 on_stop_fitting 中已经清空了列表并重置了 UI。
            # 这意味着当前这个信号来自于一个被“遗弃”的 worker（在停止操作前已经启动但刚跑完）。
            # 我们应该直接忽略它，不要更新计数器，也不要再次调用 on_batch_finish。
            if not self.active_workers and self.fitting_stop_requested:
                return

            idx = results.get('sample_index')
            err = results.get('error')
            df = results.get('results_df')
            fy = results.get('fit_y')

            # 从活动任务列表中移除当前任务
            if idx in self.active_workers:
                del self.active_workers[idx]

            is_interrupt = isinstance(err, str) and "stopped" in err.lower()

            # 仅在非中断状态下（或虽然请求停止但任务正常完成了）处理结果
            if not self.fitting_stop_requested or not is_interrupt:
                log_prefix = f"({self.m4_batch_counter + 1}/{self.m4_batch_total}) Smp {idx}"

                if err and not is_interrupt:
                    self.add_to_log(f"{log_prefix} Fail: {err}")

                elif df is not None and fy is not None:
                    # 1. 存入临时结果字典
                    self.m4_temp_fit_results[idx] = df
                    self.m4_temp_fit_curves[idx] = fy

                    self.add_to_log(f"{log_prefix} OK.")

                    # 2. 保存残差数据 (保留原有逻辑)
                    try:
                        # 获取原始数据
                        y_original = self.current_project.get_active_spectrum_by_index(idx)
                        x_axis = self.current_project.wavelengths
                        if y_original is None:
                            raise ValueError("无法获取原始光谱数据")

                        # 计算残差
                        residual_data = y_original - fy

                        # 创建残差 DataFrame
                        residual_df = pd.DataFrame({
                            'wavenumber': x_axis,
                            'residual_intensity': residual_data
                        })

                        # 确定保存路径
                        current_cache_dir = self._get_project_cache_dir()
                        if current_cache_dir:
                            # 使用当前的输出名称作为子文件夹
                            residual_dir = os.path.join(current_cache_dir,
                                                        "fit_results",
                                                        f"residuals_{self.m4_current_output_name}")
                            os.makedirs(residual_dir, exist_ok=True)

                            # 生成文件名 (尝试包含浓度信息)
                            labels_df = self.current_project.labels_dataframe
                            conc_cols = [col for col, info in self.current_project.task_info.items() if
                                         info['role'] == 'target']
                            if not conc_cols:
                                conc_cols = [col for col in labels_df.columns if col.startswith('conc_')]
                            if not conc_cols:
                                conc_cols = labels_df.columns.tolist()

                            if conc_cols and idx in labels_df.index:
                                conc_vector_str = "_".join([
                                    f"{labels_df.loc[idx, col]}" for col in conc_cols
                                ])
                                filename = f"residual_conc_{conc_vector_str}.csv"
                            else:
                                filename = f"residual_idx_{idx}.csv"

                            # 写入文件
                            residual_filename = os.path.join(residual_dir, filename)
                            residual_df.to_csv(residual_filename, index=False, float_format='%.6f')

                    except Exception as e_save_res:
                        # 仅打印到控制台，避免 UI 日志刷屏
                        print(f"Smp {idx} 保存残差失败: {e_save_res}")

            # 3. 更新计数器
            self.m4_batch_counter += 1

            # 4. 检查是否所有任务都已处理完毕
            if self.m4_batch_counter >= self.m4_batch_total:
                self.on_batch_finish()

        except Exception as e:
            print(f"Error in batch complete: {e}")
            traceback.print_exc()
        finally:
            self.m4_batch_mutex.unlock()

    def _vis_refresh_data_sources(self):
            """
            (新增) 扫描缓存目录，填充 Page 1 的数据源下拉框。
            """
            self.add_to_log("M7/Vis: 正在扫描数据源...")
            if not hasattr(self, 'vis_data_source_combo'):
                self.add_to_log("  > 错误: vis_data_source_combo 不存在。")
                return

            vis_combo = self.vis_data_source_combo
            vis_combo.blockSignals(True)

            try:
                current_selection = vis_combo.currentText()
                vis_combo.clear()

                found_sources = {}  # { "Display Name": "folder/path" }

                # 1. 扫描主项目缓存
                if os.path.isdir(self.PROJECT_CACHE_ROOT):
                    for project_dir in os.listdir(self.PROJECT_CACHE_ROOT):
                        full_path = os.path.join(self.PROJECT_CACHE_ROOT, project_dir)
                        if os.path.isdir(full_path):
                            display_name = f"主项目: {project_dir}"
                            found_sources[display_name] = full_path

                # 2. 扫描 M5 增广缓存
                if os.path.isdir(self.AUGMENTED_OUTPUT_ROOT):
                    for aug_dir in os.listdir(self.AUGMENTED_OUTPUT_ROOT):
                        full_path = os.path.join(self.AUGMENTED_OUTPUT_ROOT, aug_dir)
                        # (M5 Worker 会在 aug_dir 内部创建 _spectra.csv 和 _labels.csv)
                        if os.path.isdir(full_path):
                            display_name = f"增广项目: {aug_dir}"
                            found_sources[display_name] = full_path

                if not found_sources:
                    self.add_to_log("  > 未找到任何缓存的数据源。")
                    vis_combo.addItem("无可用的缓存数据")
                    vis_combo.setEnabled(False)
                else:
                    vis_combo.addItems(sorted(found_sources.keys()))
                    vis_combo.setEnabled(True)
                    # (将文件夹路径存储在 QComboBox 的 UserData 中)
                    for i in range(vis_combo.count()):
                        text = vis_combo.itemText(i)
                        vis_combo.setItemData(i, found_sources[text])  # 存储路径

                    # 尝试恢复之前的选择
                    if current_selection in found_sources:
                        vis_combo.setCurrentText(current_selection)

                self.add_to_log(f"  > 找到 {len(found_sources)} 个数据源。")

            except Exception as e:
                self.add_to_log(f"M7/Vis: 扫描数据源失败: {e}")
                traceback.print_exc()
            finally:
                vis_combo.blockSignals(False)

    def _vis_load_project_from_folder(self, folder_path: str) -> Tuple[
        SpectralProject | None, Dict[str, pd.DataFrame]]:
        """
        (修改) 从给定的缓存文件夹路径加载一个完整的 SpectralProject
        并同时加载其 M4 拟合结果。

        (关键修改) 现在会优先从 'metadata.json' 读取 'task_info'。
        如果失败，才会回退到原始的（不准确的）猜测逻辑。

        返回: (project, m4_results_dict)
        """
        self.add_to_log(f"M7/Vis: 正在从 {folder_path} 加载数据...")
        m5_generated_params_dict = {}

        # --- 1. 加载 M4 拟合结果 (CSV) ---

        m4_results_dict = {}
        fit_cache_dir = os.path.join(folder_path, "fit_results")
        if os.path.isdir(fit_cache_dir):
            for csv_file in os.listdir(fit_cache_dir):
                if csv_file == "M5_GENERATED_PARAMS.csv":
                    m5_generated_params_dict = _load_m5_params_from_csv(os.path.join(fit_cache_dir, csv_file))
                elif csv_file.endswith(".csv"):
                    version_name = csv_file.replace(".csv", "")
                    file_path = os.path.join(fit_cache_dir, csv_file)
                    try:
                        m4_results_dict[version_name] = pd.read_csv(file_path)
                        self.add_to_log(f"  > 已加载 M4 结果: {csv_file}")
                    except Exception as e_csv:
                        self.add_to_log(f"  ! 错误: 加载 M4 结果 {csv_file} 失败: {e_csv}")
        if m5_generated_params_dict:
            self.add_to_log(f"  > 已加载 M5 生成的峰参数 ({len(m5_generated_params_dict)} 个样本)。")

        # --- 2. 加载光谱版本 (NPY) ---
        # (此部分保持不变)
        spectra_versions_dict = {}
        spectra_cache_dir = os.path.join(folder_path, "spectra")
        if os.path.isdir(spectra_cache_dir):
            for npy_file in os.listdir(spectra_cache_dir):
                if npy_file.endswith(".npy"):
                    version_name = npy_file.replace(".npy", "")
                    file_path = os.path.join(spectra_cache_dir, npy_file)
                    try:
                        spectra_versions_dict[version_name] = np.load(file_path)
                        self.add_to_log(f"  > 已加载光谱版本: {npy_file}")
                    except Exception as e_npy:
                        self.add_to_log(f"  ! 错误: 加载光谱 {npy_file} 失败: {e_npy}")

        if 'original' not in spectra_versions_dict:
            self.add_to_log("  ! 警告: 在 'spectra/' 中未找到 'original.npy'。")

        # --- 3. 加载基础文件 (CSV / NPY / JSON) ---
        try:
            wavelengths = np.load(os.path.join(folder_path, "wavelengths.npy"))
            labels_df = pd.read_csv(os.path.join(folder_path, "labels.csv"))

            # --- VVVV (!!! 关键修改 !!!) VVVV ---
            # (不再使用 "智能猜测" 逻辑)
            # (我们将读取 Page 0 保存的 metadata.json)

            task_info = {}
            meta_path = os.path.join(folder_path, "metadata.json")

            try:
                import json  # (在此处导入 json 模块)
                with open(meta_path, 'r', encoding='utf-8') as f:
                    task_info = json.load(f)
                if not task_info:
                    raise ValueError("metadata.json 为空")
                self.add_to_log("  > 成功: 从 metadata.json 加载真实的 task_info。")

            except Exception as e_json:
                # (如果 metadata.json 丢失或损坏，我们将回退到原始的、不准确的猜测逻辑)
                self.add_to_log(f"  ! 警告: 无法从 metadata.json 加载 task_info ({e_json})。")
                self.add_to_log("  ! ...正在回退到基于 labels.csv 的 (不准确的) 猜测。")

                # (这是 main.py 中原始的回退猜测逻辑)
                task_info = {}
                for col in labels_df.columns:
                    is_numeric = pd.api.types.is_numeric_dtype(labels_df[col])
                    task_info[col] = {'role': 'target', 'type': 'regression' if is_numeric else 'classification'}

                self.add_to_log("  > 已使用 (回退的) 猜测逻辑重建 task_info。")
            processing_history = []  # 默认为空
            history_path = os.path.join(folder_path, "processing_history.json")
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    processing_history = json.load(f)
                if not processing_history:
                    self.add_to_log("  > 警告: processing_history.json 为空。")
                self.add_to_log(f"  > 成功: 从 history.json 加载 {len(processing_history)} 个历史条目。")
            except Exception as e_hist_load:
                self.add_to_log(f"  ! 警告: 无法加载 processing_history.json ({e_hist_load})。")


        except FileNotFoundError as e_base:
            self.add_to_log(f"  ! 严重错误: 缺少基础文件 (e.g., wavelengths.npy): {e_base}")
            return None, {}
        except Exception as e_load:
            self.add_to_log(f"  ! 严重错误: 加载基础文件失败: {e_load}")
            return None, {}

        # --- 4. 构建 Project 对象 ---
        # (此部分保持不变)
        try:
            vis_project = SpectralProject(
                wavelengths=wavelengths,
                labels_dataframe=labels_df,
                task_info=task_info,
                data_file_path=os.path.join(folder_path, "spectra/original.npy"),  # 指向缓存
                label_file_path=os.path.join(folder_path, "labels.csv"),
                spectra_versions=spectra_versions_dict,
                active_spectra_version='original' if 'original' in spectra_versions_dict else '',
                processing_history=processing_history,
                generated_peak_params=m5_generated_params_dict
            )
            self.add_to_log("  > 成功在内存中重建项目 (包含历史记录)。")
            return vis_project, m4_results_dict

        except Exception as e_obj:
            self.add_to_log(f"  ! 严重错误: 重建 SpectralProject 对象失败: {e_obj}")
            return None, {}

    def on_batch_finish(self):
        duration = time.time() - self.m4_batch_start_time
        self.add_to_log(f"--- 批量拟合完成 ---")
        if self.fitting_stop_requested: self.add_to_log(f"处理因中止。")

        success_count = len(self.m4_temp_fit_results)

        # [修改] 使用新的输出名称
        output_version_name = self.m4_current_output_name

        self.add_to_log(
            f"结果 '{output_version_name}' 成功: {success_count}/{self.m4_batch_total}. 耗时: {duration:.2f}s.")

        current_cache_dir = self._get_project_cache_dir()
        if not current_cache_dir:
            self.add_to_log("!! 严重错误: 无法获取项目缓存目录，无法保存 M4 结果。")

        # (创建拟合曲线版本的逻辑)
        if success_count > 0 and self.m4_temp_fit_curves:
            self.add_to_log(f"正在创建拟合曲线数据版本: {output_version_name} ...")
            try:
                # 1. 获取原始数据作为模板 (依然基于 input_version)
                input_spectra = self.current_project.spectra_versions.get(self.m4_batch_input_version)
                if input_spectra is None:
                    raise ValueError(f"找不到输入版本 {self.m4_batch_input_version} 来创建曲线版本")

                # 2. 复制模板
                new_version_data = input_spectra.copy()

                # 3. 填充曲线
                for idx, curve_data in self.m4_temp_fit_curves.items():
                    if 0 <= idx < len(new_version_data):
                        new_version_data[idx] = curve_data

                # 4. [修改] 使用用户指定的 output_version_name 作为新版本名
                out_name = output_version_name
                # (如果需要唯一性检查，之前在 start 时已经做过，这里假设是唯一的或覆盖的)

                # 5. 创建历史记录
                hist = {
                    'step': 'FitCurveGeneration',
                    'params': self._get_m4_fit_config(),
                    'input_version': self.m4_batch_input_version,
                    'output_version': out_name,
                    'indices_processed': list(self.m4_temp_fit_curves.keys())
                }

                # 6. 添加新版本
                self.current_project.add_spectra_version(out_name, new_version_data, hist)
                self.add_to_log(f"成功创建拟合曲线版本: '{out_name}'")

                if current_cache_dir:
                    try:
                        history_path = os.path.join(current_cache_dir, "processing_history.json")
                        with open(history_path, 'w', encoding='utf-8') as f:
                            json.dump(self.current_project.processing_history, f, ensure_ascii=False, indent=4)
                    except Exception as e_hist_save:
                        self.add_to_log(f"  > 警告: 写入 history.json 缓存失败: {e_hist_save}")

                # 7. (!!!) 写入拟合曲线 .NPY 缓存
                if current_cache_dir:
                    spectra_cache_dir = os.path.join(current_cache_dir, "spectra")
                    os.makedirs(spectra_cache_dir, exist_ok=True)
                    file_path = os.path.join(spectra_cache_dir, f"{out_name}.npy")
                    np.save(file_path, new_version_data)
                    self.add_to_log(f"  > 已将拟合曲线 {out_name} 写入缓存。")

                # 8. (!!!) [关键修改] 存储 M4 结果
                # 使用 output_version_name 作为 Key，而不是 input_version
                self.m4_batch_fit_results[output_version_name] = self.m4_temp_fit_results.copy()
                self.add_to_log(f"  > M4 参数已关联到名称: {output_version_name}")

                # 9. (!!!) [关键修改] 写入 M4 参数 .CSV 缓存
                if current_cache_dir:
                    # 传入 output_version_name 作为文件名
                    self._save_m4_results_to_disk(output_version_name, self.m4_temp_fit_results)
                else:
                    self.add_to_log("  ! 警告: 无法写入参数 .csv 缓存（无缓存目录）。")

            except Exception as e:
                self.add_to_log(f"错误: 创建拟合曲线版本或写入缓存失败: {e}")
                traceback.print_exc()

        # (清理 M4 临时存储 - 保持不变)
        self.m4_temp_fit_curves.clear()
        self.m4_temp_fit_results.clear()

        # (重置 UI 和状态)
        self._set_m4_buttons_enabled(True)
        self.m4_stop_batch_button.setEnabled(False)
        self.fitting_stop_requested = False
        self.active_workers.clear()
        self.add_to_log(f"刷新版本列表...")
        self._update_version_selector()
        if success_count == 0: self.add_to_log(f"警告: 此次拟合无成功样本。")

    @Slot()
    def on_stop_fitting(self):
        # 检查按钮是否存在且可用
        if not getattr(self, 'm4_stop_batch_button', None) or not self.m4_stop_batch_button.isEnabled():
            return

        self.fitting_stop_requested = True
        self.m4_stop_batch_button.setEnabled(False)
        self.add_to_log("停止请求...")

        # 1. 清除排队中的任务
        self.thread_pool.clear()

        # 2. 尝试停止正在运行的任务
        stopped = 0
        # 使用 list() 创建副本以避免遍历时字典变化（虽然这里只读不删，但在多线程下更安全）
        for w in list(self.active_workers.values()):
            if hasattr(w, 'stop'):
                w.stop()
                stopped += 1
        self.add_to_log(f"已清空等待队列。向 {stopped} 个运行中任务发送停止信号。")

        # 3. [关键修复] 强制执行完成逻辑以重置 UI
        # 这会清空 active_workers，并重新启用“开始拟合”按钮
        self.on_batch_finish()

    @Slot(str)
    def on_worker_error(self, error_message):  # Simplified
            self.add_to_log(f"严重线程错误: {error_message}");
            traceback.print_exc()
            self._set_m4_buttons_enabled(True);
            if hasattr(self, 'm4_stop_batch_button'): self.m4_stop_batch_button.setEnabled(False);
            self.fitting_stop_requested = True;
            self.active_workers.clear()
            if self.m4_batch_counter > 0 and self.m4_batch_counter < self.m4_batch_total: self.add_to_log(
                f"批量处理因错误中止。")
            QMessageBox.critical(self, "线程错误", error_message)

        # --- M6 Export Slot (保持不变) ---
    def _save_m4_results_to_disk(self, m4_input_version_name: str, results_dict: Dict[int, pd.DataFrame]):
        """
        (新增) 辅助函数：将 M4 拟合结果合并并保存到缓存 .csv 文件。
        此函数重用了 on_export_fit_results 的逻辑。
        """
        self.add_to_log(f"  > 正在将 {len(results_dict)} 个 M4 参数写入磁盘...")
        try:
            current_cache_dir = self._get_project_cache_dir()
            if not current_cache_dir:
                raise ValueError("无法获取当前项目缓存目录。")

            fit_cache_dir = os.path.join(current_cache_dir, "fit_results")
            os.makedirs(fit_cache_dir, exist_ok=True)

            # (!!!) 文件名基于 *输入* 版本 (e.g., "original_airpls.csv")
            save_path = os.path.join(fit_cache_dir, f"{m4_input_version_name}.csv")

            # --- (开始重用 on_export_fit_results 的逻辑) ---
            all_dfs = []
            labels_df = self.current_project.labels_dataframe

            for idx, df in results_dict.items():
                if df is not None and not df.empty:
                    df_copy = df.copy();
                    df_copy['sample_index'] = idx
                    try:
                        if not labels_df.empty and idx in labels_df.index:
                            label_info = labels_df.loc[idx]
                            for col in label_info.index:
                                if col not in df_copy.columns: df_copy[col] = label_info[col]
                    except Exception as e:
                        print(f"Warn: No labels for smp {idx}: {e}")
                    all_dfs.append(df_copy)

            if not all_dfs:
                self.add_to_log(f"  > 警告: M4 结果为空，无需写入 .csv。")
                return

            combined_df = pd.concat(all_dfs, ignore_index=True)

            # (重新排序逻辑 - 保持不变)
            cols = list(combined_df.columns)
            new_order = ['sample_index']
            label_cols_to_add = [c for c in labels_df.columns if c in cols and c not in new_order]
            new_order.extend(label_cols_to_add)
            remaining_cols_to_add = [c for c in cols if c not in new_order]
            new_order.extend(remaining_cols_to_add)
            try:
                combined_df = combined_df[new_order]
            except Exception as e:
                print(f"Warn: M4 缓存列重新排序失败: {e}")

            # 保存
            combined_df.to_csv(save_path, index=False, encoding='utf-8-sig');
            self.add_to_log(f"  > M4 参数已写入缓存: {save_path}")
            # --- (结束重用 on_export_fit_results 的逻辑) ---

        except Exception as e:
            self.add_to_log(f"  ! 严重错误: 保存 M4 参数到磁盘失败: {e}")
            traceback.print_exc()

    @Slot()
    def on_export_fit_results(self):  # Simplified Logic
            """Exports M4 batch fit results for the selected input version to CSV."""
            if not self.current_project or not self.m4_batch_fit_results: self.add_to_log(
                "错误: 无项目或无拟合结果。"); return
            combo = getattr(self, 'm6_fit_version_combo', None)
            selected_version = combo.currentText() if combo else None
            if not selected_version: self.add_to_log("错误: 请在 M6 选择版本。"); return
            results_for_version = self.m4_batch_fit_results.get(selected_version)
            if not results_for_version: self.add_to_log(f"错误: 版本 '{selected_version}' 无拟合结果。"); return

            self.add_to_log(f"准备导出 '{selected_version}' 的 {len(results_for_version)} 样本结果...")
            all_dfs = []
            labels_df = self.current_project.labels_dataframe if hasattr(self.current_project,
                                                                         'labels_dataframe') else pd.DataFrame()
            for idx, df in results_for_version.items():
                if df is not None and not df.empty:
                    df_copy = df.copy();
                    df_copy['sample_index'] = idx
                    try:  # Add labels safely
                        if not labels_df.empty and idx in labels_df.index:
                            label_info = labels_df.loc[idx]
                            for col in label_info.index:
                                if col not in df_copy.columns: df_copy[col] = label_info[col]
                    except Exception as e:
                        print(f"Warn: No labels for smp {idx}: {e}")
                    all_dfs.append(df_copy)
            if not all_dfs: self.add_to_log(f"错误: 版本 '{selected_version}' 无有效结果导出。"); return

            try:
                combined_df = pd.concat(all_dfs, ignore_index=True)
            except Exception as e:
                self.add_to_log(f"Err merging: {e}");
                traceback.print_exc();
                return

            # --- VVVV 修正 'new_order' 逻辑 VVVV ---
            # 修正: 将列表构建分解为多个步骤
            cols = list(combined_df.columns)
            new_order = ['sample_index']

            # 1. 添加标签列
            label_cols_to_add = [c for c in labels_df.columns if c in cols and c not in new_order]
            new_order.extend(label_cols_to_add)

            # 2. 添加所有剩余的列 (例如峰参数)
            remaining_cols_to_add = [c for c in cols if c not in new_order]
            new_order.extend(remaining_cols_to_add)

            try:
                combined_df = combined_df[new_order]
            except Exception as e:
                print(f"Warn: Column reordering failed: {e}")
                traceback.print_exc()
            # --- ^^^^ 修正结束 ^^^^ ---

            # Save
            default_fn = f"fit_results_{selected_version}.csv"
            save_path, _ = QFileDialog.getSaveFileName(self, "导出拟合结果 CSV", default_fn, "CSV (*.csv)")
            if save_path:
                try:
                    combined_df.to_csv(save_path, index=False, encoding='utf-8-sig');
                    self.add_to_log(
                        f"结果导出到: {save_path}")
                except Exception as e:
                    self.add_to_log(f"Err saving: {e}");
                    traceback.print_exc();
                    QMessageBox.critical(self, "Save Fail",
                                         f"无法保存:\n{e}")
            else:
                self.add_to_log("导出取消。")

    @Slot()
    def on_export_spectra(self):
        """
        导出选中的光谱数据版本到 CSV 或 NPY。
        如果是 CSV，会将标签数据合并在左侧，波长作为表头。
        (修改: 智能过滤，仅导出该版本实际处理过的样本)
        """
        if not self.current_project:
            self.add_to_log("错误: 无项目。")
            return

        version_name = self.m6_spectra_version_combo.currentText()
        if not version_name:
            self.add_to_log("错误: 未选择光谱版本。")
            return

        spectra_data = self.current_project.spectra_versions.get(version_name)
        if spectra_data is None:
            self.add_to_log(f"错误: 找不到版本 '{version_name}' 的数据。")
            return

        # --- VVVV 修改: 新增过滤逻辑 VVVV ---
        # 尝试从历史记录中获取该版本对应的有效样本索引
        # (例如: 批量拟合只处理了 5 个样本，这里就会返回这 5 个索引)
        valid_indices_set = self.current_project.get_processed_indices_for_version(version_name)

        final_spectra = spectra_data
        final_labels = self.current_project.labels_dataframe

        if valid_indices_set is not None:
            # 如果存在有效索引集合，则进行切片过滤
            sorted_indices = sorted(list(valid_indices_set))
            if sorted_indices:
                # 检查索引是否越界
                if sorted_indices[-1] < len(final_spectra):
                    self.add_to_log(
                        f"导出过滤: 版本 '{version_name}' 仅包含 {len(sorted_indices)} 个有效样本 (原始 {len(spectra_data)} 个)。")
                    final_spectra = final_spectra[sorted_indices]
                    if not final_labels.empty:
                        # 同时过滤标签，确保对应
                        final_labels = final_labels.iloc[sorted_indices]
                else:
                    self.add_to_log(f"警告: 版本 '{version_name}' 的索引记录越界，将导出全部数据。")
            else:
                self.add_to_log(f"提示: 版本 '{version_name}' 记录的有效索引为空，导出全部数据。")
        # --- ^^^^ 修改结束 ^^^^ ---

        # 默认文件名
        default_fn = f"spectra_{version_name}.csv"

        # 文件对话框
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出光谱数据",
            default_fn,
            "CSV Files (*.csv);;NumPy Files (*.npy)"
        )

        if not save_path:
            return

        self.add_to_log(f"正在导出光谱 '{version_name}' (最终导出 {final_spectra.shape[0]} 样本)...")

        try:
            if save_path.lower().endswith(".npy"):
                np.save(save_path, final_spectra)
                self.add_to_log(f"✅ 光谱数据 (NPY) 已导出至: {save_path}")
            else:
                # 导出为 CSV
                # 1. 准备波长作为列头
                wavelengths = self.current_project.wavelengths

                # 2. 创建光谱 DataFrame (列名为波长)
                df_spectra = pd.DataFrame(final_spectra, columns=wavelengths)

                # 3. 合并标签信息 (放在左侧以便阅读)
                if not final_labels.empty:
                    # 重置索引以确保对齐, 并保留原始索引作为 'Original_Index' 列，方便追溯
                    labels_reset = final_labels.reset_index()
                    labels_reset.rename(columns={'index': 'Original_Index'}, inplace=True)
                    df_export = pd.concat([labels_reset, df_spectra], axis=1)
                else:
                    df_export = df_spectra

                # 4. 保存
                df_export.to_csv(save_path, index=False, encoding='utf-8-sig')
                self.add_to_log(f"✅ 光谱数据 (CSV) 已导出至: {save_path}")

        except Exception as e:
            self.add_to_log(f"❌ 导出失败: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(e))

        # --- Other Slots (保持不变) ---
    def open_import_wizard(self):  # Simplified check
            if self.thread_pool.activeThreadCount() > 0: QMessageBox.warning(self, "Busy", "Wait"); return
            if not imports_ok: self.add_to_log("错误: 导入模块缺失"); return
            self.wizard = DataImportWizard(self);
            self.wizard.projectCreated.connect(self.load_project);
            self.wizard.log_message.connect(self.add_to_log);
            self.wizard.exec()

    @Slot(str)
    def add_to_log(self, msg):
            self.log_console.append(msg)  # Simplified

    def _update_version_selector(self):  # Refactored Update Logic
        """
        (修改) Helper to refresh ALL version selector comboboxes.
        (新增) 现在也会同步更新 Page 1 (可视化) 的版本下拉框。
        """

        combo_main = getattr(self.project_manager, 'version_selector_combo', None)
        combo_compare = getattr(self.project_manager, 'compare_version_combo', None)

        combo_m6_fit = getattr(self, 'm6_fit_version_combo', None)
        combo_m6_spec = getattr(self, 'm6_spectra_version_combo', None)
        combo_m5_param = getattr(self, 'm5_param_source_combo', None)
        combo_m5_mogp = getattr(self, 'm5_mogp_source_combo', None)
        btn_m6_fit = getattr(self, 'm6_export_fit_button', None)
        btn_m6_spec = getattr(self, 'm6_export_spectra_button', None)

        # 1. 清除所有找到的 combobox
        combos_to_update = [c for c in
                            [combo_main, combo_m6_fit, combo_m6_spec, combo_compare, combo_m5_param, combo_m5_mogp]
                            if c]

        for combo in combos_to_update:
            combo.blockSignals(True);
            current_text = combo.currentText()  # 保存当前选择
            combo.clear()
            combo.setProperty("saved_selection", current_text)  # 暂存

        fit_versions_with_results = []
        all_versions = []
        active_version = 'original'

        if self.current_project:
            all_versions = self.current_project.get_version_names()  # Get ordered names
            active_version = self.current_project.active_spectra_version
            # Get versions that have non-empty result dicts
            fit_versions_with_results = sorted([v for v, results in self.m4_batch_fit_results.items() if results])

        # --- 1. 更新 Main Selector ---
        try:
            if combo_main is None: raise AttributeError("'combo_main' not found")

            if all_versions:
                combo_main.addItems(all_versions)
                saved_sel = combo_main.property("saved_selection")
                if saved_sel in all_versions:
                    combo_main.setCurrentText(saved_sel)
                elif active_version in all_versions:
                    combo_main.setCurrentText(active_version)
                elif all_versions:
                    combo_main.setCurrentIndex(0);
                if self.current_project and not active_version in all_versions and all_versions: self.current_project.active_spectra_version = \
                    all_versions[0]
            combo_main.setEnabled(len(all_versions) > 0)
            combo_main.blockSignals(False)
        except AttributeError as e:
            print(f"!! 严重错误: 更新 'version_selector_combo' (主下拉框) 失败: {e}。")

        # --- 2. 更新 M6 Fit Selector ---
        try:
            if combo_m6_fit is None: raise AttributeError("'combo_m6_fit' not found")

            if fit_versions_with_results:
                combo_m6_fit.addItems(fit_versions_with_results)
                saved_sel = combo_m6_fit.property("saved_selection")
                if saved_sel in fit_versions_with_results:
                    combo_m6_fit.setCurrentText(saved_sel)
                elif active_version in fit_versions_with_results:
                    combo_m6_fit.setCurrentText(active_version)
                elif fit_versions_with_results:
                    combo_m6_fit.setCurrentIndex(0)
            combo_m6_fit.setEnabled(len(fit_versions_with_results) > 0)
            combo_m6_fit.blockSignals(False)
            if btn_m6_fit: btn_m6_fit.setEnabled(len(fit_versions_with_results) > 0)
        except AttributeError as e:
            print(f"!! 严重错误: 更新 'm6_fit_version_combo' (M6 拟合下拉框) 失败: {e}。")

        # --- 3. 更新 M6 Spectra Selector ---
        try:
            if combo_m6_spec is None: raise AttributeError("'combo_m6_spec' not found")

            if all_versions:
                combo_m6_spec.addItems(all_versions)
                saved_sel = combo_m6_spec.property("saved_selection")
                current_main_selection = combo_main.currentText() if combo_main and combo_main.count() > 0 else active_version

                if saved_sel in all_versions:
                    combo_m6_spec.setCurrentText(saved_sel)
                elif current_main_selection in all_versions:
                    combo_m6_spec.setCurrentText(current_main_selection)
                elif all_versions:
                    combo_m6_spec.setCurrentIndex(0)
            combo_m6_spec.setEnabled(len(all_versions) > 0)
            combo_m6_spec.blockSignals(False)

            if btn_m6_spec:
                btn_m6_spec.setEnabled(len(all_versions) > 0)
        except AttributeError as e:
            print(f"!! 严重错误: 更新 'm6_spectra_version_combo' (M6 光谱下拉框) 失败: {e}。")

        # --- 4. (新增) 更新 Compare Selector ---
        try:
            if combo_compare is None: raise AttributeError("'combo_compare' not found")

            current_compare_selection = combo_compare.property("saved_selection")  # 获取暂存的选择

            combo_compare.addItem("-- 无对比 --")  # 添加默认选项
            if all_versions:
                combo_compare.addItems(all_versions)

            # 尝试恢复之前的选择
            if current_compare_selection and combo_compare.findText(current_compare_selection) > -1:
                combo_compare.setCurrentText(current_compare_selection)

            combo_compare.setEnabled(len(all_versions) > 0)
            combo_compare.blockSignals(False)
        except AttributeError as e:
            print(f"!! 严重错误: 更新 'compare_version_combo' 失败: {e}")

        # --- 5. (新增) 更新 M5 Source Selectors ---
        m5_combos = [c for c in [combo_m5_param, combo_m5_mogp] if c is not None]
        for combo_m5 in m5_combos:
            try:
                if fit_versions_with_results:
                    combo_m5.addItems(fit_versions_with_results)
                    saved_sel = combo_m5.property("saved_selection")

                    if saved_sel in fit_versions_with_results:
                        combo_m5.setCurrentText(saved_sel)
                    elif active_version in fit_versions_with_results:
                        combo_m5.setCurrentText(active_version)
                    elif fit_versions_with_results:
                        combo_m5.setCurrentIndex(0)

                combo_m5.setEnabled(len(fit_versions_with_results) > 0)
            except Exception as e:
                print(f"!! 严重错误: 更新 M5 source combo '{combo_m5.objectName()}' 失败: {e}。")
            finally:
                combo_m5.blockSignals(False)

        # --- 6. (!!! 关键修复 !!!) ---
        # --- 主动更新 Page 1 (可视化) 的版本下拉框 ---
        try:
            # 检查 Page 1 是否存在并且正在查看主项目
            if (hasattr(self, 'vis_data_source_combo') and
                    self.vis_data_source_combo.currentText() == "当前主项目" and
                    hasattr(self, 'vis_project_manager')):

                print("[DEBUG UpdateSelector] Page 1 正在查看主项目，正在强制刷新其版本列表...")

                vis_version_combo = self.vis_project_manager.version_selector_combo
                # 保存 Page 1 当前选择的版本
                saved_vis_selection = vis_version_combo.currentText()

                # 使用 helper (它会清除并重新填充)
                # _vis_fill_version_combo 会使用 self.current_project (它刚刚被更新了)
                self._vis_fill_version_combo(self.current_project)

                # 尝试恢复 Page 1 的选择
                all_vis_versions = [vis_version_combo.itemText(i) for i in range(vis_version_combo.count())]

                if saved_vis_selection in all_vis_versions:
                    vis_version_combo.setCurrentText(saved_vis_selection)
                elif active_version in all_vis_versions:  # active_version 来自此函数开头
                    vis_version_combo.setCurrentText(active_version)

                print("[DEBUG UpdateSelector] Page 1 版本列表已刷新。")

        except AttributeError as e:
            print(f"警告: 自动更新 Page 1 版本下拉框失败 (控件可能不存在): {e}")
        except Exception as e:
            print(f"错误: 自动更新 Page 1 版本下拉框时发生意外错误: {e}")
            traceback.print_exc()
        # --- ^^^^ 修复结束 ^^^^ ---

    @Slot(SpectralProject)
    def load_project(self, project: SpectralProject):  # (修改后)
        """
        加载新的项目，并重置/初始化所有模块的状态和 *磁盘缓存*。
        """

        # 1. 停止所有正在运行的后台任务 (保持不变)
        if self.thread_pool.activeThreadCount() > 0:
            self.add_to_log("警告: 正在任务中加载新项目，将尝试停止当前任务...")
            self.on_stop_fitting()  # 尝试停止 M4
            if not self.thread_pool.waitForDone(1500):
                self.add_to_log("警告: 等待线程池停止超时。加载可能不稳定。")

        self.current_project = project

        n = project.spectra_versions['original'].shape[
            0] if project and 'original' in project.spectra_versions else 0
        if hasattr(self, 'msc_ref_index_input'):
            self.msc_ref_index_input.setRange(0, max(0, n - 1))

        # 2. 重置 M4/M5 状态
        self.m4_batch_fit_results.clear()
        self.m4_batch_counter = 0
        self.m4_batch_total = 0
        self.fitting_stop_requested = False
        if hasattr(self, 'm4_stop_batch_button'): self.m4_stop_batch_button.setEnabled(False)
        if hasattr(self, 'm4_batch_fit_curves'): self.m4_batch_fit_curves.clear()

        # (!!!) 关键修改 (删除 Page 1 共享状态) (!!!)
        # self.augmented_projects.clear()  # <--- (已删除)

        # (新增) 清空 Page 1 (可视化) 自己的私有 M4 缓存
        self.vis_m4_results_cache.clear()

        # (修改) 重置 Page 1 下拉框 (不再添加 "当前主项目")
        try:
            vis_combo = self.vis_data_source_combo
            vis_combo.blockSignals(True)
            vis_combo.clear()
            # (Page 1 将在激活时自己扫描文件夹)
            vis_combo.blockSignals(False)
        except AttributeError:
            print("[DEBUG LoadProject] vis_data_source_combo not found during reset.")
        except Exception as e:
            print(f"[DEBUG LoadProject] Error resetting vis_data_source_combo: {e}")

        # 启用可视化按钮 (因为主项目总是可查看)
        if self.vis_action: self.vis_action.setEnabled(True)

        # 3. 重置 M4 UI (保持不变)
        self.active_workers.clear()
        self.thread_pool.clear()  # 再次清空线程池
        self.on_clear_anchors_all()
        self.on_clear_regions()
        self._disable_all_plot_interactions()

        # 4. 初始化 M2 UI (保持不变)
        self._initialize_m2_tab()

        # --- VVVV (新增) 5. 初始化磁盘缓存 VVVV ---
        try:
            self.add_to_log("正在初始化项目缓存...")
            # (假设 _get_project_cache_dir 已按上一节所述添加)
            cache_dir = self._get_project_cache_dir(project)
            if cache_dir is None:
                raise ValueError("无法获取有效的项目缓存目录。")

            # (可选) 先清除此项目的旧缓存
            if os.path.isdir(cache_dir):
                self.add_to_log(f"找到旧缓存，正在删除: {cache_dir}")
                shutil.rmtree(cache_dir)

            # 创建所有子目录
            spectra_cache_dir = os.path.join(cache_dir, "spectra")
            fit_cache_dir = os.path.join(cache_dir, "fit_results")
            os.makedirs(spectra_cache_dir, exist_ok=True)
            os.makedirs(fit_cache_dir, exist_ok=True)

            # 写入基础文件
            np.save(os.path.join(cache_dir, "wavelengths.npy"), project.wavelengths)
            project.labels_dataframe.to_csv(os.path.join(cache_dir, "labels.csv"), index=False)

            # (!!!) 写入 'original' 光谱版本 (!!!)
            original_spectra = project.spectra_versions.get('original')
            if original_spectra is not None:
                np.save(os.path.join(spectra_cache_dir, "original.npy"), original_spectra)

            # (可选但推荐: 写入 metadata.json)
            try:
                import json
                meta_path = os.path.join(cache_dir, "metadata.json")
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(project.task_info, f, ensure_ascii=False, indent=4)
                self.add_to_log("  > 已写入 metadata.json (task_info)")

                history_path = os.path.join(cache_dir, "processing_history.json")

                with open(history_path, 'w', encoding='utf-8') as f:
                                       json.dump(project.processing_history, f, ensure_ascii=False, indent=4)
                self.add_to_log("  > 已写入 history.json (processing_history)")
            except Exception as e_json:
                self.add_to_log(f"  > 警告: 写入 metadata.json 失败: {e_json}")

            self.add_to_log(f"项目缓存已在 {cache_dir} 中初始化。")

        except Exception as e:
            self.add_to_log(f"严重错误: 创建项目缓存失败: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "缓存错误", f"创建项目缓存时出错:\n{e}")
        # --- ^^^^ (新增) 缓存创建结束 ^^^^ ---

        # 6. 更新 Page 0 (工作流) UI (保持不变)
        print("[DEBUG] load_project: Calling _update_version_selector...")
        self._update_version_selector()  # 刷新 M1-M6 下拉框
        print("[DEBUG] load_project: Loading project manager...")
        self.project_manager.load_project_data(project)
        self.plot_widget.clear_plot()
        self.setWindowTitle(f"框架 - [{project.data_file_path}]")
        self.add_to_log(f"项目加载: {project.data_file_path}")

        if hasattr(self, 'reset_action'): self.reset_action.setEnabled(True)

        # 7. 切换视图并自动选择 (保持不变)
        if self.workflow_action: self.workflow_action.setChecked(True)
        if self.main_stack: self.main_stack.setCurrentIndex(0)

        # 自动选择第一个样本
        if n > 0:
            if hasattr(self.project_manager, 'table_view') and self.project_manager.table_view.rowCount() > 0:
                self.project_manager.table_view.selectRow(0)
            elif hasattr(self.project_manager, 'tree_view'):
                tli = self.project_manager.tree_view.topLevelItem(
                    0) if self.project_manager.tree_view.topLevelItemCount() > 0 else None
                first_child = tli.child(0) if tli and tli.childCount() > 0 else None
                if first_child: self.project_manager.tree_view.setCurrentItem(first_child)

    @Slot(str)
    def on_version_selected(self, version_name: str):  # Simplified
            """Handle user selecting a different data version."""
            if self.current_project and version_name and version_name in self.current_project.spectra_versions:
                if version_name != self.current_project.active_spectra_version:
                    self.current_project.active_spectra_version = version_name
                    self.add_to_log(f"激活版本: '{version_name}'")
                    idx = self.project_manager.get_current_selected_index()
                    if idx != -1:
                        # --- VVVV 修改 VVVV ---
                        self.on_sample_selected(idx, force_replot=True)  # 强制重绘
                        # --- ^^^^ 修改结束 ^^^^ ---
                    else:
                        self.plot_widget.clear_plot()
            elif self.current_project and version_name:
                self.add_to_log(f"警告: 版本 '{version_name}' 不存在。")
                # Revert combo?
                active = self.current_project.active_spectra_version
                # --- VVVV 修改 VVVV ---
                # 指向 project_manager 内部的控件
                combo = getattr(self.project_manager, 'version_selector_combo', None)
                if combo and active in self.current_project.get_version_names():
                    combo.blockSignals(True);
                    combo.setCurrentText(active);
                    combo.blockSignals(False)
                # --- ^^^^ 修改结束 ^^^^ ---

    @Slot(int)
    def on_sample_selected(self, idx, force_replot=False):  # <-- 修改签名
            """Plot the active version and optionally a compare version."""
            if not self.current_project: return

            # (可选：添加逻辑以避免在样本未变化时重置M4, 但目前保持与原设计一致)
            # if not force_replot and self.project_manager.get_current_selected_index() == idx:
            #     return # 样本未变, 且非强制, 则不动作

            # 总是重置 M4 (保持不变)
            self.on_clear_anchors_all();
            self.on_clear_regions();
            self._disable_all_plot_interactions()

            # --- VVVV 核心修改 VVVV ---
            # 1. 获取对比版本名称
            compare_name = None
            if hasattr(self.project_manager, 'get_compare_version_name'):
                compare_name = self.project_manager.get_compare_version_name()

            # 2. 调用修改后的 plot_spectrum
            self.plot_widget.plot_spectrum(
                self.current_project,
                idx,
                compare_version_name=compare_name  # 传递新参数
            )
            # --- ^^^^ 修改结束 ^^^^ ---

            log_msg = f"加载样本 {idx} (版本: {self.current_project.active_spectra_version})"
            if compare_name:
                log_msg += f" (对比: {compare_name})"
            log_msg += ". M4 重置."
            self.add_to_log(log_msg)

            # Check for batch results for the *current active version*
            active_ver = self.current_project.active_spectra_version
            # Check existence using .get() for safety
            if self.m4_batch_fit_results.get(active_ver, {}).get(idx) is not None:
                self.add_to_log(f"提示: 样本 {idx} (版本: {active_ver}) 已有批量结果.")

    @Slot()
    def on_compare_version_selected(self):
            """当 '对比版本' 下拉框变化时，重绘当前样本"""
            if not self.current_project: return
            idx = self.project_manager.get_current_selected_index()

            if idx != -1:
                # 调用 on_sample_selected 并强制重绘
                self.on_sample_selected(idx, force_replot=True)
            else:
                # 如果没有样本被选中, 仅记录日志
                compare_name = self.project_manager.get_compare_version_name()
                if compare_name:
                    self.add_to_log(f"对比版本选为: {compare_name}")
                else:
                    self.add_to_log("对比已取消。")

    @Slot()
    def on_reset_project(self):
        """
        重置当前项目到 'original' 状态，并清空所有模块的状态。
        """
        if not self.current_project:
            self.add_to_log("错误: 没有项目可重置。")
            return

        # 1. 弹窗确认
        reply = QMessageBox.question(self, "确认重置项目",
                                     "您确定要清除所有已处理的数据版本和M4拟合结果吗？\n"
                                     "项目将恢复到刚导入时的 'original' 状态，M2伪标签也将被移除。\n"
                                     "此操作无法撤销。",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            self.add_to_log("重置操作已取消。")
            return

        self.add_to_log("--- 正在重置项目... ---")

        # 2. 停止所有工作线程
        self.on_stop_fitting()  # 停止 M4
        self.thread_pool.clear()
        if not self.thread_pool.waitForDone(1500):
            self.add_to_log("警告: 等待线程池停止超时。")

        try:
            # 3. 重置数据模型 (假设 data_model.py 已添加此方法)
            if not hasattr(self.current_project, 'reset_to_original'):
                raise RuntimeError("数据模型 (data_model.py) 缺少 'reset_to_original' 方法。")

            reset_ok = self.current_project.reset_to_original()
            if not reset_ok:
                raise RuntimeError("数据模型重置失败 (reset_to_original 返回 False)。")

            # 4. 重置 Main 窗口 M4/M5 状态
            self.m4_batch_fit_results.clear()
            self.m4_batch_counter = 0
            self.m4_batch_total = 0
            if hasattr(self, 'm4_batch_fit_curves'):
                self.m4_batch_fit_curves.clear()  # 清除拟合曲线
            self.active_workers.clear()

            # 5. 重置 M7 沙盒
            # self.augmented_projects.clear() # <--- (!!! 删除此行 !!!)

            # (改为清空 Page 1 的私有缓存)
            self.vis_m4_results_cache.clear()

            try:
                vis_combo = self.vis_data_source_combo
                vis_combo.blockSignals(True)
                vis_combo.clear()
                # (不再添加 "当前主项目"，Page 1 会自己扫描)
                vis_combo.blockSignals(False)
            except AttributeError:
                print("[DEBUG ResetProject] vis_data_source_combo not found during reset.")
            except Exception as e:
                print(f"[DEBUG ResetProject] Error resetting vis_data_source_combo: {e}")
            try:
                if self.vis_project_manager:
                    self.vis_project_manager.load_project_data(None)
                    self.add_to_log("M7/Vis: 已卸载可视化管理器。")
            except Exception as e_vis_clear:
                self.add_to_log(f"警告: 卸载 vis_project_manager 失败: {e_vis_clear}")

            # 保持可视化按钮启用 (因为主项目仍可查看)
            if self.vis_action: self.vis_action.setEnabled(True)

            # --- VVVV 新增: 重置 M2 VVVV ---
            if hasattr(self, 'tab_label_eng'):
                self.m2_peak_config.clear()
                if hasattr(self, '_update_m2_table_from_config'):
                    self._update_m2_table_from_config()  # 清空 UI 表格
                if hasattr(self, 'm2_class_selector'):
                    self.m2_class_selector.clear()
                if hasattr(self, 'm2_peak_pos_display'):
                    self.m2_peak_pos_display.setText("[ 未选择 ]")

                m2_tab_index = self.processing_tabs.indexOf(self.tab_label_eng)
                self.processing_tabs.setTabEnabled(m2_tab_index, False)  # 禁用 M2 选项卡
            # --- ^^^^ 新增结束 ^^^^ ---

            # 6. 重置 M4 UI
            self.on_clear_anchors_all()
            self.on_clear_regions()
            self._disable_all_plot_interactions()

            # 7. 刷新所有下拉框 (版本、M6、对比等)
            self._update_version_selector()
            self._clear_all_cache_files()

            # 8. 重新加载主项目管理器 (因为 labels_dataframe 可能已变)
            self.project_manager.load_project_data(self.current_project)

            if self.workflow_action: self.workflow_action.setChecked(True)
            if self.main_stack: self.main_stack.setCurrentIndex(0)

            # 9. 重绘当前样本 (现在它会显示 'original' 数据)
            idx = self.project_manager.get_current_selected_index()
            if idx != -1:
                # 确保在重绘前项目管理器已更新
                QApplication.processEvents()
                self.on_sample_selected(idx, force_replot=True)
            else:
                self.plot_widget.clear_plot()

            self.add_to_log("--- 项目已成功重置 (M2, Vis, 缓存已清空) ---")

        except Exception as e:
            self.add_to_log(f"错误: 重置项目失败: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "重置失败", f"重置项目时发生错误:\n{e}")

    def _clear_all_cache_files(self):
        """
        (新增) 辅助函数：安全地删除所有项目的缓存文件。
        在 on_reset_project 和 closeEvent 中调用。
        """
        self.add_to_log("--- 正在清除所有缓存文件 ---")

        # 从我们定义的常量中获取目录
        dirs_to_delete = [self.PROJECT_CACHE_ROOT, self.AUGMENTED_OUTPUT_ROOT]
        deleted_count = 0
        error_count = 0

        for cache_dir in dirs_to_delete:
            abs_path = os.path.abspath(cache_dir)
            if os.path.isdir(abs_path):
                self.add_to_log(f"  > 正在删除: {abs_path}")
                try:
                    # 使用 shutil.rmtree 安全地删除整个目录树
                    shutil.rmtree(abs_path)
                    self.add_to_log(f"  > 成功删除: {abs_path}")
                    deleted_count += 1
                except Exception as e:
                    self.add_to_log(f"  ! 错误: 删除 {abs_path} 失败: {e}")
                    traceback.print_exc()
                    error_count += 1
            else:
                self.add_to_log(f"  > 缓存目录 {abs_path} 不存在，跳过。")

        if error_count > 0:
            self.add_to_log("--- 缓存清除时遇到错误 ---")
            QMessageBox.warning(self, "缓存清除失败",
                                f"尝试删除缓存文件时遇到错误。详情请查看日志。")
        elif deleted_count > 0:
            self.add_to_log("--- 缓存已成功清除 ---")
        else:
            self.add_to_log("--- 未找到缓存，无需清除 ---")

    def closeEvent(self, event):  # Simplified
            self.on_stop_fitting()  # 尝试停止所有线程
            self.thread_pool.waitForDone(2000)  # 等待 2 秒
            self._clear_all_cache_files()
            self.add_to_log("退出。");
            event.accept()

        # --- M2 Label Engineering Slots ---

    def _initialize_m2_tab(self):
            """
            (新增) 检查项目是否为分类任务，并填充 M2 UI。
            在 load_project 时调用。
            """
            m2_tab = getattr(self, 'tab_label_eng', None)
            if not self.current_project or not m2_tab:
                return

            # 清空旧状态
            self.m2_peak_config.clear()
            self.m2_class_selector.blockSignals(True)
            self.m2_class_selector.clear()
            self.m2_config_table.setRowCount(0)
            self.m2_peak_pos_display.setText("[ 未选择 ]")

            # 检查是否为分类任务
            cls_col, _ = self.current_project.get_primary_target_col(task_filter='classification')
            if not cls_col:
                self.add_to_log("M2: 当前项目不是分类任务，M2 模块已禁用。")
                self.processing_tabs.setTabEnabled(self.processing_tabs.indexOf(m2_tab), False)
                return

            self.add_to_log(f"M2: 检测到分类标签: '{cls_col}'")
            self.processing_tabs.setTabEnabled(self.processing_tabs.indexOf(m2_tab), True)

            try:
                categories = self.current_project.labels_dataframe[cls_col].unique()
                categories = sorted([str(c) for c in categories])

                self.m2_class_selector.addItems(categories)

                # 初始化配置字典
                for cat in categories:
                    # 替换不安全的文件名/列名字符
                    safe_cat_name = re.sub(r'[^\w\d_]', '_', cat.lower())
                    self.m2_peak_config[cat] = {
                        "peak_pos": None,
                        "window": 10.0,
                        "col_name": f"pseudo_{safe_cat_name}",
                        "status": "Pending"
                    }

                self._update_m2_table_from_config()
                self.m2_class_selector.blockSignals(False)

            except Exception as e:
                self.add_to_log(f"错误: 初始化 M2 选项卡失败: {e}")
                traceback.print_exc()

    def _update_m2_table_from_config(self):
            """
            (新增) 使用 self.m2_peak_config 中的数据刷新 M2 UI 表格。
            """
            table = getattr(self, 'm2_config_table', None)
            if not table: return

            table.setRowCount(len(self.m2_peak_config))
            for i, (class_name, config) in enumerate(self.m2_peak_config.items()):
                peak_pos_str = f"{config['peak_pos']:.3f}" if config['peak_pos'] is not None else "(未设置)"

                table.setItem(i, 0, QTableWidgetItem(str(class_name)))
                table.setItem(i, 1, QTableWidgetItem(peak_pos_str))
                table.setItem(i, 2, QTableWidgetItem(f"{config['window']:.1f}"))
                table.setItem(i, 3, QTableWidgetItem(config['col_name']))

                status_item = QTableWidgetItem(config['status'])
                if config['status'] == 'Generated':
                    status_item.setBackground(QColor("lightgreen"))
                elif config['status'] == 'Configured':
                    status_item.setBackground(QColor("lightblue"))
                table.setItem(i, 4, status_item)

    @Slot(str)
    def on_m2_class_selector_changed(self, class_name: str):
            """
            (新增) 当用户在 M2 下拉框中选择一个类别时，更新交互控件的值。
            """
            if not class_name or class_name not in self.m2_peak_config:
                # 如果选择无效或配置不存在，重置UI控件
                if hasattr(self, 'm2_peak_pos_display'):
                    self.m2_peak_pos_display.setText("[ 未选择 ]")
                if hasattr(self, 'm2_peak_window_input'):
                    self.m2_peak_window_input.setValue(10.0)  # 恢复默认值
                # 禁用 M2 选峰按钮，防止在无效类别下选峰
                if hasattr(self, 'm2_pick_peak_btn'):
                    self.m2_pick_peak_btn.setEnabled(False)
                return

            try:
                # 启用 M2 选峰按钮
                if hasattr(self, 'm2_pick_peak_btn'):
                    self.m2_pick_peak_btn.setEnabled(True)

                # 从配置字典中获取该类别的信息
                config = self.m2_peak_config[class_name]
                peak_pos = config.get('peak_pos')
                window = config.get('window', 10.0)  # 如果没有，使用默认值

                # 更新 M2 UI 上的显示
                if hasattr(self, 'm2_peak_pos_display'):
                    if peak_pos is not None:
                        self.m2_peak_pos_display.setText(f"{peak_pos:.3f}")
                    else:
                        self.m2_peak_pos_display.setText("[ 未选择 ]")

                if hasattr(self, 'm2_peak_window_input'):
                    self.m2_peak_window_input.setValue(window)

            except KeyError:
                self.add_to_log(f"警告: M2 配置字典中找不到类别 '{class_name}'。")
                # 重置UI控件
                if hasattr(self, 'm2_peak_pos_display'): self.m2_peak_pos_display.setText("[ 未选择 ]")
                if hasattr(self, 'm2_peak_window_input'): self.m2_peak_window_input.setValue(10.0)
                if hasattr(self, 'm2_pick_peak_btn'): self.m2_pick_peak_btn.setEnabled(False)
            except Exception as e:
                self.add_to_log(f"错误: M2 切换类别失败: {e}")
                traceback.print_exc()

    @Slot(bool)
    def on_m2_pick_peak_toggled(self, checked):
            """
            (新增) M2 的 "选峰" 按钮，与 M4 互斥。
            """
            btn_self = getattr(self, 'm2_pick_peak_btn', None)

            # M4 按钮
            btn_m4_pick = getattr(self, 'm4_pick_mode_btn', None)
            btn_m4_region = getattr(self, 'm4_region_mode_btn', None)
            btn_m4_thresh = getattr(self, 'm4_threshold_mode_btn', None)

            if checked:
                self.current_pick_mode_owner = 'M2'
                # 取消 M4 的所有交互按钮
                if btn_m4_pick and btn_m4_pick.isChecked():
                    btn_m4_pick.blockSignals(True);
                    btn_m4_pick.setChecked(False);
                    btn_m4_pick.blockSignals(False)
                if btn_m4_region and btn_m4_region.isChecked(): btn_m4_region.setChecked(False)
                if btn_m4_thresh and btn_m4_thresh.isChecked(): btn_m4_thresh.setChecked(False)

                if btn_self and not btn_self.isChecked(): btn_self.setChecked(True)
                self.request_plot_interaction_mode.emit(MODE_PICK_ANCHOR);
                self.add_to_log("交互: [M2 选峰] 启用")

            elif (not btn_m4_pick or not btn_m4_pick.isChecked()) and \
                    (not btn_m4_region or not btn_m4_region.isChecked()) and \
                    (not btn_m4_thresh or not btn_m4_thresh.isChecked()):

                self._disable_all_plot_interactions()

            if not checked and self.current_pick_mode_owner == 'M2':
                self.current_pick_mode_owner = None

    @Slot(float)
    def on_m2_peak_received(self, x_pos):
            """
            (新增) 当 M2 模式下在图上选峰时，此槽被 on_anchor_added_from_plot 调用。
            """
            # 1. 更新 UI
            self.m2_peak_pos_display.setText(f"{x_pos:.3f}")

            # 2. 禁用选峰按钮 (使其成为一次性操作)
            btn_m2_pick = getattr(self, 'm2_pick_peak_btn', None)
            if btn_m2_pick:
                btn_m2_pick.blockSignals(True);
                btn_m2_pick.setChecked(False);
                btn_m2_pick.blockSignals(False)

            # 3. 禁用绘图交互
            self.request_plot_interaction_mode.emit(MODE_DISABLED)
            self.current_pick_mode_owner = None
            self.add_to_log(f"M2: 捕获到峰位: {x_pos:.3f}")

    @Slot()
    def on_m2_apply_peak_config(self):
            """
            (新增) 当点击 "设置/更新此类别配置" 按钮时。
            将 UI 上的值保存到 self.m2_peak_config 字典中。
            """
            try:
                current_class = self.m2_class_selector.currentText()
                if not current_class:
                    raise ValueError("未选择任何类别。")

                peak_pos_str = self.m2_peak_pos_display.text()
                if "[ 未选择 ]" in peak_pos_str or not peak_pos_str:
                    raise ValueError("未选择特征峰。请先在图上点击选峰。")

                peak_pos = float(peak_pos_str)
                window = self.m2_peak_window_input.value()

                # 更新配置字典
                self.m2_peak_config[current_class]["peak_pos"] = peak_pos
                self.m2_peak_config[current_class]["window"] = window
                self.m2_peak_config[current_class]["status"] = "Configured"

                # 刷新表格
                self._update_m2_table_from_config()
                self.add_to_log(f"M2: 类别 '{current_class}' 配置已更新。")

            except Exception as e:
                self.add_to_log(f"错误: M2 配置失败: {e}")
                QMessageBox.warning(self, "配置失败", f"M2 配置失败: {e}")

    @Slot()
    def on_m2_run_generation(self):
            """
            (新增) 当点击 "生成/更新所有伪标签" 按钮时。
            遍历所有光谱，计算峰高，并将新列添加到 labels_dataframe。
            """
            if not self.current_project:
                self.add_to_log("错误: M2 生成失败，无项目。");
                return

            # 检查是否有任何类别被配置
            if not any(cfg.get('status') == 'Configured' for cfg in self.m2_peak_config.values()):
                if not any(cfg.get('status') == 'Generated' for cfg in self.m2_peak_config.values()):
                    self.add_to_log("M2: 没有任何类别被配置。请先设置特征峰。")
                    QMessageBox.warning(self, "未配置", "没有任何类别被配置。请先设置特征峰。")
                    return

            self.add_to_log("--- M2: 开始生成伪标签列 ---")
            try:
                # 0. 获取主分类列
                cls_col, _ = self.current_project.get_primary_target_col(task_filter='classification')
                if not cls_col:
                    raise RuntimeError("无法找到主分类列。")

                # 1. 获取数据
                df = self.current_project.labels_dataframe
                # (重要) 我们应该使用哪个版本的光谱来计算峰高？
                # 建议使用 'original'，因为它最干净，未受 M3 预处理影响。
                current_ver = self.current_project.active_spectra_version
                spectra = self.current_project.spectra_versions.get(current_ver)
                if spectra is None:
                    # 回退保护
                    spectra = self.current_project.spectra_versions.get('original')

                self.add_to_log(f"M2: 使用光谱版本 '{current_ver}' 进行伪标签计算。")
                x_axis = self.current_project.wavelengths
                if spectra is None:
                    raise RuntimeError("找不到 'original' 光谱版本。")

                scale_enabled = getattr(self, 'm2_scale_checkbox', None) and self.m2_scale_checkbox.isChecked()
                target_min = getattr(self, 'm2_scale_min_input', None) and self.m2_scale_min_input.value() or 0.0
                target_max = getattr(self, 'm2_scale_max_input', None) and self.m2_scale_max_input.value() or 1.0
                if scale_enabled and target_min >= target_max:
                    raise ValueError("缩放范围错误：最小值必须小于最大值。")

                # 2. 遍历所有*已配置*的类别
                new_cols_added = []
                for class_name, config in self.m2_peak_config.items():

                    if config.get('peak_pos') is None:
                        continue  # 跳过未配置的

                    col_name = config["col_name"]
                    peak_pos = config["peak_pos"]
                    window = config["window"]
                    self.add_to_log(f"M2: ...正在处理 '{class_name}' -> '{col_name}' (峰: {peak_pos:.2f})")

                    new_label_col = []
                    raw_intensities_for_scaling = []
                    mask = (x_axis >= peak_pos - window) & (x_axis <= peak_pos + window)
                    if not np.any(mask):
                        self.add_to_log(f"警告: 峰位 {peak_pos}±{window} 在波长范围之外，此列将全为0。")

                    # 3. 遍历所有样本
                    for i in range(len(df)):
                        # 检查此样本是否属于目标类别
                        if str(df.iloc[i][cls_col]) == str(class_name):
                            if np.any(mask):
                                spec = spectra[i]
                                peak_intensity = np.max(spec[mask])
                                new_label_col.append(peak_intensity)
                                if scale_enabled: raw_intensities_for_scaling.append(peak_intensity)
                            else:
                                new_label_col.append(0.0)
                        else:
                            # 不属于此类别，伪标签为 0
                            new_label_col.append(0.0)

                    # 3.5 (新增) 执行缩放
                    if scale_enabled and raw_intensities_for_scaling:
                        min_intensity = np.min(raw_intensities_for_scaling)
                        max_intensity = np.max(raw_intensities_for_scaling)
                        self.add_to_log(
                            f"M2: ...'{class_name}' 原始峰高范围: [{min_intensity:.2f}, {max_intensity:.2f}]")

                        # 缩放因子 (处理分母为0的情况)
                        scale_range = max_intensity - min_intensity
                        if scale_range < 1e-9:  # 如果所有峰高都一样
                            self.add_to_log("警告: 此类别所有样本峰高相同，将统一映射到目标范围中点。")
                            scaled_value = (target_min + target_max) / 2.0
                            # 遍历 new_label_col，替换非零值为 scaled_value
                            for i in range(len(new_label_col)):
                                if not np.isclose(new_label_col[i], 0.0):  # 只替换属于本类别的样本
                                    new_label_col[i] = scaled_value
                        else:
                            # 执行 Min-Max 缩放
                            # 遍历 new_label_col，替换非零值为缩放后的值
                            for i in range(len(new_label_col)):
                                if not np.isclose(new_label_col[i], 0.0):  # 只替换属于本类别的样本
                                    original_intensity = new_label_col[i]
                                    scaled_val = target_min + (original_intensity - min_intensity) * (
                                            target_max - target_min) / scale_range
                                    new_label_col[i] = scaled_val
                        self.add_to_log(f"M2: ...已缩放到范围 [{target_min:.4f}, {target_max:.4f}]")
                    elif scale_enabled:
                        self.add_to_log(f"警告: 类别 '{class_name}' 未找到任何样本，无法执行缩放。")
                    # --- ^^^^ 新增结束 ^^^^ ---

                    # 4. 将新列添加到 DataFrame 和 task_info
                    df[col_name] = new_label_col
                    self.current_project.task_info[col_name] = {'role': 'target', 'type': 'regression'}
                    config["status"] = "Generated"  # 更新 M2 状态
                    new_cols_added.append(col_name)

                # 5. 完成后，刷新所有 UI
                if new_cols_added:
                    self.add_to_log(f"M2: 成功添加/更新伪标签列: {', '.join(new_cols_added)}")
                    # 刷新 M2 自己的表格
                    self._update_m2_table_from_config()
                    # 刷新左侧的主项目管理器 (它现在会显示新列)
                    self.project_manager.load_project_data(self.current_project)
                    QMessageBox.information(self, "M2 完成",
                                            f"成功生成 {len(new_cols_added)} 个伪标签列。\n"
                                            "您现在可以在 M5 (数据增强) 模块中使用这些新列作为回归目标。")
                else:
                    self.add_to_log("M2: 未执行任何操作（没有已配置'Configured'的类别）。")

            except Exception as e:
                self.add_to_log(f"错误: M2 伪标签生成失败: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "M2 失败", f"M2 伪标签生成失败: {e}")

        # --- M5 Augmentation Slots ---
    def _get_m5_buttons(self):
            """(新增) 辅助函数：获取 M5 按钮"""
            return [getattr(self, 'm5_run_button', None)]

    @Slot()
    def _update_m5_target_widgets(self, mode_combo, step_spinbox, specific_widget):
            """
            (新增) 根据 M5 生成模式下拉框，启用/禁用步长和特定值输入框。
            """
            try:
                is_specific = mode_combo.currentIndex() == 1  # Specific 是第二个选项
                step_spinbox.setEnabled(not is_specific)
                specific_widget.setEnabled(is_specific)
            except Exception as e:
                print(f"Warning: Failed to update M5 target widgets state: {e}")

    @Slot()
    def on_run_augmentation(self):
        """
        (修改) 当用户点击 M5 选项卡上的 "开始生成" 按钮时触发。
        修复: MOGP 模式下智能筛选数值型浓度列，避免读取字符串列导致崩溃。
        """
        if not self.current_project:
            self.add_to_log("错误: 请先加载一个项目。")
            QMessageBox.warning(self, "无项目", "请先加载一个项目再运行数据增强。")
            return

        # 0. 禁用按钮，防止重复点击
        for btn in self._get_m5_buttons():
            if btn: btn.setEnabled(False)
        QApplication.processEvents()

        try:
            config = {}
            output_prefix = self.m5_output_prefix.text().strip()
            # (自动生成前缀的逻辑移到 Worker 内部，以包含时间戳)
            config["OUTPUT_PREFIX"] = output_prefix if output_prefix else None

            active_tab_index = self.m5_sub_tabs.currentIndex()
            worker = None
            mode = 'Unknown'

            # =================================================================
            # 策略 A: 线性光谱插值 (Linear Spectrum)
            # =================================================================
            if active_tab_index == 0:
                mode = 'LinearSpectrum'
                self.add_to_log(f"--- 准备启动 [线性光谱插值] 增广 ---")

                config["INPUT_VERSION"] = self.project_manager.version_selector_combo.currentText()
                if not config["INPUT_VERSION"]:
                    raise ValueError("未选择输入光谱版本。")

                config["GENERATION_MODE"] = "specific" if self.m5_spec_mode.currentIndex() == 1 else "interpolate"
                config["INTERPOLATION_STEP"] = self.m5_spec_step.value()

                try:
                    specific_str = self.m5_spec_specific.text()
                    config["SPECIFIC_CONCENTRATIONS_TO_GENERATE"] = [float(x.strip()) for x in
                                                                     specific_str.split(',')
                                                                     if x.strip()]
                    if config["GENERATION_MODE"] == "specific" and not config["SPECIFIC_CONCENTRATIONS_TO_GENERATE"]:
                        raise ValueError("选择了 'Specific' 模式，但未输入特定值。")
                except ValueError as e:
                    raise ValueError(f"特定值格式错误: {e}")

                config["INTERPOLATION_DIMENSION_INDEX"] = self.m5_spec_interp_dim.value()

                worker = AugmentationWorker(
                    mode=mode,
                    project=self.current_project,
                    config=config,
                    m4_fit_results_all=None,
                    augmented_output_root_path=self.AUGMENTED_OUTPUT_ROOT,
                    project_cache_root_path=self.PROJECT_CACHE_ROOT
                )

            # =================================================================
            # 策略 B: 峰参数插值 (Linear Param)
            # =================================================================
            elif active_tab_index == 1:
                mode = 'LinearParam'
                self.add_to_log(f"--- 准备启动 [峰参数插值] 增广 ---")

                m4_version = self.m5_param_source_combo.currentText()
                if not m4_version or m4_version not in self.m4_batch_fit_results:
                    raise ValueError(f"未选择有效的 M4 拟合结果来源版本 ('{m4_version}')")
                config["M4_RESULTS_VERSION"] = m4_version

                try:
                    current_cache_dir = self._get_project_cache_dir()
                    if not current_cache_dir:
                        raise ValueError("无法获取项目缓存目录")
                    residuals_abs_path = os.path.join(current_cache_dir, "fit_results", f"residuals_{m4_version}")
                    config["RESIDUALS_PATH_ABSOLUTE"] = residuals_abs_path
                    self.add_to_log(f"M5: 正在设置残差路径: {residuals_abs_path}")
                except Exception as e:
                    self.add_to_log(f"警告: 无法确定残差路径: {e}")
                    config["RESIDUALS_PATH_ABSOLUTE"] = None

                config["INTERPOLATION_DIMENSION_INDEX"] = self.m5_param_interp_dim.value()
                config["ADD_RESIDUALS"] = self.m5_param_add_residuals.isChecked()

                config["GENERATION_MODE"] = "specific" if self.m5_param_mode.currentIndex() == 1 else "interpolate"
                config["INTERPOLATION_STEP"] = self.m5_param_step.value()

                try:
                    specific_str = self.m5_param_specific.toPlainText()
                    gen_list = []
                    lines = [line for line in specific_str.split('\n') if
                             line.strip() and not line.strip().lower().startswith('conc')]
                    for line in lines:
                        gen_list.append([float(x.strip()) for x in line.split(',') if x.strip()])
                    config["SPECIFIC_CONCENTRATIONS_TO_GENERATE"] = gen_list
                    if config["GENERATION_MODE"] == "specific" and not config["SPECIFIC_CONCENTRATIONS_TO_GENERATE"]:
                        raise ValueError("选择了 'Specific' 模式，但未输入特定值。")
                except ValueError as e:
                    raise ValueError(f"特定值格式错误: {e}")

                worker = AugmentationWorker(
                    mode=mode,
                    project=self.current_project,
                    config=config,
                    m4_fit_results_all=self.m4_batch_fit_results,
                    augmented_output_root_path=self.AUGMENTED_OUTPUT_ROOT,
                    project_cache_root_path=self.PROJECT_CACHE_ROOT
                )

            # =================================================================
            # 策略 C: MOGP 参数化生成 (MOGP)
            # =================================================================
            elif active_tab_index == 2:
                mode = 'MOGP'
                self.add_to_log(f"--- 准备启动 [MOGP 参数化生成] 增广 ---")

                # 1. 获取并校验左侧勾选的样本索引
                selected_indices = self.project_manager.get_checked_items_indices()
                if not selected_indices:
                    QMessageBox.warning(self, "未选择训练样本",
                                        "请在左侧列表中勾选用于训练 MOGP 模型的样本。\n"
                                        "系统将仅使用勾选的高质量样本来学习参数分布。")
                    for btn in self._get_m5_buttons(): btn.setEnabled(True)
                    return

                self.add_to_log(f"MOGP: 已选择 {len(selected_indices)} 个样本用于模型训练。")
                config["TRAINING_INDICES"] = selected_indices

                m4_version = self.m5_mogp_source_combo.currentText()
                if not m4_version or m4_version not in self.m4_batch_fit_results:
                    raise ValueError(f"未选择有效的 M4 拟合结果来源版本 ('{m4_version}')")
                config["M4_RESULTS_VERSION"] = m4_version

                # 设置残差路径
                try:
                    current_cache_dir = self._get_project_cache_dir()
                    if not current_cache_dir: raise ValueError("无法获取项目缓存目录")
                    residuals_abs_path = os.path.join(current_cache_dir, "fit_results", f"residuals_{m4_version}")
                    config["RESIDUALS_PATH_ABSOLUTE"] = residuals_abs_path
                except Exception as e:
                    self.add_to_log(f"警告: 无法确定残差路径: {e}")
                    config["RESIDUALS_PATH_ABSOLUTE"] = None

                # 基础 MOGP 配置
                config["NUM_LATENT_PROCESSES"] = self.m5_mogp_latent_q.value()
                config["POSITION_POLY_DEGREE"] = self.m5_mogp_poly_deg.value()
                config["ADD_RESIDUALS"] = self.m5_mogp_add_residuals.isChecked()
                config["max_iters"] = 3000
                config["USE_RANSAC_FOR_AREA"] = self.m5_mogp_use_ransac.isChecked()
                config["RANSAC_MIN_SAMPLES"] = self.m5_mogp_ransac_min_samples.value()
                config["APPLY_AREA_CONSTRAINT"] = self.m5_mogp_apply_area_constraint.isChecked()

                # 解析峰分组范围
                groups_str = self.m5_mogp_peak_groups.toPlainText().replace('\n', ',')
                ranges = []
                if config["APPLY_AREA_CONSTRAINT"]:
                    try:
                        parts = [x.strip() for x in groups_str.split(',') if x.strip()]
                        for p in parts:
                            vals = re.findall(r"[\d\.]+", p)
                            if len(vals) == 2: ranges.append((float(vals[0]), float(vals[1])))
                        if not ranges:
                            self.add_to_log("警告: 启用了面积约束但未解析到有效的峰分组范围。")
                            config["APPLY_AREA_CONSTRAINT"] = False
                    except Exception as e:
                        config["APPLY_AREA_CONSTRAINT"] = False
                config["PEAK_GROUP_RANGES"] = ranges

                # 生成目标配置
                config["GENERATION_MODE"] = "specific" if self.m5_mogp_mode.currentIndex() == 1 else "interpolate"
                config["INTERPOLATION_STEP"] = self.m5_mogp_step.value()
                try:
                    specific_str = self.m5_mogp_specific.toPlainText()
                    gen_list = []
                    lines = [line for line in specific_str.split('\n') if
                             line.strip() and not line.strip().lower().startswith('conc')]
                    for line in lines: gen_list.append([float(x.strip()) for x in line.split(',') if x.strip()])
                    config["SPECIFIC_CONCENTRATIONS_TO_GENERATE"] = gen_list
                    if config["GENERATION_MODE"] == "specific" and not config["SPECIFIC_CONCENTRATIONS_TO_GENERATE"]:
                        raise ValueError("选择了 'Specific' 模式，但未输入特定值。")
                except ValueError as e:
                    raise ValueError(f"特定值格式错误: {e}")

                # === [关键修复] 智能推断浓度列 (排除非数值列) ===
                task_info = self.current_project.task_info
                labels_df = self.current_project.labels_dataframe

                # 1. 优先查找 M2 生成的伪标签 (Role=target, Type=regression)
                target_cols = [col for col, info in task_info.items()
                               if info.get('role') == 'target' and info.get('type') == 'regression']

                # 2. 如果没找到，查找所有数值列并排除 ID
                if not target_cols:
                    numeric_cols = labels_df.select_dtypes(include=[np.number]).columns.tolist()
                    id_col = self.current_project.get_id_col()
                    target_cols = [c for c in numeric_cols if c != id_col]

                if target_cols:
                    # 将列名列表传给 Core，Core 会据此提取数据
                    config["CONCENTRATION_COLUMNS_TO_USE_NAMES"] = target_cols
                    # 索引列表其实在 Core 中通过名字反查更安全，但保持兼容性：
                    # 注意：Core 中 _prepare_mogp... 实际上是循环 labels_df.loc[idx, col]
                    # 所以只要 NAMES 对了，INDICES 其实主要用于后续绘图轴的选择
                    config["CONCENTRATION_COLUMNS_TO_USE"] = list(range(len(target_cols)))
                    self.add_to_log(f"MOGP: 自动选择数值浓度列作为输入 X: {target_cols}")
                else:
                    raise ValueError(
                        "无法找到有效的数值浓度列 (回归目标)。\n请确保已运行 M2 生成伪标签，或标签文件中包含数值列。")
                # === [修复结束] ===

                worker = AugmentationWorker(
                    mode=mode,
                    project=self.current_project,
                    config=config,
                    m4_fit_results_all=self.m4_batch_fit_results,
                    augmented_output_root_path=self.AUGMENTED_OUTPUT_ROOT,
                    project_cache_root_path=self.PROJECT_CACHE_ROOT
                )

            else:
                raise ValueError("未知的 M5 选项卡索引。")

            if worker:
                self.add_to_log(f"启动 '{mode}' 增广任务...")
                worker.s.finished.connect(self.on_aug_complete)
                worker.s.error.connect(self.on_aug_error)
                self.thread_pool.start(worker)
            else:
                for btn in self._get_m5_buttons(): btn.setEnabled(True)

        except Exception as e:
            self.add_to_log(f"错误: 启动增广失败: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "启动失败", f"启动数据增强时出错:\n{e}")
            for btn in self._get_m5_buttons():
                if btn: btn.setEnabled(True)

    @Slot(str)
    def on_aug_error(self, error_message):
            """(新增) 处理增广失败"""
            self.add_to_log(f"--- 增广失败 ---")
            self.add_to_log(error_message)
            # 重新启用按钮
            for btn in self._get_m5_buttons():
                if btn: btn.setEnabled(True)
            QMessageBox.critical(self, "增广失败", error_message)

        # (在 main.py 的 MainWindow 类中)
        # --- VVVV 可视化页面重构 VVVV ---
    @Slot(dict)
    def on_aug_complete(self, results):
            """
            (修改) 当 AugmentationWorker 完成时触发。
            (移除) 不再强制切换到可视化页面。
            (保留) 将新项目添加到数据源下拉框中。
            """
            try:
                # 重新启用 M5 按钮 (保持不变)
                for btn in self._get_m5_buttons():
                    if btn: btn.setEnabled(True)

                new_project = results.get('new_project')
                save_paths = results.get('save_paths')
                aug_project_name = results.get('aug_project_name')

                if not new_project or not save_paths or not aug_project_name:
                    raise ValueError("Worker failed to return new project, save paths, or project name.")
                if not isinstance(new_project, SpectralProject):
                    raise TypeError(f"Worker returned invalid project type: {type(new_project)}")

                self.add_to_log("--- 数据增广成功 ---")
                abs_spectra_path = os.path.abspath(save_paths.get('spectra', 'N/A'))
                abs_labels_path = os.path.abspath(save_paths.get('labels', 'N/A'))
                self.add_to_log(f"✅ 新光谱已保存至: {abs_spectra_path}")
                self.add_to_log(f"✅ 新标签已保存至: {abs_labels_path}")


                # 3. 启用 可视化视图 按钮 (保持不变)
                vis_button = getattr(self, 'vis_action', None)
                if vis_button: vis_button.setEnabled(True)



                # 5. (新增) 仅记录日志
                self.add_to_log(f"M5: 增广项目 '{aug_project_name}' 已添加到可视化数据源。")
                self.add_to_log("您可以随时切换到 '可视化视图' 页面查看。")
                self.add_to_log("M7/Vis: 正在刷新数据源列表...")
                self._vis_refresh_data_sources()



            except Exception as e:
                self.on_aug_error(f"处理增广结果时出错: {e}")

    @Slot()
    def on_toggle_view_workflow(self):
            """(修改) 切换到工作流页面 (页面 0)。"""
            if hasattr(self, 'main_stack'):
                # (移除 self.add_to_log，日志将由 on_main_stack_page_changed 处理)
                self.main_stack.setCurrentIndex(0)

    @Slot()
    def on_toggle_view_visualization(self):
            """
            (修改) 切换到可视化页面 (页面 1)。
            (移除) 不再负责加载数据。
            """
            if hasattr(self, 'main_stack'):
                # (移除 self.add_to_log 和 QTimer 逻辑)
                self.main_stack.setCurrentIndex(1)

    @Slot(int)
    def on_main_stack_page_changed(self, index: int):
        """
        (修改) 当主 QStackedWidget 页面切换时触发。使用 try...except 访问下拉框。
        """
        try:
            if index == 0:
                self.add_to_log("视图: 切换到工作流")
            elif index == 1:
                self.add_to_log("视图: 切换到可视化")
                current_source_name = None  # 初始化
                self._vis_refresh_data_sources()

                # (自动加载下拉框中的第一项)
                QTimer.singleShot(50, self._vis_trigger_load_after_scan)

        except Exception as e:
            # 这个 except 处理 on_main_stack_page_changed 函数本身的错误
            error_msg = f"切换视图时出错: {e}"
            self.add_to_log(error_msg)
            print(error_msg)
            traceback.print_exc()

    def _vis_trigger_load_after_scan(self):
        """(新增) 辅助 QTimer 回调，确保下拉框填充后加载第一项"""
        try:
            current_source_name = self.vis_data_source_combo.currentText()
            if current_source_name:
                self.add_to_log(f"M7/Vis: 自动加载数据源: {current_source_name}")
                self.on_vis_source_changed(current_source_name)
        except Exception as e:
            self.add_to_log(f"M7/Vis: 自动加载失败: {e}")

    @Slot(str)
    def on_vis_source_changed(self, source_name: str | None):
        """
        (修改) 当 Page 1 的 '数据源' 下拉框变化时触发。
        - (修复) 增加了对 '无可用的缓存数据' (folder_path is None) 的检查。
        """
        print(f"[DEBUG Vis SourceChanged] Source changed to: '{source_name}'")

        # 检查 1: 处理 None 或空字符串
        if not source_name:
            try:
                self.vis_project_manager.load_project_data(None)
            except Exception as e:
                print(f"Error clearing vis_project_manager: {e}")
            return

        # (此检查现在是多余的，因为我们将在下面检查 folder_path，
        # 但保留它作为双重保险)
        if source_name == "无可用的缓存数据":
            try:
                self.vis_project_manager.load_project_data(None)
            except Exception as e:
                print(f"Error clearing vis_project_manager: {e}")
            return

        try:
            folder_path = self.vis_data_source_combo.currentData()  # 获取存储的路径

            # --- VVVV (!!! 关键修复 !!!) VVVV ---
            # 检查 2: 处理 currentData() 返回 None 的情况
            if not folder_path:
                # 这发生在 "无可用的缓存数据" 被选中时
                self.add_to_log("M7/Vis: 未选择有效的缓存数据源。")
                self.vis_project_manager.load_project_data(None)
                self.vis_m4_results_cache.clear()
                self._vis_fill_version_combo(None)  # 确保版本下拉框也被清空
                return  # <-- 安全退出
            # --- ^^^^ (修复结束) ^^^^ ---

            # 检查 3: 检查路径是否真的存在
            if not os.path.isdir(folder_path):
                raise ValueError(f"缓存路径无效或不存在: {folder_path}")

            # (!!!) 调用我们的新加载器 (!!!)
            project, m4_results = self._vis_load_project_from_folder(folder_path)

            # (!!!) 存储在 Page 1 的私有缓存中 (!!!)
            self.vis_m4_results_cache = m4_results if m4_results else {}

        except Exception as e_load:
            self.add_to_log(f"M7/Vis: 加载 {source_name} 失败: {e_load}")
            traceback.print_exc()
            project = None
            self.vis_m4_results_cache.clear()

        # 3. (保持不变) 加载左侧面板
        try:
            self.vis_project_manager.load_project_data(project)
            print("[DEBUG Vis SourceChanged] Loaded project into vis_project_manager.")
        except AttributeError:
            print("[DEBUG Vis SourceChanged] vis_project_manager not found.")

        # 4. (保持不变) 填充版本下拉框
        try:
            self._vis_fill_version_combo(project)
        except Exception as e:
            print(f"[DEBUG Vis SourceChanged] FAILED to fill version combo: {e}")

        # 5. (保持不变) 触发版本切换逻辑
        try:
            current_version = self.vis_project_manager.version_selector_combo.currentText()
            self.on_vis_version_changed(current_version)
        except AttributeError:
            pass

    @Slot()
    def on_vis_clear_button_clicked(self):
        """当点击 Page 1 上的 '清空画板' 按钮时触发。"""
        try:
            self.vis_plot_widget.clear_plot()
            self.add_to_log("M7/Vis: 画板已清空。")
        except AttributeError:
            print("[DEBUG Vis Clear] vis_plot_widget not found.")
        except Exception as e:
            print(f"[DEBUG Vis Clear] Error clearing plot: {e}")
            self.add_to_log(f"错误: 清空画板失败: {e}")

    def _vis_plot_components(self, source_project, sample_index, selected_versions, x_axis, target_plot: PlotWidget):
        """
        (修改) 当 Page 1 "显示构成峰" 被勾选时调用。
        (修复) 修复了查找 M4 结果的逻辑，现在从 Page 1 的私有缓存 (self.vis_m4_results_cache) 中查找。
        """
        source_description = ""
        source_name = ""
        try:
            # (不变) 尝试获取数据源名称用于日志
            source_name = self.vis_data_source_combo.currentText()
        except:
            pass

        params_df: pd.DataFrame | None = None
        y_raw: np.ndarray | None = None
        y_fit: np.ndarray | None = None
        peak_shape = 'voigt'  # 默认

        try:
            # --- 1. 确定我们要查找的 M4 输入版本 ---
            param_source_version = None
            target_output_version = selected_versions[0]  # e.g., "original_fitted_curves_1"

            # (不变) 在历史记录中搜索创建此版本的条目
            # (我们假设 _vis_load_project_from_folder 会重建历史记录)
            for entry in source_project.processing_history:
                if entry.get('output_version') == target_output_version:
                    param_source_version = entry.get('input_version')
                    break

            if param_source_version is None:
                # (不变) 回退逻辑
                param_source_version = target_output_version.replace("_fitted_curves", "")
                print(f"M7/Vis: 警告: 无法在历史记录中找到 {target_output_version}。")
                print(f"        回退猜测 input_version 为: {param_source_version}")

            # --- 2. (!!! 关键修改 !!!) 从 Page 1 的私有缓存中查找 M4 结果 ---
            # (不再使用 self.m4_batch_fit_results)
            m4_results_df = self.vis_m4_results_cache.get(param_source_version)  # <--- (新逻辑)

            if m4_results_df is not None and not m4_results_df.empty:
                # (新逻辑) M4 CSV 是合并的, 我们需要按 sample_index 查找
                sample_params_df = m4_results_df[m4_results_df['sample_index'] == sample_index]

                if not sample_params_df.empty:
                    # (新逻辑) 清理掉非参数列 (sample_index 和所有标签列)
                    # (我们从 source_project 中获取标签列名)
                    label_cols = source_project.labels_dataframe.columns
                    cols_to_drop = ['sample_index'] + [c for c in label_cols if c in sample_params_df.columns]

                    # (使用 .copy() 避免 SettingWithCopyWarning)
                    params_df_cleaned = sample_params_df.drop(columns=cols_to_drop).copy()

                    # (新逻辑) 重建索引
                    params_df_cleaned.index = [f"Peak {i + 1}" for i in range(len(params_df_cleaned))]
                    params_df = params_df_cleaned

                    source_description = f"M4 拟合 (基于: {param_source_version})"

                    # (不变) 获取 y_raw (来自输入版本) 和 y_fit (来自曲线版本)
                    # (这些数据在 _vis_load_project_from_folder 中已加载到 source_project.spectra_versions)
                    y_raw_all = source_project.spectra_versions.get(param_source_version)
                    if y_raw_all is not None: y_raw = y_raw_all[sample_index]
                    fit_version_name = target_output_version
                    y_fit_all = source_project.spectra_versions.get(fit_version_name)
                    if y_fit_all is not None: y_fit = y_fit_all[sample_index]

                    # (不变) 后备
                    if y_raw is None: y_raw = y_fit
                    if y_fit is None: y_fit = y_raw

                    # (!!! 关键修改 !!!) (不再使用 self.m4_batch_cfg_fit)
                    # (新逻辑) 从 DataFrame 列猜测峰形
                    if 'Eta' in params_df.columns:
                        peak_shape = 'voigt'  # 5-param
                    elif 'Gamma (cm)' in params_df.columns:
                        peak_shape = 'voigt'  # 4-param
                    else:
                        peak_shape = 'gaussian'

                else:
                    # (日志) 在 M4 缓存中未找到该特定样本
                    print(f"M7/Vis: 在 M4 缓存 '{param_source_version}.csv' 中未找到 样本 {sample_index}。")

            else:
                # (日志) 缓存中没有此 M4 版本的结果
                print(f"M7/Vis: 未在 Page 1 缓存中找到 M4 结果 '{param_source_version}'。")

            # --- 3. (不变) 检查是否为 M5 生成的参数 ---
            # (如果 M4 查找失败，params_df 仍然是 None)
            if params_df is None:
                # (假设 _vis_load_project_from_folder 正确填充了 M5 项目的此字段)
                if hasattr(source_project, 'generated_peak_params') and source_project.generated_peak_params:
                    params_df = source_project.generated_peak_params.get(sample_index)
                    if params_df is not None:
                        source_description = f"M5 生成 ({source_name.replace('增广项目: ', '')})"
                        # (不变) M5 逻辑
                        y_total_all = source_project.spectra_versions.get(selected_versions[0])
                        if y_total_all is not None:
                            y_total = y_total_all[sample_index]
                            y_raw, y_fit = y_total, y_total
                        # (不变) M5 猜峰形
                        if 'Eta' in params_df.columns:
                            peak_shape = 'voigt'
                        elif 'Gamma (cm)' in params_df.columns:
                            peak_shape = 'voigt'
                        else:
                            peak_shape = 'gaussian'
                    else:
                        print(f"M7/Vis: 增广项目 '{source_name}' 样本 {sample_index} 没有存储的生成峰参数。")

        except Exception as e_find_params:
            self.add_to_log(f"M7/Vis: 查找峰参数时出错: {e_find_params}")
            traceback.print_exc()

        # --- 4. 绘图 (逻辑不变) ---
        title = f"样本 {sample_index} - 成分分析 ({source_description})"

        if params_df is not None and not params_df.empty and y_raw is not None:
            # (不变) 绘图逻辑
            print(f"[DEBUG Vis] Plotting components (Fill Style) for Smp {sample_index}")
            try:
                if y_fit is None:
                    # (不变) 如果 y_fit 丢失，则重建
                    print("[DEBUG Vis] y_fit is None, re-calculating from params_df...")
                    if peak_shape == 'voigt':
                        if 'Eta' in params_df.columns:  # 5-param
                            from core.peak_fitting_dl_batch import multi_pseudo_voigt_np as ff
                        else:  # 4-param
                            from core.peak_fitting_models import multi_voigt as ff
                    else:  # gaussian
                        from core.peak_fitting_models import multi_gaussian as ff

                    popt_flat = params_df.to_numpy().flatten()
                    y_fit = ff(x_axis, *popt_flat)

                # (不变) 调用 plot_widget 的方法
                target_plot.plot_peak_fitting_results(
                    x_axis, y_raw, y_fit,
                    params_df=params_df,
                    peak_shape=peak_shape,
                    title=title
                )

            except ImportError as e_imp:
                self.add_to_log(f"M7/Vis: 绘制组件失败 (ImportError): {e_imp}")
                traceback.print_exc()
                target_plot.plot_spectrum(source_project, sample_index, compare_version_name=None)
            except Exception as e_plot_comp:
                self.add_to_log(f"M7/Vis: 绘制成分峰失败: {e_plot_comp}")
                traceback.print_exc()
                target_plot.plot_spectrum(source_project, sample_index, compare_version_name=None)
        else:
            # (不变) 后备逻辑
            self.add_to_log(f"M7/Vis: 无法绘制成分峰 (缺少参数或 y_raw)。仅绘制光谱 '{selected_versions[0]}'。")
            target_plot.plot_spectrum(source_project, sample_index, compare_version_name=None)


    def _vis_fill_version_combo(self, project: SpectralProject | None):
        """
        (修改) 辅助函数：填充 Page 1 左侧的版本下拉框
        (修复) 现在会根据 project 是否存在来设置 setEnabled 状态
        """
        try:
            combo = self.vis_project_manager.version_selector_combo
            combo.blockSignals(True)
            combo.clear()

            if project:
                versions = project.get_version_names()
                combo.addItems(versions)
                print(f"[DEBUG Vis FillCombo] 填充版本: {versions}")
                # --- VVVV 关键修复 VVVV ---
                # 只有当有项目和版本时才启用下拉框
                combo.setEnabled(len(versions) > 0)
                # --- ^^^^ 修复结束 ^^^^ ---
            else:
                # --- VVVV 关键修复 VVVV ---
                # 如果没有项目，则禁用下拉框
                combo.setEnabled(False)
                # --- ^^^^ 修复结束 ^^^^ ---

            combo.blockSignals(False)
        except AttributeError:
            print("[DEBUG Vis FillCombo] FAILED: vis_project_manager.version_selector_combo not found.")
        except Exception as e:
            print(f"[DEBUG Vis FillCombo] FAILED: {e}")

    @Slot(str)
    def on_vis_version_changed(self, version_name: str):
            """(新) 当 Page 1 的 '数据版本' 下拉框变化时触发。
            - 负责更新样本列表的可用性。
            - 负责触发预览区重绘。
            """
            print(f"[DEBUG Vis VersionChanged] Version changed to: '{version_name}'")
            try:
                project = self.vis_project_manager.project
                if not project or not version_name:
                    self.vis_project_manager.update_sample_availability(None)  # 全部可用
                    return

                # 1. 从项目历史中获取该版本有效的索引
                valid_indices_set = project.get_processed_indices_for_version(version_name)

                # 2. 更新左侧列表的 UI (启用/禁用)
                self.vis_project_manager.update_sample_availability(valid_indices_set)

            except AttributeError:
                print("[DEBUG Vis VersionChanged] FAILED: vis_project_manager or methods not found.")
            except Exception as e:
                print(f"[DEBUG Vis VersionChanged] FAILED to update sample availability: {e}")
                traceback.print_exc()

            # 3. 触发预览区刷新
            self.on_vis_preview_triggered()

    @Slot()
    def on_vis_preview_triggered(self):
            """
            (新) 当 Page 1 上的样本被单击，或版本/复选框被更改时触发。
            - 负责重绘 '预览' 画板。
            """
            print("[DEBUG Vis Preview] Triggered.")

            try:
                project = self.vis_project_manager.project
                sample_index = self.vis_project_manager.get_current_selected_index()
                version_name = self.vis_project_manager.version_selector_combo.currentText()
                show_components = self.vis_show_components_cb.isChecked()

                if not project or sample_index == -1 or not version_name:
                    self.vis_preview_plot.clear_plot()
                    print("[DEBUG Vis Preview] No project, sample, or version selected. Preview cleared.")
                    return

                # 检查样本在当前版本是否可用
                valid_set = project.get_processed_indices_for_version(version_name)
                if valid_set is not None and sample_index not in valid_set:
                    self.vis_preview_plot.clear_plot()
                    print(
                        f"[DEBUG Vis Preview] Sample {sample_index} not valid for version {version_name}. Preview cleared.")
                    return

                print(f"[DEBUG Vis Preview] Plotting Smp {sample_index} ({version_name}), Components={show_components}")

                if show_components:
                    # 调用修改后的辅助函数，指定目标画板
                    self._vis_plot_components(
                        project,
                        sample_index,
                        [version_name],  # _vis_plot_components 期望一个列表
                        project.wavelengths,
                        target_plot=self.vis_preview_plot  # <-- 指定目标
                    )
                else:
                    # 调用预览画板自己的 plot_spectrum 方法
                    self.vis_preview_plot.plot_spectrum(
                        project,
                        sample_index,
                        compare_version_name=None  # 预览区不对比
                    )

            except AttributeError as e_ui:
                print(f"!!! [DEBUG Vis Preview] UI component not ready! Error: {e_ui}")
            except Exception as e:
                self.add_to_log(f"M7/Vis: 预览绘图失败: {e}")
                traceback.print_exc()
                try:
                    self.vis_preview_plot.clear_plot()
                except:
                    pass

    @Slot()
    def on_vis_plot_selected(self):
            """
            (新) 当点击 '绘制选中项' 按钮时触发。
            - 将数据 '添加' 到主画板。
            - (不) 清空主画板。
            """
            print("[DEBUG Vis PlotSelected] 'Draw' button clicked.")
            try:
                project = self.vis_project_manager.project
                version_name = self.vis_project_manager.version_selector_combo.currentText()
                sample_indices = self.vis_project_manager.get_checked_items_indices()  # 获取所有勾选项

                if not project:
                    self.add_to_log("M7/Vis 错误: '数据源' 未加载。")
                    return
                if not version_name:
                    self.add_to_log("M7/Vis 错误: '数据版本' 未选择。")
                    return
                if not sample_indices:
                    self.add_to_log("M7/Vis: 请在左侧列表中勾选至少一个样本。")
                    return

                print(
                    f"[DEBUG Vis PlotSelected] Plotting {len(sample_indices)} samples from version '{version_name}'...")

                x_axis = project.wavelengths
                y_data_all = project.spectra_versions.get(version_name)

                if y_data_all is None:
                    self.add_to_log(f"M7/Vis 错误: 找不到版本 '{version_name}' 的光谱数据。")
                    return

                plot_count = 0
                for idx in sample_indices:
                    try:
                        y_sample = y_data_all[idx]

                        # 生成一个唯一的、信息丰富的标签
                        source_name = self.vis_data_source_combo.currentText().replace("当前主项目", "主项目").replace(
                            "增广项目: ", "增广-")
                        label = f"Smp {idx} ({version_name}) - [{source_name}]"

                        # (关键) 在 'vis_main_plot' 上绘图，不清除
                        self.vis_main_plot.ax_main.plot(x_axis, y_sample, label=label, alpha=0.7)
                        plot_count += 1
                    except IndexError:
                        self.add_to_log(f"M7/Vis 警告: 样本 {idx} 在版本 '{version_name}' 中索引越界。")
                    except Exception as e_plot:
                        self.add_to_log(f"M7/Vis 警告: 绘制样本 {idx} 失败: {e_plot}")

                if plot_count > 0:
                    # (关键) 更新图例和画布
                    self.vis_main_plot.ax_main.legend(fontsize='small')
                    self.vis_main_plot.canvas.draw()
                    self.add_to_log(f"M7/Vis: 已将 {plot_count} 个光谱添加到主画板。")

            except AttributeError as e_ui:
                print(f"!!! [DEBUG Vis PlotSelected] UI component not ready! Error: {e_ui}")
            except Exception as e:
                self.add_to_log(f"M7/Vis: 绘制选中项失败: {e}")
                traceback.print_exc()

    @Slot()
    def on_vis_clear_plot(self):
            """
            (新) 当点击 '清空画板' 按钮时触发。
            - 只清空 '主' 画板。
            """
            print("[DEBUG Vis ClearPlot] 'Clear' button clicked.")
            try:
                self.vis_main_plot.clear_plot()
                self.vis_main_plot.ax_main.set_title("主绘图区 (可叠加)")  # 恢复标题
                self.vis_main_plot.canvas.draw()
                self.add_to_log("M7/Vis: 主画板已清空。")
            except AttributeError:
                print("[DEBUG Vis ClearPlot] FAILED: vis_main_plot not found.")
            except Exception as e:
                self.add_to_log(f"M7/Vis: 清空主画板失败: {e}")
                traceback.print_exc()

        # --- ^^^^ 新增结束 ^^^^ ---
if __name__ == "__main__":
        # (保持不变)
        app = QApplication(sys.argv);
        try:
            # 1. 构建到图标文件的绝对路径
            # (os 模块已在文件顶部导入)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_dir, 'app_icon.png')

            # 2. 检查文件是否存在
            if os.path.exists(icon_path):
                # (QIcon 类已在文件顶部导入)
                app_icon = QIcon(icon_path)

                # 3. 设置应用程序图标（推荐，用于任务栏等）
                app.setWindowIcon(app_icon)

                # 4. 创建窗口
                window = MainWindow();

                # 5. 设置窗口图标（推荐，用于窗口左上角）
                window.setWindowIcon(app_icon)

            else:
                print(f"警告: 找不到图标文件: {icon_path}")
                window = MainWindow();  # 即使找不到图标也继续创建窗口

        except Exception as e:
            print(f"加载图标时出错: {e}")
            window = MainWindow();  # 即使出错也继续
        window = MainWindow();
        window.show();
        sys.exit(app.exec())