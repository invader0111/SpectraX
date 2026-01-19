# -*- coding: utf-8 -*-
# 文件名: core/data_model.py
# 描述: 定义核心数据结构 (支持数据版本管理 - 已修正字段顺序)

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

@dataclass
class SpectralProject:
    """
    统一内部数据模型，支持多版本光谱数据和处理历史。
    (修正: 无默认值的字段在前，有默认值的字段在后)
    """
    # --- Fields WITHOUT defaults MUST come first ---
    wavelengths: np.ndarray
    labels_dataframe: pd.DataFrame
    task_info: Dict[str, Dict[str, str]] # Role & Type
    data_file_path: str
    label_file_path: str

    # --- Fields WITH defaults MUST come after ---
    spectra_versions: Dict[str, np.ndarray] = field(default_factory=dict)
    active_spectra_version: str = 'original'
    processing_history: List[Dict[str, Any]] = field(default_factory=list)

    generated_peak_params: Dict[int, pd.DataFrame] = field(default_factory=dict)

    # 在 data_model.py 的 SpectralProject 类中添加:

    def get_processed_indices_for_version(self, version_name: str) -> set | None:
        """
        查找特定版本是基于哪些样本索引处理得到的。
        返回一个包含索引的 set，如果找不到或为 'original'，则返回 None (代表全部可用)。
        """
        if version_name == 'original':
            return None  # 'original' 版本所有样本都可用

        for entry in self.processing_history:
            if entry.get('output_version') == version_name:
                indices = entry.get('indices_processed')
                if indices is not None and isinstance(indices, list):
                    return set(indices)  # 返回一个集合，便于快速查找

        # 如果在历史记录中找不到 (例如，它是 M4/M5 生成的版本)
        # 我们可以假设它适用于所有样本 (因为 M4/M5 通常是批量处理)
        return None

    # --- Methods (Unchanged) ---
    def get_active_spectra(self) -> np.ndarray | None:
        """获取当前激活版本的光谱数据 (整个数据集)"""
        return self.spectra_versions.get(self.active_spectra_version)

    def get_active_spectrum_by_index(self, sample_index: int) -> np.ndarray | None:
        """获取当前激活版本的指定索引的光谱数据 (单个样本)"""
        active_data = self.get_active_spectra()
        if active_data is not None and 0 <= sample_index < active_data.shape[0]:
            return active_data[sample_index]
        return None

    def get_version_names(self) -> List[str]:
        """获取所有可用的光谱版本名称列表 (按添加顺序)"""
        print(f"[DEBUG] get_version_names: spectra_versions keys = {list(self.spectra_versions.keys())}")
        print(f"[DEBUG] get_version_names: processing_history length = {len(self.processing_history)}")
        ordered_names = ['original']
        seen_outputs = {'original'}
        for entry in self.processing_history:
            out_ver = entry.get('output_version')
            if out_ver and out_ver not in seen_outputs:
                ordered_names.append(out_ver)
                seen_outputs.add(out_ver)
        # Add any potentially orphaned versions
        for name in self.spectra_versions.keys():
            if name not in seen_outputs:
                ordered_names.append(name)
        return ordered_names

    def add_spectra_version(self, name: str, data: np.ndarray, history_entry: Dict[str, Any]):
        """
        添加一个新的光谱版本和处理历史记录。
        Args/Raises: (Unchanged)
        """
        if name in self.spectra_versions:
            print(f"警告: 版本 '{name}' 已存在，将被覆盖。")
            # Or: raise ValueError(f"Version name '{name}' already exists.")

        if 'original' in self.spectra_versions:
             original_samples = self.spectra_versions['original'].shape[0]
             if data.shape[0] != original_samples:
                  raise ValueError(f"新版本 '{name}' 样本数 ({data.shape[0]}) 与原始 ({original_samples}) 不匹配。")
        elif len(self.labels_dataframe) != data.shape[0]:
             raise ValueError(f"新版本 '{name}' 样本数 ({data.shape[0]}) 与标签数 ({len(self.labels_dataframe)}) 不匹配。")

        if data.shape[1] != len(self.wavelengths):
             raise ValueError(f"新版本 '{name}' 特征数 ({data.shape[1]}) 与波长数 ({len(self.wavelengths)}) 不匹配。")

        if 'output_version' in history_entry and history_entry['output_version'] != name:
             print(f"警告: 历史记录 output_version ('{history_entry['output_version']}') 与版本名 ('{name}') 不符。使用 '{name}'。")
             history_entry['output_version'] = name

        self.spectra_versions[name] = data
        self.processing_history.append(history_entry)
        self.active_spectra_version = name
        print(f"光谱版本 '{name}' 已添加并激活。")

    def get_latest_version_name(self) -> str:
        """获取最新的（最后添加的）版本名称"""
        if not self.processing_history: return 'original'
        latest_entry = self.processing_history[-1]
        return latest_entry.get('output_version', 'original')

    # --- Other helper methods (Unchanged) ---
    def get_task_summary(self):
        types = [v['type'] for v in self.task_info.values() if v.get('role') == 'target']
        if 'classification' in types: return 'classification'
        elif 'regression' in types: return 'regression'
        else: return 'unknown'
    def get_primary_target_col(self, task_filter=None):
        for col, info in self.task_info.items():
            if info.get('role') == 'target':
                if task_filter is None or info.get('type') == task_filter: return col, info.get('type')
        return None, None
    def get_id_col(self):
        for col, info in self.task_info.items():
            if info.get('role') == 'id': return col
        return None
    def get_category_info(self, sample_index):
        if self.get_task_summary() != 'classification': return None, None
        cls_col, _ = self.get_primary_target_col(task_filter='classification')
        if not cls_col or not (0 <= sample_index < len(self.labels_dataframe)): return None, None
        try:
            cat_name = self.labels_dataframe.loc[sample_index, cls_col]
            cat_indices = self.labels_dataframe[self.labels_dataframe[cls_col] == cat_name].index.tolist()
            return cat_name, cat_indices
        except KeyError:
            print(f"警告: 找不到列 '{cls_col}' 或索引 {sample_index}"); return None, None
        except Exception as e:
            print(f"查找类别信息出错: {e}"); return None, None

    def reset_to_original(self) -> bool:
        """
        将项目重置回其 'original' 状态，清除所有其他版本和历史。
        返回 True 表示成功, False 表示失败。
        """
        if 'original' not in self.spectra_versions:
            print(f"警告: 'original' 版本未在 spectra_versions 字典中找到。无法重置。")
            return False

        try:
            # 1. 保存原始数据
            original_data = self.spectra_versions['original']

            # 2. 清空版本字典并恢复 'original'
            self.spectra_versions.clear()
            self.spectra_versions['original'] = original_data

            # 3. 清空历史记录
            self.processing_history.clear()

            # 4. 激活 'original'
            self.active_spectra_version = 'original'

            print("数据模型已成功重置到 'original'。")
            return True

        except Exception as e:
            print(f"错误: 重置 SpectralProject 失败: {e}")
            import traceback
            traceback.print_exc()
            return False