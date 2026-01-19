# -*- coding: utf-8 -*-
# 文件名: gui/data_importer.py
# 描述: 实现灵活的数据导入向导 (QWizard - 单一语言)
# (修改: 使用 functools.partial 修复 lambda 引用问题)

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QRadioButton, QCheckBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QComboBox, QHeaderView
)
from PyQt6.QtCore import pyqtSignal
# --- VVVV 新增导入 VVVV ---
import functools
# --- ^^^^ 新增结束 ^^^^ ---

from core.data_model import SpectralProject

# --- FileSelectPage (无需修改) ---
class FileSelectPage(QWizardPage):
    """向导的第一页：选择文件并定义其基本结构。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("步骤 1: 选择文件"); self.setSubTitle("选择光谱数据和标签文件。")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("光谱数据文件 (data.csv):"))
        data_layout = QHBoxLayout(); self.data_path_edit = QLineEdit(); self.data_browse_btn = QPushButton("浏览..."); self.data_browse_btn.clicked.connect(self.browse_data); data_layout.addWidget(self.data_path_edit); data_layout.addWidget(self.data_browse_btn); layout.addLayout(data_layout)
        layout.addWidget(QLabel("数据文件结构:"))
        self.data_with_xaxis_rb = QRadioButton("包含 X 轴 (第一列波长/位移)"); self.data_no_xaxis_rb = QRadioButton("仅强度数据 (所有列都是样本)"); self.data_with_xaxis_rb.setChecked(True); layout.addWidget(self.data_with_xaxis_rb); layout.addWidget(self.data_no_xaxis_rb)
        self.registerField("data_path*", self.data_path_edit); self.registerField("data_has_xaxis", self.data_with_xaxis_rb)
        layout.addSpacing(20)
        layout.addWidget(QLabel("标签文件 (labels.csv):"))
        label_layout = QHBoxLayout(); self.label_path_edit = QLineEdit(); self.label_browse_btn = QPushButton("浏览..."); self.label_browse_btn.clicked.connect(self.browse_label); label_layout.addWidget(self.label_path_edit); label_layout.addWidget(self.label_browse_btn); layout.addLayout(label_layout)
        self.label_has_header_cb = QCheckBox("标签文件包含表头 (Header)"); self.label_has_header_cb.setChecked(True); layout.addWidget(self.label_has_header_cb)
        self.registerField("label_path*", self.label_path_edit); self.registerField("label_has_header", self.label_has_header_cb)
        self.setLayout(layout)
    def browse_data(self): path, _ = QFileDialog.getOpenFileName(self, "选择光谱数据文件", "", "CSV (*.csv);;All (*)"); path and self.data_path_edit.setText(path)
    def browse_label(self): path, _ = QFileDialog.getOpenFileName(self, "选择标签文件", "", "CSV (*.csv);;All (*)"); path and self.label_path_edit.setText(path)


# --- RoleAssignPage ---
class RoleAssignPage(QWizardPage):
    """向导的第二页：为标签列分配角色。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("步骤 2: 分配标签角色"); self.setSubTitle("为标签文件中的每一列定义其用途。")
        layout = QVBoxLayout(self); self.table = QTableWidget(); self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); layout.addWidget(self.table); self.setLayout(layout)
        self.role_combos = []; self.type_combos = []

    # --- VVVV 新增辅助方法 VVVV ---
    def _update_type_combo_enabled(self, role_combo, type_combo, target_role_str):
        """根据角色下拉框的当前文本，启用/禁用类型下拉框"""
        type_combo.setEnabled(role_combo.currentText() == target_role_str)
    # --- ^^^^ 新增结束 ^^^^ ---

    def initializePage(self):
        label_path = self.field("label_path"); has_header = self.field("label_has_header")
        try:
            # More robust CSV reading with error handling for bad lines
            df = pd.read_csv(label_path, nrows=5, header=0 if has_header else None, on_bad_lines='skip')
            if df.empty: raise ValueError("CSV 文件为空或无法读取前几行。")

            if not has_header: columns = [f"Column {i}" for i in range(len(df.columns))]
            else: columns = df.columns.astype(str) # Ensure column names are strings

            example_row = df.iloc[0].astype(str) if not df.empty else [""] * len(columns) # Ensure examples are strings

            self.table.clear(); self.table.setRowCount(len(columns)); self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["列名", "示例值", "角色", "任务类型"])
            self.role_combos.clear(); self.type_combos.clear()
            role_opts = ["忽略", "ID", "目标"]; type_opts = ["回归", "分类"]
            target_role_str = "目标"

            for i, col in enumerate(columns):
                # Add items to table
                self.table.setItem(i, 0, QTableWidgetItem(str(col)))
                self.table.setItem(i, 1, QTableWidgetItem(str(example_row[i])))

                # Role ComboBox
                rc = QComboBox(); rc.addItems(role_opts); self.table.setCellWidget(i, 2, rc); self.role_combos.append(rc)

                # Type ComboBox
                tc = QComboBox(); tc.addItems(type_opts); self.table.setCellWidget(i, 3, tc); self.type_combos.append(tc)

                # Infer initial type (best effort)
                try:
                    # Attempt to check if the column seems numeric based on first few rows
                    col_data = pd.read_csv(label_path, usecols=[i], header=0 if has_header else None, nrows=20, on_bad_lines='skip').iloc[:, 0]
                    is_numeric = pd.api.types.is_numeric_dtype(col_data) and col_data.nunique() > 2 # Avoid purely binary/constant numerics being regression
                    tc.setCurrentText("回归" if is_numeric else "分类")
                except Exception as infer_err:
                    print(f"Warning: Could not infer type for column {col}: {infer_err}. Defaulting to 分类.")
                    tc.setCurrentText("分类") # Fallback

                # Set initial enabled state for type combo
                tc.setEnabled(rc.currentText() == target_role_str)

                # --- VVVV 修改: 使用 functools.partial VVVV ---
                # Create a partial function that captures the current rc and tc
                handler = functools.partial(self._update_type_combo_enabled, rc, tc, target_role_str)
                # Connect the signal to this specific handler
                rc.currentIndexChanged.connect(handler)
                # --- ^^^^ 修改结束 ^^^^ ---

            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.wizard().role_assign_page = self

        except FileNotFoundError:
             error_msg = f"错误: 找不到标签文件 '{label_path}'"
             self.setSubTitle(error_msg)
             if hasattr(self.wizard(), 'log_message'): self.wizard().log_message.emit(error_msg)
        except pd.errors.EmptyDataError:
             error_msg = f"错误: 标签文件 '{label_path}' 为空。"
             self.setSubTitle(error_msg)
             if hasattr(self.wizard(), 'log_message'): self.wizard().log_message.emit(error_msg)
        except Exception as e:
            error_msg = f"错误: 处理标签文件时出错: {e}"
            self.setSubTitle(error_msg)
            if hasattr(self.wizard(), 'log_message'): self.wizard().log_message.emit(error_msg)
            import traceback
            traceback.print_exc() # Print full traceback for debugging

# --- DataImportWizard (修改 accept 方法以使用更新的 SpectralProject) ---
class DataImportWizard(QWizard):
    """主向导。"""
    projectCreated = pyqtSignal(SpectralProject)
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.addPage(FileSelectPage())
        self.addPage(RoleAssignPage())
        self.setWindowTitle("数据导入向导")
        self.role_assign_page = None

    def accept(self):
        """点击 "Finish" 时的最终处理"""
        page = self.role_assign_page
        if not page: self.log_message.emit("错误: 无法访问角色分配页面。"); return
        try:
            data_path = self.field("data_path"); label_path = self.field("label_path")
            has_xaxis = self.field("data_has_xaxis"); has_header = self.field("label_has_header")

            # 1. Parse Task Info
            task_info = {}
            target_role_str = "目标"; ignore_role_str = "忽略"; id_role_str = "ID"
            regression_type_str = "回归"; classification_type_str = "分类"
            col_names_in_table = [] # Store names used in table for remapping if no header
            for i in range(page.table.rowCount()):
                col_name_item = page.table.item(i, 0)
                if not col_name_item: continue
                col_name = col_name_item.text(); col_names_in_table.append(col_name)
                role = page.role_combos[i].currentText(); task_type = page.type_combos[i].currentText()
                if role != ignore_role_str:
                    internal_role = "target" if role == target_role_str else ("id" if role == id_role_str else "ignore")
                    internal_type = "regression" if task_type == regression_type_str else ("classification" if task_type == classification_type_str else None)
                    task_info[col_name] = {'role': internal_role, 'type': internal_type if internal_role == 'target' else None}

            # 2. Load Labels (with robust error handling)
            header_opt = 0 if has_header else None
            try: df_labels = pd.read_csv(label_path, header=header_opt, on_bad_lines='warn')
            except FileNotFoundError: raise ValueError(f"找不到标签文件 '{label_path}'")
            except pd.errors.EmptyDataError: raise ValueError(f"标签文件 '{label_path}' 为空")
            except Exception as e: raise ValueError(f"读取标签文件失败: {e}")

            if not has_header:
                 # Assign generated names and remap task_info
                 new_cols = [f"Column {i}" for i in range(len(df_labels.columns))]
                 df_labels.columns = new_cols
                 if task_info:
                      updated_task_info = {}
                      for idx, new_name in enumerate(new_cols):
                           if idx < len(col_names_in_table) and col_names_in_table[idx] in task_info:
                                updated_task_info[new_name] = task_info[col_names_in_table[idx]]
                      task_info = updated_task_info

            # 3. Load Spectra (with robust error handling)
            try:
                if has_xaxis:
                    df_data = pd.read_csv(data_path, header=None, on_bad_lines='warn')
                    if df_data.empty: raise ValueError("光谱文件为空")
                    wavelengths = df_data.iloc[:, 0].to_numpy(dtype=float)
                    spectra_data = df_data.iloc[:, 1:].to_numpy(dtype=float).T
                else:
                    df_data = pd.read_csv(data_path, header=None, on_bad_lines='warn')
                    if df_data.empty: raise ValueError("光谱文件为空")
                    spectra_data = df_data.to_numpy(dtype=float).T
                    wavelengths = np.arange(spectra_data.shape[1], dtype=float)
            except FileNotFoundError: raise ValueError(f"找不到光谱文件 '{data_path}'")
            except pd.errors.EmptyDataError: raise ValueError(f"光谱文件 '{data_path}' 为空")
            except Exception as e: raise ValueError(f"读取光谱文件失败: {e}")


            # 4. Validate Shapes
            if spectra_data.shape[0] != df_labels.shape[0]: raise ValueError(f"光谱样本数({spectra_data.shape[0]})与标签数({df_labels.shape[0]})不匹配")
            if spectra_data.ndim != 2 or wavelengths.ndim != 1 or spectra_data.shape[1] != wavelengths.shape[0]: raise ValueError(f"数据维度不正确: 光谱{spectra_data.shape}, 波长{wavelengths.shape}")

            # 5. Create SpectralProject with Versioning
            # 5. Create SpectralProject with Versioning
            print(f"[DEBUG] Creating SpectralProject with wavelengths shape: {wavelengths.shape}")
            print(f"[DEBUG] Creating SpectralProject with spectra shape: {spectra_data.shape}")
            print(f"[DEBUG] spectra_versions dict will be: {{'original': spectra_data}}")
            project = SpectralProject(
                # --- 1. Fields WITHOUT defaults (必须按顺序在前面) ---
                wavelengths=wavelengths,
                labels_dataframe=df_labels,
                task_info=task_info,
                data_file_path=data_path,
                label_file_path=label_path,

                # --- 2. Fields WITH defaults (在后面) ---
                spectra_versions={'original': spectra_data},  # 覆盖默认
                active_spectra_version='original',  # 覆盖默认
                processing_history=[]  # 覆盖默认
            )
            print(f"[DEBUG] Project created. Active version: {project.active_spectra_version}")
            print(f"[DEBUG] Project spectra_versions: {project.spectra_versions.keys()}")

            self.projectCreated.emit(project)
            super().accept()

        except Exception as e:
            error_msg = f"导入失败: {e}"; self.log_message.emit(error_msg)
            current_page = self.currentPage(); current_page and current_page.setSubTitle(error_msg)
            import traceback; traceback.print_exc() # Print full error for debugging