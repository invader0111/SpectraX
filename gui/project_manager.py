# -*- coding: utf-8 -*-
# 文件名: gui/project_manager.py
# 描述: 左侧的项目浏览器 (全选/全不选对所有任务可用)
# (修改: 支持 'workflow' 和 'visualization' 两种模式)

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QTableWidget,
    QStackedWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QStyle, QLabel, QComboBox, QHBoxLayout,
    QPushButton, QFormLayout
)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot as Slot
from PyQt6.QtGui import QIcon, QColor  # <-- 确保 QColor 已导入
from PyQt6.QtWidgets import QTreeWidgetItemIterator

from core.data_model import SpectralProject

SAMPLE_INDEX_ROLE = Qt.ItemDataRole.UserRole + 1


class ProjectManager(QWidget):
    """
    左侧面板，用于浏览已加载的项目数据。
    (修改: 支持 'workflow' 和 'visualization' 两种模式)
    """
    sample_selected = pyqtSignal(int)
    compare_version_changed = pyqtSignal()
    current_sample_index = -1

    def __init__(self, mode='workflow', parent=None):
        """
        初始化 ProjectManager。
        :param mode: 'workflow' (Page 0) 或 'visualization' (Page 1)
        """
        super().__init__(parent)
        self.project: SpectralProject = None
        self.mode = mode  # 存储当前模式

        main_layout = QVBoxLayout(self);
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- 1. 顶部控制容器 (用于 Page 1 '数据源') ---
        self.controls_container = QWidget()
        self.controls_layout = QFormLayout(self.controls_container)
        self.controls_layout.setContentsMargins(5, 5, 5, 0)
        main_layout.addWidget(self.controls_container)
        self.controls_container.setVisible(False)  # 默认隐藏

        # --- 2. 筛选器小部件 (通用) ---
        self.filter_widget = QWidget();
        filter_layout = QHBoxLayout(self.filter_widget);
        filter_layout.setContentsMargins(5, 5, 5, 0)
        self.filter_label = QLabel("类别筛选:");
        filter_layout.addWidget(self.filter_label)
        self.category_filter_combo = QComboBox();
        filter_layout.addWidget(self.category_filter_combo, 1);
        self.filter_widget.setVisible(False);
        main_layout.addWidget(self.filter_widget)

        # --- 3. 版本选择器 (通用) ---
        # 两个模式都需要 '数据版本' 下拉框
        self.version_widget = QWidget()
        v_layout = QHBoxLayout(self.version_widget)
        v_layout.setContentsMargins(5, 5, 5, 0)
        v_layout.addWidget(QLabel("数据版本:"))
        self.version_selector_combo = QComboBox()  # 主版本
        self.version_selector_combo.setEnabled(False)
        v_layout.addWidget(self.version_selector_combo, 1)
        self.version_widget.setVisible(False)
        main_layout.addWidget(self.version_widget)

        # --- 4. 模式特定控件 ---
        if self.mode == 'workflow':
            # 仅 Page 0 (工作流) 需要 '对比版本'
            self.compare_widget = QWidget()
            c_layout = QHBoxLayout(self.compare_widget)
            c_layout.setContentsMargins(5, 0, 5, 0)
            c_layout.addWidget(QLabel("对比版本:"))
            self.compare_version_combo = QComboBox()
            self.compare_version_combo.setEnabled(False)
            c_layout.addWidget(self.compare_version_combo, 1)
            self.compare_widget.setVisible(False)
            main_layout.addWidget(self.compare_widget)
            # 连接 'workflow' 模式特有的信号
            self.compare_version_combo.currentTextChanged.connect(self.on_compare_version_changed)

        # (Page 1 'visualization' 模式不需要 '对比版本' 控件)

        # --- 5. 全选/全不选按钮 (通用) ---
        self.selection_buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(self.selection_buttons_widget)
        buttons_layout.setContentsMargins(5, 0, 5, 5)
        self.select_all_button = QPushButton("全选可见项")
        self.deselect_all_button = QPushButton("取消所有选中")
        buttons_layout.addWidget(self.select_all_button)
        buttons_layout.addWidget(self.deselect_all_button)
        buttons_layout.addStretch()
        self.selection_buttons_widget.setVisible(False)
        main_layout.addWidget(self.selection_buttons_widget)

        # --- 6. 堆叠控件 (通用) ---
        self.stack = QStackedWidget()
        self.tree_view = QTreeWidget()
        self.table_view = QTableWidget()
        self.stack.addWidget(self.tree_view);
        self.stack.addWidget(self.table_view)
        main_layout.addWidget(self.stack, 1);  # 让 stack 占据主要剩余空间

        # --- 7. 底部按钮容器 (用于 Page 1 '绘制'/'清空') ---
        self.bottom_button_container = QWidget()
        self.bottom_button_layout = QHBoxLayout(self.bottom_button_container)
        self.bottom_button_layout.setContentsMargins(5, 0, 5, 5)
        main_layout.addWidget(self.bottom_button_container)
        self.bottom_button_container.setVisible(False)  # 默认隐藏

        # --- 8. 最终设置和连接 (通用) ---
        self.setLayout(main_layout)

        style = self.style();
        self.category_icon = style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        self.sample_icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)

        self.tree_view.setHeaderHidden(True);
        self.tree_view.itemSelectionChanged.connect(self.on_tree_selection)
        self.table_view.itemSelectionChanged.connect(self.on_table_selection);
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);
        # --- VVVV 修改: Page 1 (可视化) 需要多选 VVVV ---
        if self.mode == 'visualization':
            self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection);
        else:  # 'workflow' 模式保持单选
            self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection);
        # --- ^^^^ 修改结束 ^^^^ ---
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.category_filter_combo.currentTextChanged.connect(self.on_category_filter_changed)
        self.select_all_button.clicked.connect(self.select_all_visible)
        self.deselect_all_button.clicked.connect(self.deselect_all)

        # [新增] 在 filter_widget 附近添加视图切换按钮
        self.view_toggle_btn = QPushButton("切换视图 (树/表)")
        self.view_toggle_btn.clicked.connect(self.toggle_view_mode)
        # 将其添加到布局中 (例如加到 filter_layout 或新的一行)
        if self.mode == 'workflow':
            filter_layout.addWidget(self.view_toggle_btn)

        # (version_selector_combo 的连接由 main.py 处理)

    def _apply_table_filter(self, selected_category: str):
        """
        根据选中的类别隐藏或显示表格行。
        """
        if not self.project: return

        all_categories_text = "-- 显示所有类别 --"

        # 1. 如果选了“全部”，则显示所有行
        if selected_category == all_categories_text:
            for r in range(self.table_view.rowCount()):
                self.table_view.setRowHidden(r, False)
            return

        # 2. 找到分类目标列在表格中的索引
        # (表格列结构: [Checkbox, Sample Index, Col1, Col2, ...])
        target_col, _ = self.project.get_primary_target_col(task_filter='classification')
        if not target_col:
            return  # 如果没有分类列，就不筛选

        target_col_index = -1
        # 遍历表头寻找目标列
        for c in range(self.table_view.columnCount()):
            header_item = self.table_view.horizontalHeaderItem(c)
            if header_item and header_item.text() == target_col:
                target_col_index = c
                break

        if target_col_index == -1:
            print(f"警告: 在表格中未找到分类列 '{target_col}'，无法筛选。")
            return

        # 3. 遍历行，匹配类别文本
        for r in range(self.table_view.rowCount()):
            item = self.table_view.item(r, target_col_index)
            if item:
                val = item.text()
                # 检查值是否匹配 (通常是精确匹配)
                if val == selected_category:
                    self.table_view.setRowHidden(r, False)
                else:
                    self.table_view.setRowHidden(r, True)

    # --- [修改] 类别下拉框变更槽函数 ---
    @Slot(str)
    def on_category_filter_changed(self, selected_category: str):
        """
        (修改) 当类别筛选器变化时触发。
        现在同时支持 树状图 (隐藏节点) 和 表格 (隐藏行)。
        """
        current_widget = self.stack.currentWidget()
        all_categories_text = "-- 显示所有类别 --"

        # === 情况 A: 当前是树状图 (原有逻辑) ===
        if current_widget == self.tree_view:
            for i in range(self.tree_view.topLevelItemCount()):
                item = self.tree_view.topLevelItem(i)
                if selected_category == all_categories_text:
                    item.setHidden(False)
                else:
                    item_text = item.text(0)
                    if item_text.startswith(selected_category):
                        item.setHidden(False)
                    else:
                        item.setHidden(True)

        # === 情况 B: 当前是表格 (新增逻辑) ===
        elif current_widget == self.table_view:
            self._apply_table_filter(selected_category)

    # --- [修改/新增] 视图切换逻辑 ---
    # (这是上一轮建议中添加的方法，这里增加了筛选同步逻辑)
    def toggle_view_mode(self):
        """
        在树状图和表格之间切换，并同步当前的筛选状态。
        """
        if not self.project: return

        # 获取当前选中的筛选类别
        current_filter = self.category_filter_combo.currentText()

        if self.stack.currentWidget() == self.tree_view:
            # 1. 切换到表格
            self.populate_table_view(self.project)  # 重建表格
            self.stack.setCurrentWidget(self.table_view)

            # 2. [关键] 立即应用筛选
            self._apply_table_filter(current_filter)
            print(f"视图切换: 表格模式 (已应用筛选: {current_filter})")

        else:
            # 切换回树状图 (仅当是分类任务时)
            if self.project.get_task_summary() == 'classification':
                self.populate_tree_view(self.project)
                self.stack.setCurrentWidget(self.tree_view)

                # 树状图重建后，需要重新触发一下筛选逻辑来隐藏节点
                self.on_category_filter_changed(current_filter)
                print(f"视图切换: 树状模式 (已应用筛选: {current_filter})")

    # --- VVVV 新增: 动态添加控件的方法 VVVV ---
    def add_control_widget(self, label_text: str, widget: QWidget):
        """
        (Page 1) 向管理器顶部的控制容器添加一个小部件 (例如 '数据源' 下拉框)。
        """
        self.controls_layout.addRow(label_text, widget)
        self.controls_container.setVisible(True)

    def add_bottom_buttons(self, button_list: list):
        """
        (Page 1) 向管理器底部的容器添加按钮 (例如 '绘制'/'清空')。
        """
        for btn in button_list:
            self.bottom_button_layout.addWidget(btn)
        self.bottom_button_container.setVisible(True)

    # --- ^^^^ 新增结束 ^^^^ ---

    def get_current_selected_index(self) -> int:
        """(Page 0) 获取当前点击选中的单个样本索引"""
        return self.current_sample_index

    # --- VVVV 'workflow' 模式独有的方法 VVVV ---
    @Slot()
    def on_compare_version_changed(self):
        """(Page 0) 当对比下拉框变化时，发射信号通知 main.py 重绘"""
        if self.mode == 'workflow':
            self.compare_version_changed.emit()

    def get_compare_version_name(self) -> str | None:
        """(Page 0) 获取当前选中的对比版本名称"""
        if self.mode != 'workflow' or not hasattr(self, 'compare_version_combo'):
            return None

        if not self.compare_version_combo.isEnabled():
            return None
        text = self.compare_version_combo.currentText()
        if text == "-- 无对比 --" or not text:
            return None
        return text

    # --- ^^^^ 结束 ^^^^ ---

    def load_project_data(self, project: SpectralProject | None):
        if project is None:
            # 如果项目为空，清空所有视图和控件
            self.project = None
            self.current_sample_index = -1
            self.tree_view.clear()
            self.table_view.clear()
            self.table_view.setRowCount(0)
            self.table_view.setColumnCount(0)

            # 隐藏/禁用相关控件
            self.filter_widget.setVisible(False)
            self.selection_buttons_widget.setVisible(False)

            # (安全地隐藏所有可能的控件)
            if hasattr(self, 'version_widget'): self.version_widget.setVisible(False)
            if hasattr(self, 'compare_widget'): self.compare_widget.setVisible(False)
            if hasattr(self, 'version_selector_combo'): self.version_selector_combo.setEnabled(False)
            if hasattr(self, 'compare_version_combo'): self.compare_version_combo.setEnabled(False)
            return

        self.project = project
        task_type = project.get_task_summary()
        self.current_sample_index = -1

        self.selection_buttons_widget.setVisible(True)

        # --- VVVV 修改: 模式感知的控件显示 VVVV ---
        if self.mode == 'workflow':
            if hasattr(self, 'version_widget'): self.version_widget.setVisible(True)
            if hasattr(self, 'compare_widget'): self.compare_widget.setVisible(True)
        elif self.mode == 'visualization':
            if hasattr(self, 'version_widget'): self.version_widget.setVisible(True)
            if hasattr(self, 'compare_widget'): self.compare_widget.setVisible(False)  # 确保隐藏
        # --- ^^^^ 修改结束 ^^^^ ---

        if task_type == 'classification':
            self.populate_tree_view(project)
            self.populate_category_filter(project)
            self.filter_widget.setVisible(True)
            self.stack.setCurrentWidget(self.tree_view)
        else:
            self.populate_table_view(project)
            self.filter_widget.setVisible(False)
            self.stack.setCurrentWidget(self.table_view)

    # --- VVVV 新增: 样本可用性控制 (用于 Page 1) VVVV ---
    @Slot(set)
    def update_sample_availability(self, valid_indices_set: set | None):
        """
        (Page 1) 根据传入的有效索引集合，更新样本列表的可用性。
        - valid_indices_set: 包含有效样本索引的集合。
        - 如果为 None，则所有样本都可用。
        """
        if self.mode != 'visualization':
            return  # 此功能仅用于可视化模式

        print(
            f"[DEBUG ProjectManager] 更新样本可用性，有效数量: {len(valid_indices_set) if valid_indices_set is not None else 'ALL'}")

        current_widget = self.stack.currentWidget()

        if current_widget == self.tree_view:
            iterator = QTreeWidgetItemIterator(self.tree_view)
            while iterator.value():
                item = iterator.value()
                # 只操作样本节点 (有父节点且无子节点)
                if item.parent() is not None and item.childCount() == 0:
                    sample_index = item.data(0, SAMPLE_INDEX_ROLE)
                    if sample_index is None: continue

                    is_valid = (valid_indices_set is None) or (sample_index in valid_indices_set)
                    self._set_item_selectable(item, is_valid)
                iterator += 1

        elif current_widget == self.table_view:
            for row in range(self.table_view.rowCount()):
                # 样本索引存储在复选框项(第0列)
                item = self.table_view.item(row, 0)
                if item is None: continue

                sample_index = item.data(SAMPLE_INDEX_ROLE)
                if sample_index is None:
                    try:
                        sample_index = int(self.table_view.item(row, 1).text())
                    except:
                        continue

                is_valid = (valid_indices_set is None) or (sample_index in valid_indices_set)

                # 遍历整行来设置可用性
                for col in range(self.table_view.columnCount()):
                    row_item = self.table_view.item(row, col)
                    if row_item:  # QTableWidgetItem
                        self._set_item_selectable(row_item, is_valid)

    def _set_item_selectable(self, item, is_selectable: bool):
        """(辅助函数) 设置项是否可用(可选/可勾选)并更改外观"""
        flags = item.flags()
        if is_selectable:
            # 设为可用, 可选, 可勾选
            flags |= (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)

            # (已修复) 根据 item 类型调用不同方法
            if isinstance(item, QTreeWidgetItem):
                item.setForeground(0, QColor(Qt.GlobalColor.black))  # 需要列索引 0
            elif isinstance(item, QTableWidgetItem):
                item.setForeground(QColor(Qt.GlobalColor.black))  # 不需要列索引

        else:
            # 设为不可用, 不可选, 不可勾选
            flags &= ~(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)

            # (已修复) 根据 item 类型调用不同方法
            if isinstance(item, QTreeWidgetItem):
                item.setForeground(0, QColor(Qt.GlobalColor.gray))  # 需要列索引 0
            elif isinstance(item, QTableWidgetItem):
                item.setForeground(QColor(Qt.GlobalColor.gray))  # 不需要列索引

            # --- VVVV 关键修复 VVVV ---
            # 禁用时总是取消勾选 (也需要区分 item 类型)
            if isinstance(item, QTreeWidgetItem):
                item.setCheckState(0, Qt.CheckState.Unchecked)  # 需要列索引 0
            elif isinstance(item, QTableWidgetItem):
                item.setCheckState(Qt.CheckState.Unchecked)  # 不需要列索引
            # --- ^^^^ 修复结束 ^^^^ ---

        item.setFlags(flags)

    def populate_category_filter(self, project: SpectralProject):
        # (完整实现 - 保持不变)
        self.category_filter_combo.blockSignals(True)
        self.category_filter_combo.clear()
        self.category_filter_combo.addItem("-- 显示所有类别 --")
        target_col, _ = project.get_primary_target_col(task_filter='classification')
        if target_col:
            try:
                categories = project.labels_dataframe[target_col].unique()
                self.category_filter_combo.addItems(sorted([str(c) for c in categories]))
            except Exception as e:
                print(f"填充类别筛选器时出错: {e}")
        self.category_filter_combo.blockSignals(False)

    @Slot(str)
    def on_category_filter_changed(self, selected_category: str):
        """
        (修改) 当类别筛选器变化时触发。
        现在同时支持 树状图 (隐藏节点) 和 表格 (隐藏行)。
        """
        current_widget = self.stack.currentWidget()
        all_categories_text = "-- 显示所有类别 --"

        # === 情况 A: 当前是树状图 (原有逻辑) ===
        if current_widget == self.tree_view:
            for i in range(self.tree_view.topLevelItemCount()):
                item = self.tree_view.topLevelItem(i)
                if selected_category == all_categories_text:
                    item.setHidden(False)
                else:
                    # --- [修复 2/2] 使用原始数据进行精确匹配 ---
                    # 获取之前存储的原始类别数据
                    cat_data = item.data(0, Qt.ItemDataRole.UserRole)

                    # 必须转换为字符串比较，因为 selected_category 是字符串，
                    # 而 cat_data 可能是整数或浮点数
                    if str(cat_data) == selected_category:
                        item.setHidden(False)
                    else:
                        item.setHidden(True)
                    # ----------------------------------------

        # === 情况 B: 当前是表格 (新增逻辑) ===
        elif current_widget == self.table_view:
            self._apply_table_filter(selected_category)

    def populate_tree_view(self, project: SpectralProject):
        """
        填充树状视图（用于分类任务）。
        包含修复：在节点 UserRole 中存储原始类别值以支持精确筛选。
        """
        self.tree_view.clear()

        # 获取分类目标列
        target_col, _ = project.get_primary_target_col(task_filter='classification')
        id_col = project.get_id_col()

        if not target_col:
            self.tree_view.addTopLevelItem(QTreeWidgetItem(["错误: 未找到分类目标列"]))
            return

        # 获取所有唯一类别
        categories = project.labels_dataframe[target_col].unique()

        # 尝试排序类别以保持显示整洁（如果类型支持排序）
        try:
            categories = sorted(categories, key=lambda x: str(x))
        except Exception:
            pass  # 保持原序

        df = project.labels_dataframe

        for category in categories:
            # 1. 创建父节点（类别节点）
            # 初始文本暂定，后面会更新带数量的文本
            cat_item = QTreeWidgetItem([f"{category} (计算中...)"])
            cat_item.setIcon(0, self.category_icon)
            # 父节点通常不可勾选，或者需要单独处理勾选逻辑，这里设为不可勾选以免混淆
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

            # --- [关键修改] 存储原始类别值用于精确筛选 ---
            # 将原始的 category 值存储在 UserRole 中。
            # 这样在筛选时可以取出这个值进行 '==' 比较，而不是 startswith 匹配文本。
            cat_item.setData(0, Qt.ItemDataRole.UserRole, category)
            # ----------------------------------------

            self.tree_view.addTopLevelItem(cat_item)

            count = 0
            # 2. 查找属于该类别的所有样本
            sample_indices = df[df[target_col] == category].index

            for i in sample_indices:
                # 构建样本标签
                label = f"Sample {i}"
                if id_col:
                    # 如果有 ID 列，显示 ID
                    val = df.loc[i, id_col]
                    label += f" ({val})"

                # 创建子节点（样本节点）
                sample_item = QTreeWidgetItem([label])
                sample_item.setIcon(0, self.sample_icon)

                # 存储样本索引 (SAMPLE_INDEX_ROLE = UserRole + 1)
                sample_item.setData(0, SAMPLE_INDEX_ROLE, i)

                # 设置样本节点为可勾选
                sample_item.setFlags(sample_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                sample_item.setCheckState(0, Qt.CheckState.Unchecked)

                cat_item.addChild(sample_item)
                count += 1

            # 3. 更新父节点文本，显示样本数量
            cat_item.setText(0, f"{category} (共 {count} 个)")

            # 默认展开
            cat_item.setExpanded(True)

    def populate_table_view(self, project: SpectralProject):
        # (完整实现 - 保持不变)
        self.table_view.clear()
        df = project.labels_dataframe
        display_cols = []
        id_col = project.get_id_col()
        if id_col: display_cols.append(id_col)
        for col, info in project.task_info.items():
            if info['role'] == 'target': display_cols.append(col)
        if not display_cols: display_cols = df.columns.tolist()

        self.table_view.setRowCount(len(df))
        self.table_view.setColumnCount(len(display_cols) + 2)
        header = ["", "Sample Index"] + display_cols
        self.table_view.setHorizontalHeaderLabels(header)

        for i in range(len(df)):
            checkbox_item = QTableWidgetItem();
            checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled);
            checkbox_item.setCheckState(Qt.CheckState.Unchecked);
            checkbox_item.setData(SAMPLE_INDEX_ROLE, i);
            self.table_view.setItem(i, 0, checkbox_item)
            idx_item = QTableWidgetItem(str(i));
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEditable);
            self.table_view.setItem(i, 1, idx_item)
            for j, col_name in enumerate(display_cols):
                val = df.loc[i, col_name];
                self.table_view.setItem(i, j + 2, QTableWidgetItem(str(val)))
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.horizontalHeader().setSectionResizeMode(len(display_cols) + 1, QHeaderView.ResizeMode.Stretch)
        self.table_view.resizeColumnToContents(0)

    def on_tree_selection(self):
        # (完整实现 - 保持不变)
        selected_items = self.tree_view.selectedItems()
        if not selected_items: self.current_sample_index = -1; return
        item = selected_items[0]
        if item.childCount() > 0: self.current_sample_index = -1; return
        sample_index = item.data(0, SAMPLE_INDEX_ROLE)
        if sample_index is not None:
            self.current_sample_index = int(sample_index)
            self.sample_selected.emit(int(sample_index))
        else:
            self.current_sample_index = -1

    def on_table_selection(self):
        # (修改: 区分 Page 0 单选 和 Page 1 多选)
        if self.mode == 'visualization':
            # 在 Page 1, 我们只关心点击的那个（用于预览），
            # 勾选状态由 get_checked_items_indices 处理。
            # 我们仍然可以更新 current_sample_index 以便预览。
            selected_indexes = self.table_view.selectionModel().selectedIndexes()
            if not selected_indexes:
                self.current_sample_index = -1
                return
            row = selected_indexes[0].row()  # 只取第一个点击的
        else:
            # Page 0 保持单选行为
            selected_rows = self.table_view.selectionModel().selectedRows()
            if not selected_rows: self.current_sample_index = -1; return
            row = selected_rows[0].row()

        try:
            # 逻辑保持一致：获取索引
            index_item = self.table_view.item(row, 1)  # 第1列是 Sample Index
            if index_item:
                sample_index = int(index_item.text())
                self.current_sample_index = sample_index
                self.sample_selected.emit(sample_index)  # 发送信号以触发预览
            else:
                self.current_sample_index = -1
        except Exception as e:
            self.current_sample_index = -1;
            print(f"无法从表格获取索引: {e}")

    @Slot()
    def select_all_visible(self):
        # (完整实现 - 保持不变)
        current_widget = self.stack.currentWidget()
        if current_widget == self.tree_view:
            selected_category_text = self.category_filter_combo.currentText()
            all_categories_text = "-- 显示所有类别 --"
            for i in range(self.tree_view.topLevelItemCount()):
                cat_item = self.tree_view.topLevelItem(i)
                if not cat_item.isHidden():
                    should_select_this_category = False
                    if selected_category_text == all_categories_text:
                        should_select_this_category = True
                    elif cat_item.text(0).startswith(selected_category_text):
                        should_select_this_category = True
                    if should_select_this_category:
                        for j in range(cat_item.childCount()):
                            sample_item = cat_item.child(j)
                            # (新增) 只勾选可用的
                            if sample_item.data(0, SAMPLE_INDEX_ROLE) is not None and (
                                    sample_item.flags() & Qt.ItemFlag.ItemIsEnabled):
                                sample_item.setCheckState(0, Qt.CheckState.Checked)
        elif current_widget == self.table_view:
            for row in range(self.table_view.rowCount()):
                checkbox_item = self.table_view.item(row, 0)
                # (新增) 只勾选可用的
                if checkbox_item and (checkbox_item.flags() & Qt.ItemFlag.ItemIsEnabled):
                    checkbox_item.setCheckState(Qt.CheckState.Checked)

    @Slot()
    def deselect_all(self):
        # (完整实现 - 保持不变)
        current_widget = self.stack.currentWidget()
        if current_widget == self.tree_view:
            iterator = QTreeWidgetItemIterator(self.tree_view)
            while iterator.value():
                item = iterator.value()
                if item.parent() is not None and item.childCount() == 0 and item.data(0, SAMPLE_INDEX_ROLE) is not None:
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                iterator += 1
        elif current_widget == self.table_view:
            for row in range(self.table_view.rowCount()):
                checkbox_item = self.table_view.item(row, 0)
                if checkbox_item: checkbox_item.setCheckState(Qt.CheckState.Unchecked)

    def get_checked_items_indices(self):
        # (完整实现 - 保持不变)
        checked_indices = []
        current_widget = self.stack.currentWidget()
        if current_widget == self.tree_view:
            iterator = QTreeWidgetItemIterator(self.tree_view, QTreeWidgetItemIterator.IteratorFlag.Checked)
            while iterator.value():
                item = iterator.value()
                if item.parent() is not None and item.childCount() == 0:
                    # (新增) 只返回可用的
                    if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                        sample_index = item.data(0, SAMPLE_INDEX_ROLE)
                        if sample_index is not None: checked_indices.append(int(sample_index))
                iterator += 1
        elif current_widget == self.table_view:
            for row in range(self.table_view.rowCount()):
                checkbox_item = self.table_view.item(row, 0)
                # (新增) 只返回可用的
                if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked and (
                        checkbox_item.flags() & Qt.ItemFlag.ItemIsEnabled):
                    sample_index = checkbox_item.data(SAMPLE_INDEX_ROLE)
                    if sample_index is not None: checked_indices.append(int(sample_index))
        return sorted(list(set(checked_indices)))