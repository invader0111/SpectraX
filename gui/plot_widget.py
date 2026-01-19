# -*- coding: utf-8 -*-
# 文件名: gui/plot_widget.py
# 描述: 承载 Matplotlib 画布的中心模块
# (修改: plot_spectrum 现在会在存在对比数据时绘制残差/差值图)

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSlider, QHBoxLayout, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot as Slot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback

try:
    from core.data_model import SpectralProject
    from core.peak_fitting_models import get_single_peak_function, get_params_per_peak
except ImportError:
    class SpectralProject:
        pass


    def get_single_peak_function(s):
        return lambda x, *p: np.zeros_like(x)


    def get_params_per_peak(s):
        return 3

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- Interaction Mode Constants ---
MODE_DISABLED = 0;
MODE_PICK_ANCHOR = 1;
MODE_SELECT_REGION = 2;
MODE_THRESHOLD_LINE = 3


# --- 自定义工具栏 ---
class CustomPlotToolbar(NavigationToolbar):
    """
    一个自定义的 Matplotlib 工具栏，集成了主/副图比例滑块。
    """
    # 当滑块值改变时发出信号 (int: 10-90)
    ratio_changed = pyqtSignal(int)

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)

        # 1. 创建滑块控件
        self.slider_widget = QWidget()
        slider_layout = QHBoxLayout(self.slider_widget)
        slider_layout.setContentsMargins(4, 0, 4, 0)

        slider_layout.addWidget(QLabel("主/副图比例:"))
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(10, 90)  # 主图占比 10% 到 90%
        self.ratio_slider.setValue(75)  # 默认值 75% (即 3:1)
        self.ratio_slider.setToolTip("调整主图和副图的显示比例")

        # 2. 连接滑块信号到类信号
        self.ratio_slider.valueChanged.connect(self.ratio_changed.emit)

        slider_layout.addWidget(self.ratio_slider)

        # 3. 将控件添加到工具栏
        self.addSeparator()  # 在标准按钮后添加分隔符
        self.addWidget(self.slider_widget)

        self.set_slider_visible(False)  # 默认隐藏

    @Slot(bool)
    def set_slider_visible(self, visible: bool):
        """显示或隐藏滑块控件"""
        self.slider_widget.setVisible(visible)


class PlotWidget(QWidget):
    """Matplotlib Plot Widget with interactive capabilities."""
    # --- Signals ---
    anchor_added = pyqtSignal(float)
    region_defined = pyqtSignal(float, float)
    threshold_set = pyqtSignal(float)
    anchor_delete_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # 实例化自定义工具栏
        self.toolbar = CustomPlotToolbar(self.canvas, self)

        # 保存 gridspec
        self.gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1])
        self.ax_main = self.figure.add_subplot(self.gs[0])
        self.ax_sub = self.figure.add_subplot(self.gs[1], sharex=self.ax_main)

        plt.setp(self.ax_main.get_xticklabels(), visible=False)

        layout = QVBoxLayout(self);
        layout.addWidget(self.toolbar);
        layout.addWidget(self.canvas)

        # --- Interaction State ---
        self.interaction_mode = MODE_DISABLED
        self.region_start_x = None
        self.plot_region_spans = []
        self.plot_anchor_lines = []
        self.threshold_line = None
        self.dragging_threshold = False

        # --- Connect Matplotlib Events ---
        self.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)

        # 连接到工具栏的新信号
        self.toolbar.ratio_changed.connect(self.on_ratio_slider_changed)

        # --- Initial Setup ---
        self.ax_main.set_title("请先导入项目...");
        self.ax_main.set_ylabel("强度")
        self.ax_sub.set_xlabel("波长 / 像素索引");
        self.ax_sub.set_ylabel("强度")
        self.figure.tight_layout()

    def _clear_axes(self):
        """Clears both axes and resets all visual helpers and states."""
        self.ax_main.clear();
        self.ax_sub.clear()

        # 调用工具栏的方法隐藏滑块
        self.toolbar.set_slider_visible(False)

        self.plot_region_spans = [];
        self.plot_anchor_lines = []
        self.threshold_line = None;
        self.dragging_threshold = False
        self.region_start_x = None

        self.ax_main.set_ylabel("强度")
        plt.setp(self.ax_main.get_xticklabels(), visible=False)
        self.ax_sub.set_xlabel("波长 / 像素索引");
        self.ax_sub.set_ylabel("强度")

    def _remove_threshold_line(self):
        """Removes the threshold line if it exists."""
        if self.threshold_line:
            try:
                self.threshold_line.remove()
            except:
                pass
            self.threshold_line = None
            self.canvas.draw_idle()

    @Slot(int)
    def on_ratio_slider_changed(self, value: int):
        """当滑块移动时，更新 gridspec 的高宽比"""
        try:
            main_ratio = value
            sub_ratio = 100 - value
            if sub_ratio <= 0: sub_ratio = 1

            if hasattr(self, 'gs'):
                self.gs.set_height_ratios([main_ratio, sub_ratio])
                self.figure.tight_layout()
                self.canvas.draw_idle()
        except Exception as e:
            print(f"警告: 设置绘图比例失败: {e}")

    # --- Interaction Control Slots ---
    @Slot(int)
    def set_interaction_mode(self, mode):
        """Sets the current plot interaction mode."""
        previous_mode = self.interaction_mode
        self.interaction_mode = mode
        self.region_start_x = None;
        self.dragging_threshold = False

        if mode in [MODE_PICK_ANCHOR, MODE_SELECT_REGION]:
            cursor = Qt.CursorShape.CrossCursor
            if previous_mode == MODE_THRESHOLD_LINE: self._remove_threshold_line()
        elif mode == MODE_THRESHOLD_LINE:
            cursor = Qt.CursorShape.SizeVerCursor
            # Line is drawn by on_update_threshold_line
        else:  # MODE_DISABLED
            cursor = Qt.CursorShape.ArrowCursor
            if previous_mode == MODE_THRESHOLD_LINE: self._remove_threshold_line()
        self.canvas.setCursor(cursor)

    @Slot(float)
    def on_update_threshold_line(self, value):
        """Draws or updates the threshold line based on value."""
        current_visible = self.threshold_line is not None
        should_be_visible = (self.interaction_mode == MODE_THRESHOLD_LINE and np.isfinite(value))

        if should_be_visible:
            if current_visible:
                # Update existing line
                self.threshold_line.set_ydata([value, value])
            else:
                # Draw new line
                try:
                    self.threshold_line = self.ax_main.axhline(
                        value, color='red', linestyle='--', linewidth=1.5, picker=5  # 5 points tolerance
                    )
                except Exception as e:
                    print(f"Error drawing threshold line: {e}")
            self.canvas.draw_idle()
        elif current_visible:
            # Should not be visible, but exists -> remove it
            self._remove_threshold_line()

    # --- Matplotlib Event Callbacks ---
    def on_canvas_press(self, event):
        """Handles mouse button press events (Left=Add/Select, Right=Delete)."""
        if event.inaxes != self.ax_main: return
        x, y = event.xdata, event.ydata
        if x is None or y is None: return

        # --- Left Click Logic ---
        if event.button == 1:
            if self.interaction_mode == MODE_PICK_ANCHOR:
                self.anchor_added.emit(x)
            elif self.interaction_mode == MODE_SELECT_REGION:
                if self.region_start_x is None:
                    self.region_start_x = x; print(f"[P] Region Start:{x:.1f}")
                else:
                    print(f"[P] Region End:{x:.1f}"); self.region_defined.emit(self.region_start_x,
                                                                               x); self.region_start_x = None
            elif self.interaction_mode == MODE_THRESHOLD_LINE and self.threshold_line:
                line_y = self.threshold_line.get_ydata()[0]
                try:  # Check proximity using pixel distance
                    pixel_tolerance = 5  # Pixels
                    line_display_y = self.ax_main.transData.transform((0, line_y))[1]
                    click_display_y = event.y  # Event y is already in display coords
                    if abs(click_display_y - line_display_y) < pixel_tolerance:
                        self.dragging_threshold = True;
                        print("[P] Start drag thresh")
                except Exception as e:
                    print(f"Err check thresh prox: {e}")

        # --- Right Click Logic for Deleting Anchors ---
        elif event.button == 3:
            if self.interaction_mode in [MODE_DISABLED, MODE_PICK_ANCHOR]:
                closest_anchor_x = None;
                min_pixel_dist = float('inf')
                pixel_tolerance = 5  # How close in pixels to click

                if not self.plot_anchor_lines: return  # No anchors exist

                try:
                    click_pixel_x = event.x  # Event x is display coords
                    for line in self.plot_anchor_lines:
                        line_x_data = line.get_xdata()[0]
                        line_pixel_x = self.ax_main.transData.transform((line_x_data, 0))[0]
                        dist = abs(click_pixel_x - line_pixel_x)

                        if dist < pixel_tolerance and dist < min_pixel_dist:
                            min_pixel_dist = dist;
                            closest_anchor_x = line_x_data

                    if closest_anchor_x is not None:
                        print(f"[Plot] R-click near anchor {closest_anchor_x:.2f}. Requesting deletion.")
                        self.anchor_delete_requested.emit(closest_anchor_x)  # Emit signal
                except Exception as e:
                    print(f"Error finding anchor on R-click: {e}")

    def on_canvas_motion(self, event):
        """Handles mouse motion events for threshold line dragging."""
        if self.dragging_threshold and event.inaxes == self.ax_main and event.ydata is not None:
            if self.threshold_line:
                y_min, y_max = self.ax_main.get_ylim()
                new_y = max(y_min, min(event.ydata, y_max))  # Clamp
                self.threshold_line.set_ydata([new_y, new_y])
                self.canvas.draw_idle()  # Use draw_idle for smoother dragging

    def on_canvas_release(self, event):
        """Handles mouse button release events to stop dragging."""
        if event.button == 1 and self.dragging_threshold:
            self.dragging_threshold = False
            if self.threshold_line:
                final_y = self.threshold_line.get_ydata()[0]
                print(f"[P] Stop drag thresh at {final_y:.2f}")
                self.threshold_set.emit(final_y)  # Emit the final value

    # --- Slots to Update Visual Helpers ---
    @Slot(list)
    def on_update_regions(self, regions_list):  # Simplified
        """Updates the highlighted regions (axvspan)."""
        for span in self.plot_region_spans:
            try:
                span.remove()
            except:
                pass;
        self.plot_region_spans.clear()
        if regions_list:
            for s, e in regions_list:
                try:
                    sp = self.ax_main.axvspan(min(s, e), max(s, e), color='y', alpha=0.2,
                                              zorder=0); self.plot_region_spans.append(sp)
                except Exception as ex:
                    print(f"Error drawing region span: {ex}")
        self.canvas.draw_idle()

    @Slot(list)
    def on_update_anchors(self, anchors_list):  # Simplified
        """Updates the vertical lines indicating anchors (axvline)."""
        for line in self.plot_anchor_lines:
            try:
                line.remove()
            except:
                pass;
        self.plot_anchor_lines.clear()
        if anchors_list:
            for x in anchors_list:
                try:
                    li = self.ax_main.axvline(x, color='m', ls=':', lw=1, alpha=0.8); self.plot_anchor_lines.append(li)
                except Exception as ex:
                    print(f"Error drawing anchor line: {ex}")
        self.canvas.draw_idle()

    # --- Plotting Functions ---

    def plot_spectrum(self, project: SpectralProject, idx: int, compare_version_name: str | None = None):
        """
        绘制指定索引的激活版本光谱，并可选对比另一个版本。
        (修改: 如果存在对比版本，则在副图绘制残差)
        """
        try:
            self._clear_axes()  # 清除画布 (这也会隐藏滑块)

            # --- 1. 绘制激活版本 ---
            x = project.wavelengths
            y_active = project.get_active_spectrum_by_index(idx)
            active_version = project.active_spectra_version

            if y_active is None:
                self.ax_main.set_title(f"错误: 无法加载样本 {idx} 版本 '{active_version}'")
                print(f"错误: 无法加载样本 {idx} 版本 '{active_version}'")
                self.canvas.draw()
                return

            # --- 2. 获取标签信息用于标题 ---
            info = project.labels_dataframe.iloc[idx]
            target, _ = project.get_primary_target_col()
            title = f"Smp {idx} (激活: {active_version})"
            if target and target in info:
                title += f" | {target}: {info[target]}"

            # --- 3. 绘图 (激活) ---
            self.ax_main.plot(x, y_active, label=f"激活: {active_version}", zorder=10)

            y_compare_sample = None

            # --- 4. 绘制对比版本 ---
            if compare_version_name and compare_version_name != active_version:
                # 直接从 project.spectra_versions 获取对比数据
                y_compare_data = project.spectra_versions.get(compare_version_name)
                if y_compare_data is not None and 0 <= idx < y_compare_data.shape[0]:
                    y_compare_sample = y_compare_data[idx]
                    self.ax_main.plot(x, y_compare_sample,
                                      label=f"对比: {compare_version_name}",
                                      linestyle='--', color='gray', alpha=0.8, zorder=5)
                    title += f"\n(对比: {compare_version_name})"
                else:
                    print(f"警告: 无法加载对比版本 '{compare_version_name}' 的样本 {idx}")

            # --- 5. (新增) 绘制残差/差值 (如果有对比数据) ---
            if y_compare_sample is not None:
                self.ax_sub.set_visible(True)
                self.toolbar.set_slider_visible(True)  # 显示比例滑块

                # 计算差值: 激活 - 对比
                # (通常如果是 fitted vs original，这里的残差意义取决于谁是激活版本)
                residual = y_active - y_compare_sample

                self.ax_sub.plot(x, residual, color='black', linewidth=1, label='差值 (激活-对比)')
                self.ax_sub.axhline(0, color='gray', linestyle='--', linewidth=0.8)

                self.ax_sub.set_ylabel("差值")
                # 修改图注: 使用 'best' 位置并允许拖动
                self.ax_sub.legend(loc='best', fontsize='small').set_draggable(True)
                self.ax_sub.grid(True, linestyle=':')
            else:
                self.ax_sub.set_visible(False)
                # self.toolbar.set_slider_visible(False) # 已在 _clear_axes 中处理

            # --- 6. 最终设置 ---
            self.ax_main.set_title(title)
            # 修改图注: 使用 'best' 位置并允许拖动
            self.ax_main.legend(loc='best').set_draggable(True)
            self.ax_main.grid(True, linestyle='--')
            self.figure.tight_layout()
            self.canvas.draw()

        except IndexError:
            self._clear_axes()
            self.ax_main.set_title(f"绘图错误: 样本索引 {idx} 超出标签范围")
            print(f"绘图错误: 样本索引 {idx} 超出标签范围")
            self.canvas.draw()
        except Exception as e:
            self._handle_plot_error(e)

    def plot_baseline_results(self, algo, x, y, corr, bl=None, ref=None, title=""):
        try:
            self._clear_axes();
            self.ax_sub.set_visible(True)
            self.toolbar.set_slider_visible(True)  # <-- 修改: 调用工具栏的方法

            if algo == 'AirPLS':
                b = bl if bl is not None else np.zeros_like(x); self.ax_main.plot(x, y, 'b--', label='原始', lw=1,
                                                                                  alpha=0.5); self.ax_main.plot(x, b,
                                                                                                                'g-',
                                                                                                                label='基线',
                                                                                                                lw=2); self.ax_sub.plot(
                    x, corr, 'k-', label='校正后', lw=1.5); self.ax_sub.axhline(0, c='k', ls='--', lw=1)
            elif algo == 'MSC':
                r = ref if ref is not None else np.zeros_like(x); self.ax_main.plot(x, y, 'b-', label='原始',
                                                                                    lw=1.5); self.ax_main.plot(x, r,
                                                                                                               'r:',
                                                                                                               label='参考',
                                                                                                               lw=1); self.ax_sub.plot(
                    x, corr, 'k-', label='校正后', lw=1.5)

            self.ax_main.set_title(f"{algo}:{title}", fontsize=12);
            # 修改图注
            self.ax_main.legend(loc='best').set_draggable(True);
            self.ax_main.grid(True, ls='--');
            h, l = self.ax_sub.get_legend_handles_labels();
            # 修改图注
            if l: self.ax_sub.legend(loc='best').set_draggable(True);
            self.ax_sub.grid(True, ls='--');
            self.figure.tight_layout();
            self.canvas.draw()
        except Exception as e:
            self._handle_plot_error(e)

    def plot_denoise_results(self, x, y, den, title=""):
        try:
            self._clear_axes();
            self.ax_sub.set_visible(True)
            self.toolbar.set_slider_visible(True)  # <-- 修改: 调用工具栏的方法

            self.ax_main.plot(x, y, 'b--', label='原始', lw=1, alpha=0.6);
            self.ax_main.plot(x, den, 'r-', label='降噪', lw=1.5);
            self.ax_main.set_title(f'降噪:{title}', fontsize=12);
            # 修改图注
            self.ax_main.legend(loc='best').set_draggable(True);
            self.ax_main.grid(True, ls='--');
            self.ax_sub.plot(x, den, 'k-', label='降噪', lw=1.5);
            h, l = self.ax_sub.get_legend_handles_labels();
            # 修改图注
            if l: self.ax_sub.legend(loc='best').set_draggable(True);
            self.ax_sub.grid(True, ls='--');
            self.figure.tight_layout();
            self.canvas.draw()
        except Exception as e:
            self._handle_plot_error(e)

    def plot_peak_fitting_results(self, x, y, fit_y,
                                  params_df: pd.DataFrame,
                                  peak_shape: str,
                                  title=""):
        """
        (修改) 使用 params_df (DataFrame) 来绘制拟合结果和组件。
        (修改) 使用“填充”样式 (fill) 来显示构成峰。
        """
        try:
            self._clear_axes();
            self.ax_sub.set_visible(True)
            self.toolbar.set_slider_visible(True)

            self.ax_main.plot(x, y, 'b.', label='原始', markersize=2, alpha=0.6);
            self.ax_main.plot(x, fit_y, 'r-', label='拟合', lw=2)

            # --- VVVV 关键修改：从 DataFrame 绘制“填充”组件 VVVV ---
            try:
                # 确定需要哪些列和哪个Numpy峰型函数
                if peak_shape == 'voigt':
                    if 'Eta' in params_df.columns:  # 5-param DL/Pseudo-Voigt
                        peak_cols = ['Amplitude', 'Center (cm)', 'Sigma (cm)', 'Gamma (cm)', 'Eta']
                        # (需要从 DL 模块导入 5-param numpy 函数)
                        from core.peak_fitting_dl_batch import pseudo_voigt_np as sf
                    else:  # 4-param SciPy Voigt
                        peak_cols = ['Amplitude', 'Center (cm)', 'Sigma (cm)', 'Gamma (cm)']
                        from core.peak_fitting_models import voigt as sf
                else:  # Gaussian (3-param)
                    peak_cols = ['Amplitude', 'Center (cm)', 'Sigma (cm)']
                    from core.peak_fitting_models import gaussian as sf

                # 遍历 DataFrame 的每一行 (每个峰)
                for i, row in params_df.iterrows():
                    p = row[peak_cols].values
                    center_val = row['Center (cm)']
                    if p[0] > 1e-6:  # 检查振幅
                        y_peak = sf(x, *p);

                        # (!!!) 使用 fill 而不是 plot (!!!)
                        self.ax_main.fill(x, y_peak, alpha=0.5, label=f'拟合峰 @ {center_val:.2f} cm')

            except ImportError as e_imp:
                print(f"Err plot single peaks (Import): {e_imp}")
                self.ax_main.text(0.5, 0.5, f'绘制组件失败: {e_imp}', transform=self.ax_main.transAxes, color='red')
            except Exception as e:
                print(f"Err plot single peaks: {e}")
                traceback.print_exc()
                self.ax_main.text(0.5, 0.5, f'绘制组件失败: {e}', transform=self.ax_main.transAxes, color='red')
            # --- ^^^^ 修改结束 ^^^^ ---

            self.ax_main.set_title(f'拟合:{title}', fontsize=12);

            # (新增) 图例限制逻辑 (来自 plot_conc_0.9_0.png)
            handles, labels = self.ax_main.get_legend_handles_labels()
            max_legend_entries = 15  # 限制图例中峰的数量
            if len(handles) > max_legend_entries:
                displayed_handles = handles[:2] + handles[2:max_legend_entries]
                displayed_labels = labels[:2] + labels[2:max_legend_entries]
                # 修改图注
                self.ax_main.legend(displayed_handles, displayed_labels, loc='best', fontsize='small').set_draggable(True)
            else:
                # 修改图注
                self.ax_main.legend(loc='best', fontsize='small').set_draggable(True);

            self.ax_main.grid(True, ls='--');

            # (不变) 绘制残差
            res = y - fit_y;
            self.ax_sub.plot(x, res, 'k-', label='残差', lw=1);
            self.ax_sub.axhline(0, c='gray', ls='--', lw=0.8);
            # 修改图注
            self.ax_sub.legend(loc='best').set_draggable(True);
            self.ax_sub.grid(True, ls='--');

            self.figure.tight_layout();
            self.canvas.draw()
        except Exception as e:
            self._handle_plot_error(e)

    def clear_plot(self):  # Simplified
        self._clear_axes();
        self.ax_sub.set_visible(False);
        self.ax_main.set_title("请导入项目...");
        self.figure.tight_layout();
        self.canvas.draw()

    def _handle_plot_error(self, e):  # Simplified
        print(f"Plot Error: {e}");
        traceback.print_exc()
        try:
            self._clear_axes(); self.ax_sub.set_visible(False); self.ax_main.set_title(
                f"绘图错误: {e}"); self.figure.tight_layout(); self.canvas.draw()
        except Exception as E:
            print(f"CRIT ERR handle plot err: {E}")