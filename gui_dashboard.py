#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
"""
import sys
import os
import re
import json
import pandas as pd
import numpy as np
from PyQt5.QtCore import QTimer, Qt, QDate, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QListWidget, QTextEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget, QMessageBox, QFrame,
    QLineEdit, QInputDialog, QSplitter, QRadioButton, QButtonGroup, QCheckBox,
    QLabel, QComboBox, QListWidget, QListWidgetItem, QDateEdit, QGridLayout, QDialog, QFileDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import subprocess
import threading
from datetime import datetime
import pytz
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
import matplotlib.cm as cm
import random
import matplotlib.colors as mcolors


def format_folder_name(folder_name):
    try:
        parts = folder_name.split("_")
        timestamp_part = parts[-2] + "_" + parts[-1]
        name_part = "_".join(parts[:-2])

        dt_utc = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
        dt_utc = pytz.utc.localize(dt_utc)
        dt_et = dt_utc.astimezone(pytz.timezone("America/New_York"))
        readable = dt_et.strftime("%b %d, %Y at %I:%M %p")
        return f"{name_part} – {readable} (ET)"
    except Exception:
        return folder_name
class GroupingDialog(QDialog):
    def __init__(self, selected_items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Groups for Selected Experiments")
        self.selected_items = selected_items
        self.group_names = []  # Start empty
        self.combos = {}

        self.layout = QVBoxLayout(self)

        # Group creation section
        input_layout = QHBoxLayout()
        self.new_group_input = QLineEdit()
        self.new_group_input.setPlaceholderText("Enter new group name...")
        add_btn = QPushButton("Add Group")
        add_btn.clicked.connect(self.add_group)
        input_layout.addWidget(self.new_group_input)
        input_layout.addWidget(add_btn)
        self.layout.addLayout(input_layout)

        # Experiment-to-group mapping rows
        for exp_name in self.selected_items:
            row = QHBoxLayout()
            label = QLabel(exp_name)
            combo = QComboBox()
            combo.setEnabled(False)  # initially disabled until at least one group exists
            row.addWidget(label)
            row.addWidget(combo)
            self.layout.addLayout(row)
            self.combos[exp_name] = combo

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        self.layout.addLayout(button_layout)

    def add_group(self):
        new_group = self.new_group_input.text().strip()
        if not new_group:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid group name.")
            return
        if new_group in self.group_names:
            QMessageBox.warning(self, "Duplicate Name", "This group already exists.")
            return
        self.group_names.append(new_group)
        # Add the new group to all combo boxes
        for combo in self.combos.values():
            combo.setEnabled(True)
            combo.addItem(new_group)
        self.new_group_input.clear()

    def get_group_assignments(self):
        result = {}
        for label, combo in self.combos.items():
            group = combo.currentText()
            if group:
                result.setdefault(group, []).append(label)
        return result        

class WickingDashboard(QMainWindow):
    def generate_single_summary(self):
        selected_items = self.exp_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Experiment Selected", "Please select an experiment.")
            return

        display_name = selected_items[0].text()
        folder_name = self.folder_display_map[display_name]
        folder = os.path.join(self.output_dir, folder_name)
        csv_path = os.path.join(folder, "data.csv")
        pdf_path = os.path.join(folder, "summary_report.pdf")

        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "Missing Data", "No data.csv found.")
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process data:\n{e}")
            return

        summary_text = self.meta_view.toPlainText().strip()
        if not summary_text:
            summary_text = "No summary available."

        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
            gs = GridSpec(2, 2, height_ratios=[5, 2], figure=fig)

            # Clean experiment name
            exp_name = re.sub(r"_\d{8}_\d{6}$", "", folder_name)

            # Safer margin + smaller font
            fig.suptitle(f"Experiment: {exp_name}", fontsize=14, fontweight='bold', y=0.96)

            # Plot 1: Height
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(df["Time_Uniform"], df["Height_Model"], label="Modeled", linewidth=2)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Height (mm)")
            ax1.set_title("Height")
            ax1.legend()
            ax1.grid(True)
            ymin, ymax = ax1.get_ylim()
            ax1.set_yticks(np.arange(0, ymax + 10, 10))

            # Plot 2: Average Wicking Rate
            # Start from 350th row for Time and Avg Wicking Rate
            # df_wicking = df.iloc[350:]
            df_wicking = df[df["Time_Uniform"] > 60]
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(df_wicking["Time_Uniform"], df_wicking["Modeled Avg Wicking Rate"], label="Wicking Rate", color="red")
            # ax2.plot(df["Time"], df["Avg Wicking Rate"], label="Wicking Rate", color="red")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Wicking Rate (mm/s)")
            ax2.set_title("Average Wicking Rate")
            ax2.legend()
            ax2.grid(True)
            ymin, ymax = ax2.get_ylim()
            ax2.set_yticks(np.arange(0, ymax + 0.05, 0.05))

            # Summary
            ax3 = fig.add_subplot(gs[1, :])
            ax3.axis("off")
            ax3.text(0.01, 0.98, "Summary", fontsize=13, fontweight="bold", va="top")
            ax3.text(0.01, 0.84, summary_text, fontsize=10, va="top", family="monospace", linespacing=1.5)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        QMessageBox.information(self, "PDF Saved", f"Summary saved to:\n{pdf_path}")

    def handle_generate_summary(self):
        selected_items = self.exp_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one experiment.")
            return

        if len(selected_items) == 1:
            self.generate_single_summary()
            return

        display_names = [item.text() for item in selected_items]
        folder_name = self.folder_display_map[display_names[0]]
        folder = os.path.join(self.output_dir, folder_name)
        dialog = GroupingDialog(display_names, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        group_map = dialog.get_group_assignments()
        if not group_map:
            QMessageBox.warning(self, "No Groups", "Please create and assign at least one group.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Summary PDF", "grouped_summary.pdf", "PDF Files (*.pdf)")
        if not save_path:
            return


        with PdfPages(save_path) as pdf:
            fig = plt.figure(figsize=(8.3, 11.7))  # A4 portrait
            gs = GridSpec(5, 1, height_ratios=[3, 0.4, 3, 0, 2], figure=fig)

            # Clean experiment name
            # exp_name = re.sub(r"_\d{8}_\d{6}$", "", folder_name)  # remove timestamp
            # exp_name = re.sub(r"\s+\S+$", "", exp_name)      


            exp_name = os.path.splitext(os.path.basename(save_path))[0]
            
            # Safer margin + smaller font
            fig.suptitle(f"Experiment: {exp_name}", fontsize=14, fontweight='bold', y=0.96)

            # Get group names from the group_map
            group_names = list(group_map.keys())
            
            # Assign a unique color to each group
            # cmap = plt.get_cmap("tab20")   # supports up to 20 distinct colors
            # group_colors = {group: cmap(i % 20) for i, group in enumerate(group_names)}

            # Assign yellow and blue to groups in order
            custom_colors = ["#FFD700", "#1E90FF"]  # gold yellow, dodger blue
            group_colors = {group: custom_colors[i % len(custom_colors)] for i, group in enumerate(group_names)}

            # Fixed colors for first two groups
            fixed_colors = ["#FFD700", "#1E90FF"]  # yellow, blue
            group_colors = {}

            # Assign fixed colors for first two groups
            for i, group in enumerate(group_names):
                if i < 2:
                    group_colors[group] = fixed_colors[i]
                else:
                    # pick a random color excluding yellow and blue
                    available_colors = list(mcolors.CSS4_COLORS.values())
                    available_colors = [c for c in available_colors if c.lower() not in ["#ffd700", "#1e90ff", "gold", "blue"]]
                    group_colors[group] = random.choice(available_colors)



            # === Plot 1: Fitted Height ===
            ax1 = fig.add_subplot(gs[0])
            for group_name, items in group_map.items():
                color = group_colors[group_name]
                for disp_name in items:
                    folder = self.folder_display_map[disp_name]
                    csv_path = os.path.join(self.output_dir, folder, "data.csv")
                    if not os.path.exists(csv_path):
                        continue
                    try:
                        df = pd.read_csv(csv_path)
                        x = df["Time_Uniform"] if "Time_Uniform" in df.columns else df["Time"]
                        y = df["Height_Model"] if "Height_Model" in df.columns else df["Height"]
                        ax1.plot(x, y, color=color, linewidth=1.5, alpha=0.9)
                    except Exception as e:
                        print(f"Plot error in {disp_name}: {e}")
            
            ax1.set_title("Fitted Height vs Time")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Height (mm)")
            ax1.grid(True)
            
            # Add legend with group names and colors
            for group_name, color in group_colors.items():
                ax1.plot([], [], label=group_name, color=color)
            ax1.legend(fontsize=7)
            
            ymin, ymax = ax1.get_ylim()
            ax1.set_yticks(np.arange(0, ymax + 10, 10))

            # === Plot 2: Wicking Rate ===
            ax2 = fig.add_subplot(gs[2])
            for group_name, items in group_map.items():
                color = group_colors[group_name]
                for disp_name in items:
                    folder = self.folder_display_map[disp_name]
                    csv_path = os.path.join(self.output_dir, folder, "data.csv")
                    if not os.path.exists(csv_path):
                        continue
                    try:
                        df = pd.read_csv(csv_path)
                        df = df[df["Time_Uniform"] > 60]  # exclude startup noise
                        if "Modeled Avg Wicking Rate" in df.columns:
                            ax2.plot(df["Time_Uniform"], df["Modeled Avg Wicking Rate"], color=color, linewidth=1.3, alpha=0.8)
                    except Exception as e:
                        print(f"Wicking plot error in {disp_name}: {e}")
            
            ax2.set_title("Average Wicking Rate vs Time")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Wicking Rate (mm/s)")
            ax2.grid(True)
            
            # Add legend with group names and colors
            for group_name, color in group_colors.items():
                ax2.plot([], [], label=group_name, color=color)
            ax2.legend(fontsize=7)
            
            ymin, ymax = ax2.get_ylim()
            ax2.set_yticks(np.arange(0, ymax + 0.05, 0.05))

            # === Summary Text Table ===
            ax3 = fig.add_subplot(gs[4])
            ax3.axis("off")

            minute_values = list(range(1, 11))
            height_table = {}   # only store groups with data
            rate_table = {}

            for group_name, items in group_map.items():
                height_rows, rate_rows = [], []
                for disp_name in items:
                    folder = self.folder_display_map[disp_name]
                    csv_path = os.path.join(self.output_dir, folder, "data.csv")
                    if not os.path.exists(csv_path):
                        continue
                    try:
                        df = pd.read_csv(csv_path)
                        # Require needed columns
                        if "Time_Uniform" not in df.columns or "Height_Model" not in df.columns:
                            continue

                        # Build one row per minute, but only if that minute exists in this file
                        h_row, r_row = [], []
                        for t in (60 * i for i in minute_values):
                            if df["Time_Uniform"].max() < t:
                                # not enough coverage; mark as NaN for this file at this minute
                                h_row.append(np.nan)
                                r_row.append(np.nan)
                                continue
                            closest_idx = (df["Time_Uniform"] - t).abs().idxmin()
                            h = float(df.loc[closest_idx, "Height_Model"])
                            if "Modeled Avg Wicking Rate" in df.columns:
                                r = float(df.loc[closest_idx, "Modeled Avg Wicking Rate"])
                            else:
                                r = h / t if t > 0 else np.nan
                            h_row.append(h)
                            r_row.append(r)

                        height_rows.append(h_row)
                        rate_rows.append(r_row)
                    except Exception:
                        continue

                # keep only groups with at least one valid entry
                if height_rows:
                    height_table[group_name] = np.nanmean(height_rows, axis=0)
                    rate_table[group_name]   = np.nanmean(rate_rows, axis=0)

            # ✅ Render only groups that actually have data
            valid_groups = [g for g in group_names if g in height_table]

            if not valid_groups:
                ax3.text(0.0, 0.5, "No valid data to summarize.", fontsize=10, va="center")
            else:
                # Layout positioning with proper spacing
                x_left = -0.05
                x_right = 0.45
                y_start = 0.98
                title_spacing = 0.08
                header_spacing = 0.06
                line_spacing = 0.07

                # Column width based on number of valid groups
                max_groups = max(len(valid_groups), 3)  # minimum 3 for formatting
                col_width = min(12, max(8, 40 // max_groups))

                # Titles
                height_title = "Group Summary – Avg Height (mm)"
                rate_title   = "Group Summary – Avg Wicking Rate (mm/s)"
                ax3.text(x_left, y_start, height_title, fontsize=11, fontweight="bold",
                        family="monospace", va="top")
                ax3.text(x_right, y_start, rate_title, fontsize=11, fontweight="bold",
                        family="monospace", va="top")

                # Headers
                height_header = f"{'Time (min)':<12}" + "".join(f"{g[:col_width]:>{col_width}}" for g in valid_groups)
                rate_header   = f"{'Time (min)':<12}" + "".join(f"{g[:col_width]:>{col_width}}" for g in valid_groups)
                ax3.text(x_left,  y_start - title_spacing, height_header, fontsize=9, family="monospace", va="top")
                ax3.text(x_right, y_start - title_spacing, rate_header,   fontsize=9, family="monospace", va="top")

                # Data rows - ONLY ONE SECTION HERE
                for i, min_val in enumerate(minute_values):
                    y_pos = y_start - title_spacing - header_spacing - (i * line_spacing)

                    def fmt(val, width, prec):
                        return f"{val:>{width}.{prec}f}" if np.isfinite(val) else f"{'':>{width}}"

                    h_values = "".join(fmt(height_table[g][i], col_width, 2) for g in valid_groups)
                    r_values = "".join(fmt(rate_table[g][i],   col_width, 4) for g in valid_groups)

                    ax3.text(x_left,  y_pos, f"{min_val:<12}" + h_values, fontsize=8.5, family="monospace", va="top")
                    ax3.text(x_right, y_pos, f"{min_val:<12}" + r_values, fontsize=8.5, family="monospace", va="top")

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        QMessageBox.information(self, "Saved", f"Summary saved to:\n{save_path}")



    def __init__(self, output_dir):
        super().__init__()
        self.setWindowTitle("Wicking Tracker Dashboard")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f2f5;
                font-family: Segoe UI, sans-serif;
                font-size: 14px;
            }

            QFrame, #MainCard, #PlotPanel, #MetaPanel {
                background-color: #f9fafe;
                border: 1px solid #ffffff;
                border-radius: 10px;
            }

            QListWidget, QTextEdit {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: white;
            }
                           
        """)

        self.output_dir = output_dir
        self.plot_mode = "height"

        self.sidebar_frame = QFrame()
        self.sidebar_frame.setStyleSheet("background-color: #f9fafe;")
        self.sidebar_frame.setFixedWidth(200)

        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedSize(30, 30)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                font-size: 18px;
                margin: 5px;
            }
            QPushButton:hover {
                color: #3498db;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        self.btn_start = QPushButton("▶ Run Experiment")
        self.btn_view = QPushButton("@View Experiments")
        for btn in [self.btn_start, self.btn_view]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                    border-radius: 6px;
                    padding: 10px;
                    margin: 10px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            btn.setFixedWidth(180)
        
        self.btn_start.clicked.connect(self.show_start_view)
        self.btn_view.clicked.connect(self.show_experiment_view)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 20, 10, 10)
        sidebar_layout.setSpacing(10)
        sidebar_layout.addWidget(self.btn_start)
        sidebar_layout.addWidget(self.btn_view)
        sidebar_layout.addStretch()
        self.sidebar_frame.setLayout(sidebar_layout)


        self.status_label = QLabel("🟢 Idle")
        self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        
        self.stack = QStackedWidget()
        self.init_start_view()
        self.init_experiment_view()

        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.sidebar_frame)
        self.main_layout.addWidget(self.stack)

        self.header_bar = QHBoxLayout()

        self.header_bar.addWidget(self.toggle_btn)
        self.header_bar.addStretch()


        main_container = QVBoxLayout()
        main_container.addLayout(self.header_bar)
        main_container.addLayout(self.main_layout)

        container = QWidget()
        container.setLayout(main_container)
        self.setCentralWidget(container)

    @pyqtSlot(str, str)
    def update_status(self, message, color="green"):
        print(f"[UPDATE_STATUS] {message} ({color})")
        icons = {
            "green": "🟢",
            "blue": "⚙️",
            "red": "⛔",
            "orange": "📸",
            "gray": "⏹️",
        }
        icon = icons.get(color, "ℹ️")
        self.status_label.setText(f"{icon} {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def extract_summary_from_df(self, df):
        """Extract summary data for live display after experiment completion"""
        summary_lines = []
        minutes = list(range(1, 11))  # 1 to 10 minutes

        if "Time_Uniform" not in df.columns or "Height_Model" not in df.columns:
            return "Missing required columns: 'Time_Uniform' and/or 'Height_Model'"

        for min_val in minutes:
            target_time = min_val * 60
            
            # Check if we have data for this time point
            if df["Time_Uniform"].max() < target_time:
                summary_lines.append(f"{min_val} min height: Not Available | Avg Rate: Not Available")
                continue
                
            try:
                closest_idx = (df["Time_Uniform"] - target_time).abs().idxmin()
                closest_time = df.loc[closest_idx, "Time_Uniform"]
                closest_height = df.loc[closest_idx, "Height_Model"]

                # Use the modeled avg wicking rate if available, otherwise calculate
                if "Modeled Avg Wicking Rate" in df.columns:
                    avg_rate = df.loc[closest_idx, "Modeled Avg Wicking Rate"]
                else:
                    avg_rate = closest_height / closest_time if closest_time > 0 else 0

                summary_lines.append(
                    f"{min_val} min height: {closest_height:.2f} mm | Avg Rate: {avg_rate:.4f} mm/s"
                )
            except Exception as e:
                print(f"Error extracting data for {min_val} min: {e}")
                summary_lines.append(
                    f"{min_val} min height: Not Available | Avg Rate: Not Available"
                )

        return "\n".join(summary_lines)


    def toggle_sidebar(self):
        self.sidebar_frame.setVisible(not self.sidebar_frame.isVisible())

    def init_start_view(self):
        start_widget = QWidget()
        outer_layout = QVBoxLayout()

        # --- Add Title ---
        heading_label = QLabel("Sweat Management")
        heading_label.setAlignment(Qt.AlignCenter)
        heading_label.setStyleSheet("""
            font-size: 36px;
            font-weight: 600;
            letter-spacing: 1px;
            color: #2c3e50;
            margin-top: 10px;
        """)
        outer_layout.addWidget(heading_label)

        # --- Button + Status Row ---
        button_row = QHBoxLayout()
        self.start_button = QPushButton("▶ Start Wicking")
        self.start_button.setFixedWidth(200)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.start_button.clicked.connect(self.handle_start_wicking)

        self.status_label = QLabel("🟢 Idle")
        self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold; padding-right: 10px;")

        button_row.addWidget(self.start_button, alignment=Qt.AlignLeft)
        button_row.addStretch()
        button_row.addWidget(self.status_label, alignment=Qt.AlignRight)
        outer_layout.addLayout(button_row)

        # === Live Statistics Grid ===
        self.stat_labels = {}
        stat_grid = QGridLayout()
        stat_grid.setVerticalSpacing(8)
        for i, label_text in enumerate(["Time", "Delta E", "Delta Threshold", "Height", "Wicking Rate"]):
            label = QLabel(f"{label_text}:")
            value = QLabel("...")
            value.setStyleSheet("font-weight: bold; color: #34495e;")
            self.stat_labels[label_text] = value
            stat_grid.addWidget(label, i, 0)
            stat_grid.addWidget(value, i, 1)

        stat_widget = QWidget()
        stat_widget.setLayout(stat_grid)

        # === Live Output Text Box (with titled frame) ===
        self.live_output_box = QTextEdit()
        self.live_output_box.setReadOnly(True)
        self.live_output_box.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #2c3e50;
            }
        """)
        self.live_output_box.setPlaceholderText("📊 Summary will appear here after the experiment...")
        self.live_output_box.setFixedHeight(160)

        # Frame to hold the summary title and box
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet("""
            QFrame {
                background-color: #f9fafe;
                border: 1px solid #dcdcdc;
                border-radius: 12px;
                padding: 10px;
            }
        """)

        summary_layout = QVBoxLayout()
        summary_title = QLabel("📊 Summary")
        summary_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #34495e;
                margin-bottom: 4px;
            }
        """)
        summary_layout.addWidget(summary_title)
        summary_layout.addWidget(self.live_output_box)
        self.summary_frame.setLayout(summary_layout)

        # === Put Stats + Output in Split Layout ===
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(stat_widget, stretch=1)
        bottom_row.addWidget(self.summary_frame, stretch=2)
        outer_layout.addLayout(bottom_row)

        outer_layout.addStretch()

        start_widget.setLayout(outer_layout)
        self.stack.addWidget(start_widget)
        self.show_start_view()

    def init_experiment_view(self):
        view_widget = QWidget()
        outer_layout = QVBoxLayout()

        # === LEFT PANEL ===
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Search experiments...")
        self.search_bar.textChanged.connect(self.apply_filters)

        self.exp_list = QListWidget()
        self.exp_list.setSelectionMode(QListWidget.SingleSelection)
        self.exp_list.itemClicked.connect(self.display_experiment_data)
        self.exp_list.itemClicked.connect(self.plot_selected_experiments)


        self.multi_select_checkbox = QCheckBox("Enable multi-select")
        self.multi_select_checkbox.stateChanged.connect(self.toggle_selection_mode)

        self.plot_selected_button = QPushButton("📈 Plot Selected")
        self.plot_selected_button.clicked.connect(self.plot_selected_experiments)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.search_bar)
        left_panel.addWidget(self.exp_list)
        left_panel.addWidget(self.multi_select_checkbox)
        left_panel.addWidget(self.plot_selected_button)

        left_widget = QWidget()
        left_widget.setObjectName("LeftPanel")
        left_widget.setLayout(left_panel)

        # === CENTER PLOT PANEL ===
        # Primary plot type radios
        self.height_radio = QRadioButton("Height")
        self.wicking_radio = QRadioButton("Wicking Rate")
        self.height_radio.setChecked(True)

        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.height_radio)
        self.radio_group.addButton(self.wicking_radio)

        self.height_radio.toggled.connect(self.on_radio_change)
        self.wicking_radio.toggled.connect(self.on_radio_change)

        # === Secondary radios for Height
        self.height_raw_radio = QRadioButton("Raw Only")
        self.height_fitted_radio = QRadioButton("Fitted Only")
        self.height_raw_and_fitted_radio = QRadioButton("Raw + Fitted")
        self.height_fitted_radio.setChecked(True)


        self.height_mode_group = QButtonGroup()
        self.height_mode_group.addButton(self.height_raw_radio)
        self.height_mode_group.addButton(self.height_fitted_radio)
        self.height_mode_group.addButton(self.height_raw_and_fitted_radio)

        self.height_mode_widget = QWidget()
        height_mode_layout = QHBoxLayout()
        height_mode_layout.setContentsMargins(0, 0, 0, 0)
        height_mode_layout.addWidget(self.height_raw_radio)
        height_mode_layout.addWidget(self.height_fitted_radio)
        height_mode_layout.addWidget(self.height_raw_and_fitted_radio)
        self.height_mode_widget.setLayout(height_mode_layout)

        # === Secondary radios for Wicking
        self.model_rate_radio = QRadioButton("Model-Based Rate")
        self.avg_rate_radio = QRadioButton("Avg Wicking Rate")
        self.model_rate_radio.setChecked(True)

        self.height_raw_radio.toggled.connect(self.plot_selected_experiments)
        self.height_fitted_radio.toggled.connect(self.plot_selected_experiments)
        self.height_raw_and_fitted_radio.toggled.connect(self.plot_selected_experiments)
        self.model_rate_radio.toggled.connect(self.plot_selected_experiments)
        self.avg_rate_radio.toggled.connect(self.plot_selected_experiments)


        self.wicking_mode_group = QButtonGroup()
        self.wicking_mode_group.addButton(self.model_rate_radio)
        self.wicking_mode_group.addButton(self.avg_rate_radio)

        self.wicking_mode_widget = QWidget()
        wicking_mode_layout = QHBoxLayout()
        wicking_mode_layout.setContentsMargins(0, 0, 0, 0)
        wicking_mode_layout.addWidget(self.model_rate_radio)
        wicking_mode_layout.addWidget(self.avg_rate_radio)
        self.wicking_mode_widget.setLayout(wicking_mode_layout)

        # Top row: height/wicking
        row1_layout = QHBoxLayout()
        row1_layout.addWidget(self.height_radio)
        row1_layout.addWidget(self.wicking_radio)

        # Create the stacked widget to switch between the two mode options
        self.dynamic_mode_container = QStackedWidget()
        self.dynamic_mode_container.addWidget(self.height_mode_widget)   # index 0
        self.dynamic_mode_container.addWidget(self.wicking_mode_widget)  # index 1

        # Wrap it in a container widget (optional but sometimes useful for layout)
        self.dynamic_mode_wrapper = QWidget()
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(self.dynamic_mode_container)
        self.dynamic_mode_wrapper.setLayout(wrapper_layout)

        # Add to final radio layout
        radio_layout = QVBoxLayout()
        radio_layout.addLayout(row1_layout)
        radio_layout.addWidget(self.dynamic_mode_wrapper)

        self.plot_area = FigureCanvas(Figure(figsize=(5, 4)))
        self.ax = self.plot_area.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.plot_area, self)

        plot_container = QVBoxLayout()
        plot_container.addWidget(self.toolbar, stretch=0)
        plot_container.addLayout(radio_layout, stretch=0)
        plot_container.addWidget(self.plot_area, stretch=1)

        plot_widget = QWidget()
        plot_widget.setObjectName("PlotPanel")
        plot_widget.setLayout(plot_container)

        # === RIGHT PANEL (Metadata) ===
        self.meta_view = QTextEdit()
        self.meta_view.setReadOnly(True)

        self.type_filter_label = QLabel("Filter by Experiment Type:")
        self.type_filter_dropdown = QComboBox()
        self.type_filter_dropdown.currentTextChanged.connect(self.apply_filters)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.type_filter_label)
        right_panel.addWidget(self.type_filter_dropdown)
        right_panel.addWidget(self.meta_view)

        #Button to save summary 
        self.save_pdf_button = QPushButton("Generate Summary")
        self.save_pdf_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px;
                margin-top: 6px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.save_pdf_button.clicked.connect(self.handle_generate_summary)
        right_panel.addWidget(self.save_pdf_button)

        right_panel.addStretch()

        right_widget = QWidget()
        right_widget.setObjectName("RightPanel")
        right_widget.setLayout(right_panel)

        # === COMBINE PANELS ===
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(plot_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([250, 600, 350])

        outer_layout.addWidget(splitter)
        view_widget.setLayout(outer_layout)
        self.stack.addWidget(view_widget)
        self.refresh_experiment_list()
        self.on_radio_change()

    def on_radio_change(self):
        if self.height_radio.isChecked():
            self.plot_mode = "height"
            self.dynamic_mode_container.setCurrentIndex(0)

        elif self.wicking_radio.isChecked():
            self.plot_mode = "wicking"
            self.dynamic_mode_container.setCurrentIndex(1)

        else:
            return

        self.plot_selected_experiments()


    def toggle_selection_mode(self, state):
        mode = QListWidget.MultiSelection if state == Qt.Checked else QListWidget.SingleSelection
        self.exp_list.setSelectionMode(mode)

    def show_start_view(self):
        self.stack.setCurrentIndex(0)

    def show_experiment_view(self):
        self.stack.setCurrentIndex(1)
        self.refresh_experiment_list()

    def get_latest_experiment_folder(self):
        folders = [f for f in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, f))]
        latest = max(folders, key=lambda x: os.path.getctime(os.path.join(self.output_dir, x)))
        return latest

    def generate_temp_dataframe(self, t_data, h_data):
        def model_f(t, H, tau, A):
            return H * (1 - np.exp(-t / tau)) + A * np.sqrt(t)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                popt, _ = curve_fit(model_f, t_data, h_data, p0=[31, 9, 4], maxfev=5000)
            except Exception as e:
                print(f"Curve fit failed: {e}")
                popt = [1, 1, 1]  # fallback values

        t_uniform = np.linspace(0, max(t_data), len(t_data))
        h_model = model_f(t_uniform, *popt)

        df = pd.DataFrame({
            "Time_Uniform": t_uniform,
            "Height_Model": h_model,
            "Modeled Avg Wicking Rate": h_model / np.maximum(t_uniform, 1)
        })
        return df


    def handle_start_wicking(self):
        def run_main_py():
            full_output_log = ""
            QMetaObject.invokeMethod(
                self, "update_status", Qt.QueuedConnection,
                Q_ARG(str, "Starting Wicking Tracker"),
                Q_ARG(str, "blue")
            )

            try:
                proc = subprocess.Popen(
                    [sys.executable, "-u", "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                for line in iter(proc.stdout.readline, ""):
                    line = line.strip()
                    full_output_log += line + "\n"

                    if line.startswith("STATUS:"):
                        message = line.split("STATUS:")[1].strip()
                        QMetaObject.invokeMethod(
                            self, "update_status", Qt.QueuedConnection,
                            Q_ARG(str, message),
                            Q_ARG(str, "blue")
                        )

                    elif "Time:" in line and "Height:" in line:
                        try:
                            time_val = line.split("Time:")[1].split("s")[0].strip()
                            delta_e = line.split("Delta E:")[1].split("|")[0].strip()
                            height = line.split("Height:")[1].split("mm")[0].strip()
                            threshold = line.split("Delta Threshold:")[1].split("|")[0].strip()
                            rate = line.split("Wicking Rate:")[1].split("mm")[0].strip()

                            QMetaObject.invokeMethod(self.stat_labels["Time"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{time_val} s"))
                            QMetaObject.invokeMethod(self.stat_labels["Delta E"], "setText", Qt.QueuedConnection, Q_ARG(str, delta_e))
                            QMetaObject.invokeMethod(self.stat_labels["Height"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{height} mm"))
                            QMetaObject.invokeMethod(self.stat_labels["Delta Threshold"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{threshold}"))
                            QMetaObject.invokeMethod(self.stat_labels["Wicking Rate"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{rate} mm/s"))
                        except Exception as e:
                            print("Error parsing live stats:", e)

                proc.stdout.close()
                proc.wait()

                try:
                    # === Parse modeled data from log ===
                    modeled_data = []
                    for line in full_output_log.splitlines():
                        if line.startswith("Time:") and "Height:" in line:
                            try:
                                time_val = float(line.split("Time:")[1].split("s")[0].strip())
                                height_val = float(line.split("Height:")[1].split("mm")[0].strip())
                                modeled_data.append((time_val, height_val))
                            except:
                                continue

                    if not modeled_data:
                        raise ValueError("No modeled data could be extracted.")

                    modeled_data = np.array(modeled_data)
                    time_vals = modeled_data[:, 0]
                    height_vals = modeled_data[:, 1]

                    # === Generate DataFrame using model ===
                    df = self.generate_temp_dataframe(time_vals, height_vals)

                    # === Generate and display summary ===
                    summary = self.extract_summary_from_df(df)
                    QMetaObject.invokeMethod(
                        self.live_output_box,
                        "clear",  # Clear previous content first
                        Qt.QueuedConnection
                    )
                    QMetaObject.invokeMethod(
                        self.live_output_box,
                        "append",
                        Qt.QueuedConnection,
                        Q_ARG(str, "EXPERIMENT SUMMARY\n" + "="*50 + "\n" + summary)
                    )

                except Exception as e:
                    QMetaObject.invokeMethod(
                        self.live_output_box,
                        "append",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"\nSUMMARY ERROR:\n{e}")
                    )

                QMetaObject.invokeMethod(
                    self, "update_status", Qt.QueuedConnection,
                    Q_ARG(str, "Experiment complete"),
                    Q_ARG(str, "green")
                )
                QTimer.singleShot(0, lambda: self.show_message(self, "Done", "Wicking tracking completed."))

            except subprocess.CalledProcessError:
                QMetaObject.invokeMethod(
                    self, "update_status", Qt.QueuedConnection,
                    Q_ARG(str, "Error running experiment"),
                    Q_ARG(str, "red")
                )
                QTimer.singleShot(0, lambda: self.show_message(self, "Error", "main.py failed to run."))

        # Start the experiment subprocess thread
        threading.Thread(target=run_main_py, daemon=True).start()



    def refresh_experiment_list(self):
        self.exp_info = []
        if not os.path.exists(self.output_dir):
            return

        for folder in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, folder)
            if os.path.isdir(path):
                try:
                    parts = folder.rsplit("_", 2)
                    if len(parts) != 3:
                        continue
                    name_tokens = parts[0].split()
                    exp_type = " ".join(name_tokens[:2]).title() if len(name_tokens) >= 2 else parts[0]
                    dt = datetime.strptime(parts[1] + "_" + parts[2], "%Y%m%d_%H%M%S")
                    self.exp_info.append({"folder": folder, "type": exp_type, "datetime": dt})
                except:
                    continue
        
        types = sorted(set(info["type"] for info in self.exp_info))
        self.type_filter_dropdown.blockSignals(True)
        self.type_filter_dropdown.clear()
        self.type_filter_dropdown.addItem("All Types")
        self.type_filter_dropdown.addItems(types)
        self.type_filter_dropdown.blockSignals(False)

        self.apply_filters()

    def apply_filters(self):
        search_text = self.search_bar.text().lower()
        selected_type = self.type_filter_dropdown.currentText()

        filtered = [
            info for info in self.exp_info
            if (selected_type.lower() == "all types" or info["type"].lower() == selected_type.lower())
            and (search_text in info["folder"].lower())
        ]

        filtered.sort(key=lambda x: x["datetime"], reverse=True)
        self.exp_list.clear()
        self.folder_display_map = {}
        for info in filtered:
            display_name = format_folder_name(info["folder"])
            self.folder_display_map[display_name] = info["folder"]
            self.exp_list.addItem(display_name)

    def display_experiment_data(self, item):
        folder_name = self.folder_display_map[item.text()]
        folder = os.path.join(self.output_dir, folder_name)
        csv_path = os.path.join(folder, "data.csv")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                summary_lines = []

                # Make sure necessary columns exist
                if "Time_Uniform" not in df.columns or "Height_Model" not in df.columns:
                    self.meta_view.setText("Missing required columns: 'Time_Uniform' or 'Height_Model'")
                    return

                for min_val in range(1, 11):
                    target_time = min_val * 60;

                    if df["Time_Uniform"].max() < target_time:
                        continue  # skip if time range is insufficient

                    closest_idx = (df["Time_Uniform"] - target_time).abs().idxmin()
                    time_val = df.loc[closest_idx, "Time_Uniform"]
                    height_val = df.loc[closest_idx, "Height_Model"];

                    # Calculate average wicking rate properly
                    if time_val > 0:
                        avg_rate = height_val / time_val
                    else:
                        avg_rate = 0

                    summary_lines.append(
                        f"{min_val} min height: {height_val:.2f} mm | Avg Rate: {avg_rate:.4f} mm/s"
                    )

                self.meta_view.setText("\n".join(summary_lines) if summary_lines else "No summary data available.")

            except Exception as e:
                self.meta_view.setText(f"Failed to parse data.csv: {e}")
        else:
            self.meta_view.setText("No data.csv found.")

    
    #methods

    def plot_selected_experiments(self):
        if not hasattr(self, "ax"):
            return  # Skip if ax isn't ready
        self.ax.clear()

        def model_f(t, H, tau, A):
            return H * (1 - np.exp(-t / tau)) + A * np.sqrt(t)

        def wicking_rate(t, H, tau, A):
            np.seterr(divide='ignore', invalid='ignore')
            return H / tau * np.exp(-t / tau) + A / (2 * np.sqrt(t))

        # Determine the selected plot mode
        if self.height_radio.isChecked():
            self.plot_mode = "height"
        elif self.wicking_radio.isChecked():
            self.plot_mode = "wicking"
        else:
            return

        selected_items = self.exp_list.selectedItems()
        if not selected_items:
            self.ax.text(0.5, 0.5, "No experiments selected", transform=self.ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
            self.plot_area.draw()
            return

        for item in selected_items:
            display_name = item.text()
            folder_name = self.folder_display_map[display_name]
            folder = os.path.join(self.output_dir, folder_name)
            clean_label = re.sub(r"_\d{8}_\d{6}$", "", folder_name)
            csv_path = os.path.join(folder, "data.csv")
            if not os.path.exists(csv_path):
                continue

            try:
                df = pd.read_csv(csv_path)
                t_data = df["Time"].to_numpy()
                h_data = df["Height"].to_numpy()

                # Fit model to height data
                popt, _ = curve_fit(model_f, t_data, h_data, p0=[31, 9, 4], maxfev=5000)
                H_opt, tau_opt, A_opt = popt
                # t_model = np.linspace(min(t_data), max(t_data), 100)
                t_model = np.linspace(0, max(t_data), len(t_data))

                if self.plot_mode == "height":
                    is_raw_only = self.height_raw_radio.isChecked()
                    is_fitted_only = self.height_fitted_radio.isChecked()
                    is_both = self.height_raw_and_fitted_radio.isChecked()

                    h_model = model_f(t_model, H_opt, tau_opt, A_opt)

                    if is_raw_only or is_both:
                        self.ax.plot(t_data, h_data, color='orange', label=f"{clean_label} (raw)", linewidth=1.5, alpha=0.8)
                    
                    # Plot model data if "Fitted Only" or "Raw + Fitted" is selected
                    if is_fitted_only or is_both:
                        self.ax.plot(t_model, h_model, color='#1f77b4', label=f"{clean_label} (model)", linewidth=1.5, alpha=0.9)

                elif self.plot_mode == "wicking":
                    use_avg = self.avg_rate_radio.isChecked()

                    if use_avg and "Modeled Avg Wicking Rate" in df.columns:
                        df_filtered = df[df["Time_Uniform"] > 20]
                        self.ax.plot(df_filtered["Time_Uniform"], df_filtered["Modeled Avg Wicking Rate"], label=f"{clean_label}")
                    else:
                        h_rate_model = wicking_rate(t_model, H_opt, tau_opt, A_opt)
                        self.ax.plot(t_model, h_rate_model, label=f"{clean_label}")


            except Exception as e:
                print(f"Error processing {clean_label}: {e}")

        # Set plot title and labels
        if self.plot_mode == "height":
            self.ax.set_title("Fitted Height vs Time")
            self.ax.set_ylabel("Height (mm)")
        elif self.plot_mode == "wicking":
            self.ax.set_title("Wicking Rate vs Time (Fitted Model)")
            self.ax.set_ylabel("Wicking Rate (mm/s)")

        self.ax.set_xlabel("Time (s)")

        # Add custom y-ticks only for wicking rate plot
        # Custom Y-ticks
        y_min, y_max = self.ax.get_ylim()
        if self.plot_mode == "wicking":
            if use_avg and "Modeled Avg Wicking Rate" in df.columns:
                self.ax.set_yticks(np.arange(0, y_max + 0.1, 0.1))  # every 0.1 mm/s
            else:
                self.ax.set_yticks(np.arange(0, y_max + 0.5, 0.5))  # every 0.5 mm/s

        elif self.plot_mode == "height":
            self.ax.set_yticks(np.arange(0, y_max + 10, 10))    # every 10 mm

        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.plot_area.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    output_path = os.path.expanduser("output")
    window = WickingDashboard(output_path)
    window.show()
    sys.exit(app.exec_())