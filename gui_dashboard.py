#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper and Shivam Ghodke
"""
import sys
import os
import json
import pandas as pd
from PyQt5.QtCore import QTimer, Qt, QDate, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QListWidget, QTextEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget, QMessageBox, QFrame,
    QLineEdit, QInputDialog, QSplitter, QRadioButton, QButtonGroup, QCheckBox,
    QLabel, QComboBox, QListWidget, QListWidgetItem, QDateEdit, QGridLayout
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import subprocess
import threading
from datetime import datetime
import pytz


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

class WickingDashboard(QMainWindow):

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

    def extract_summary_from_output(self, full_text):
        summary_heights = {}
        summary_rates = {}
        minutes = [1, 5, 10]

        for min_val in minutes:
            target_time = min_val * 60
            closest_height = None
            closest_rate = None
            min_diff = float("inf")

            for line in full_text.splitlines():
                if line.startswith("Time:") and "Height:" in line and "Wicking Rate:" in line:
                    try:
                        time_str = line.split("Time:")[1].split("s")[0].strip()
                        height_str = line.split("Height:")[1].split("mm")[0].strip()
                        rate_str = line.split("Wicking Rate:")[1].split("mm")[0].strip()

                        time_val = float(time_str)
                        height_val = float(height_str)
                        rate_val = float(rate_str)

                        if time_val >= target_time and abs(time_val - target_time) < min_diff:
                            min_diff = abs(time_val - target_time)
                            closest_height = height_val
                            closest_rate = rate_val
                    except:
                        continue

            summary_heights[min_val] = f"{closest_height:.2f}" if closest_height is not None else "Not Available"
            summary_rates[min_val] = f"{closest_rate:.2f}" if closest_rate is not None else "Not Available"

        summary_lines = [
            f"{min_val} min height: {summary_heights[min_val]} mm | Wicking Rate: {summary_rates[min_val]} mm/s"
            for min_val in minutes
        ]
        return "\n".join(summary_lines)

    def toggle_sidebar(self):
        self.sidebar_frame.setVisible(not self.sidebar_frame.isVisible())

    def init_start_view(self):
        start_widget = QWidget()
        outer_layout = QVBoxLayout()

        # --- Add Title ---
        heading_label = QLabel("Welcome to Wicking Tracker")
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
        self.height_radio = QRadioButton("Height")
        self.wicking_radio = QRadioButton("Wicking Rate")

        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.height_radio)
        self.radio_group.addButton(self.wicking_radio)
        self.height_radio.toggled.connect(self.on_radio_change)
        self.wicking_radio.toggled.connect(self.on_radio_change)
        self.height_radio.setChecked(True)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.height_radio)
        radio_layout.addWidget(self.wicking_radio)


        self.plot_area = FigureCanvas(Figure(figsize=(5, 4)))
        self.ax = self.plot_area.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.plot_area, self)

        plot_container = QVBoxLayout()
        plot_container.addWidget(self.toolbar)
        plot_container.addLayout(radio_layout)
        plot_container.addWidget(self.plot_area)

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

    def on_radio_change(self):
        self.plot_mode = "height" if self.height_radio.isChecked() else "wicking"
        self.plot_selected_experiments()

    def toggle_selection_mode(self, state):
        mode = QListWidget.MultiSelection if state == Qt.Checked else QListWidget.SingleSelection
        self.exp_list.setSelectionMode(mode)

    def show_start_view(self):
        self.stack.setCurrentIndex(0)

    def show_experiment_view(self):
        self.stack.setCurrentIndex(1)
        self.refresh_experiment_list()

    def handle_start_wicking(self):
        def run_main_py():
            full_output_log = ""
            # Immediately notify status before launching subprocess
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
                        # parse live values
                        try:
                            time_val = line.split("Time:")[1].split("s")[0].strip()
                            delta_e = line.split("Delta E:")[1].split("|")[0].strip()
                            height = line.split("Height:")[1].split("mm")[0].strip()
                            threshold = line.split("Delta Threshold:")[1].split("mm")[0].strip()
                            rate = line.split("Wicking Rate:")[1].split("mm")[0].strip()
                            

                            QMetaObject.invokeMethod(self.stat_labels["Time"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{time_val} s"))
                            QMetaObject.invokeMethod(self.stat_labels["Delta E"], "setText", Qt.QueuedConnection, Q_ARG(str, delta_e))
                            QMetaObject.invokeMethod(self.stat_labels["Height"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{height} mm"))
                            QMetaObject.invokeMethod(self.stat_labels["Delta Threshold"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{threshold} mm"))
                            QMetaObject.invokeMethod(self.stat_labels["Wicking Rate"], "setText", Qt.QueuedConnection, Q_ARG(str, f"{rate} mm/s"))
                        except Exception as e:
                            print("Error parsing time line:", e)

                proc.stdout.close()
                proc.wait()

                summary = self.extract_summary_from_output(full_output_log)
                QMetaObject.invokeMethod(
                    self.live_output_box,
                    "append",
                    Qt.QueuedConnection,
                    Q_ARG(str, "\nSUMMARY\n" + summary)
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
        json_path = os.path.join(folder, "metadata.json")

        # Only call this:
        self.plot_selected_experiments()

        # No need to replot again manually here
        # Just draw the metadata section:
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                self.meta_view.setText(json.dumps(meta, indent=2))
            except Exception as e:
                self.meta_view.setText(f"Failed to load metadata: {e}")
        else:
            self.meta_view.setText("No metadata found.")

    def plot_selected_experiments(self):
        if not hasattr(self, "ax"):
            return  # Skip if ax isn't ready
        self.ax.clear()
        self.plot_mode = "height" if self.height_radio.isChecked() else "wicking"
        selected_items = self.exp_list.selectedItems()
        if not selected_items:
            self.ax.text(0.5, 0.5, "No experiments selected", transform=self.ax.transAxes,
                         ha='center', va='center', fontsize=12, color='gray')
            self.plot_area.draw()
            self.plot_area.repaint()   # 👈 Force repaint
            self.plot_area.update()  
            return

        for item in selected_items:
            display_name = item.text()
            folder_name = self.folder_display_map[display_name]
            folder = os.path.join(self.output_dir, folder_name)
            csv_path = os.path.join(folder, "data.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if self.plot_mode == "height":
                        self.ax.plot(df["Time_Uniform"], df["Filtered Height (Raw)"], label=folder_name)
                    elif self.plot_mode == "wicking":
                        self.ax.plot(df["Time_Uniform"], df["Wicking Rate Filtered (Spline)"], label=folder_name)
                except Exception as e:
                    print(f"Error reading {folder_name}: {e}")

        self.ax.set_title("Height vs Time" if self.plot_mode == "height" else "Wicking Rate")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Height (mm)" if self.plot_mode == "height" else "Wicking Rate (mm/s)")
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.plot_area.draw()
        self.plot_area.repaint()   # 👈 Force repaint
        self.plot_area.update()  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    output_path = os.path.expanduser("output")
    window = WickingDashboard(output_path)
    window.show()
    sys.exit(app.exec_())
