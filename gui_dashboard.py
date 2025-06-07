import sys
import os
import json
import pandas as pd
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import subprocess
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QListWidget, QTextEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget, QMessageBox, QFrame, QLineEdit, QInputDialog, QSplitter
)
from PyQt5.QtCore import Qt


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
            QListWidget, QTextEdit {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: white;
            }
        """)

        self.output_dir = output_dir

        # === Sidebar ===
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setStyleSheet("background-color: #ffffff;")
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

        self.btn_start = QPushButton("▶ Start Wicking")
        self.btn_view = QPushButton("📂 View Experiments")

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

        # === Stack Views (Start + Experiments) ===
        self.stack = QStackedWidget()
        self.init_start_view()
        self.init_experiment_view()

        # === Layout for Sidebar + Main Stack ===
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.sidebar_frame)
        self.main_layout.addWidget(self.stack)

        # === Top Header Bar with Toggle Button ===
        self.header_bar = QHBoxLayout()
        self.header_bar.addWidget(self.toggle_btn)
        self.header_bar.addStretch()

        # === Final Layout ===
        main_container = QVBoxLayout()
        main_container.addLayout(self.header_bar)
        main_container.addLayout(self.main_layout)

        container = QWidget()
        container.setLayout(main_container)
        self.setCentralWidget(container)

        self.show_start_view()


    def toggle_sidebar(self):
        if self.sidebar_frame.isVisible():
            self.sidebar_frame.hide()
        else:
            self.sidebar_frame.show()

    def init_start_view(self):
        start_widget = QWidget()
        layout = QVBoxLayout()

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

        layout.addStretch()
        layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        start_widget.setLayout(layout)

        self.stack.addWidget(start_widget)

    def init_experiment_view(self):
        view_widget = QWidget()
        outer_layout = QVBoxLayout()
        # inner_layout = QHBoxLayout()

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Search experiments...")
        self.search_bar.textChanged.connect(self.filter_experiments)

        self.exp_list = QListWidget()
        self.exp_list.setSelectionMode(QListWidget.MultiSelection)
        self.exp_list.itemClicked.connect(self.display_experiment_data)

        self.plot_selected_button = QPushButton("📈 Plot Selected")
        self.plot_selected_button.clicked.connect(self.plot_selected_experiments)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.search_bar)
        left_panel.addWidget(self.exp_list)
        left_panel.addWidget(self.plot_selected_button)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)

        self.plot_area = FigureCanvas(Figure(figsize=(5, 4)))
        self.ax = self.plot_area.figure.add_subplot(111)

        self.meta_view = QTextEdit()
        self.meta_view.setReadOnly(True)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.plot_area)
        splitter.addWidget(self.meta_view)

        # Optional: Set initial sizes (you can tweak these)
        splitter.setSizes([250, 600, 350])

        # Add to the main experiment layout
        outer_layout.addWidget(splitter)

        view_widget.setLayout(outer_layout)
        self.stack.addWidget(view_widget)
        self.refresh_experiment_list()

    def show_start_view(self):
        self.stack.setCurrentIndex(0)

    def show_experiment_view(self):
        self.stack.setCurrentIndex(1)
        self.refresh_experiment_list()

    def handle_start_wicking(self):
        def run_main_py():
            script_path = os.path.expanduser("main.py")
            try:
                subprocess.run(["python", script_path], check=True)
                QTimer.singleShot(0, lambda: self.show_message(self, "Done", "Wicking tracking completed."))
            except subprocess.CalledProcessError:
                QTimer.singleShot(0, lambda: self.show_message(self, "Error", "main.py failed to run."))

        threading.Thread(target=run_main_py, daemon=True).start()
        QMessageBox.information(self, "Started", "Wicking tracker started.")

    def refresh_experiment_list(self):
        self.exp_list.clear()
        self.all_folders = []
        if not os.path.exists(self.output_dir):
            return
        for folder in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, folder)
            if os.path.isdir(path):
                self.all_folders.append(folder)
        self.filter_experiments(self.search_bar.text())

    def filter_experiments(self, text):
        self.exp_list.clear()
        filtered = [f for f in self.all_folders if text.lower() in f.lower()]
        for folder in filtered:
            self.exp_list.addItem(folder)

    def display_experiment_data(self, item):
        folder_name = item.text()
        folder = os.path.join(self.output_dir, folder_name)
        csv_path = os.path.join(folder, "data.csv")
        json_path = os.path.join(folder, "metadata.json")

        self.ax.clear()
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'Time' in df.columns and 'Height' in df.columns:
                    self.ax.plot(df["Time"], df["Height"], marker='o', linestyle='-')
                    self.ax.set_title(f"{folder_name} - Height vs Time")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Height")
                else:
                    self.ax.text(0.5, 0.5, "Time or Height column missing", transform=self.ax.transAxes,
                                 ha='center', va='center', fontsize=12, color='red')
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Failed to load plot:\n{e}", transform=self.ax.transAxes,
                             ha='center', va='center', fontsize=10, color='red')
        else:
            self.ax.text(0.5, 0.5, "No CSV data found", transform=self.ax.transAxes,
                         ha='center', va='center', fontsize=12, color='red')
        self.plot_area.draw()

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                pretty = json.dumps(meta, indent=2)
                self.meta_view.setText(pretty)
            except Exception as e:
                self.meta_view.setText(f"Failed to load metadata: {e}")
        else:
            self.meta_view.setText("No metadata found.")

    def plot_selected_experiments(self):
        self.ax.clear()
        selected_items = self.exp_list.selectedItems()
        if not selected_items:
            self.ax.text(0.5, 0.5, "No experiments selected", transform=self.ax.transAxes,
                         ha='center', va='center', fontsize=12, color='gray')
            self.plot_area.draw()
            return

        for item in selected_items:
            folder_name = item.text()
            folder = os.path.join(self.output_dir, folder_name)
            csv_path = os.path.join(folder, "data.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if 'Time' in df.columns and 'Height' in df.columns:
                        self.ax.plot(df["Time"], df["Height"], label=folder_name, linewidth=2)
                except Exception as e:
                    print(f"Error reading {folder_name}: {e}")

        self.ax.set_title("Height vs Time - Multiple Experiments")
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Height (mm)")
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.plot_area.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    output_path = os.path.expanduser("output")
    window = WickingDashboard(output_path)
    window.show()
    sys.exit(app.exec_())