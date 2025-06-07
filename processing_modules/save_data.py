# save_data.py

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QTextEdit, QLabel, QPushButton, QFileDialog, QMessageBox
)
import os
import json
import datetime


class SaveDialog(QDialog):
    def __init__(self, df, height_plot_image, wicking_plot_image=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Experiment")
        self.df = df
        self.height_plot_image = height_plot_image
        self.wicking_plot_image = wicking_plot_image

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Experiment Name")

        self.operator_input = QLineEdit()
        self.operator_input.setPlaceholderText("Operator Name")

        self.comment_input = QTextEdit()
        self.comment_input.setPlaceholderText("Any comments...")

        self.save_button = QPushButton("💾 Save")
        self.save_button.clicked.connect(self.save_data)

        layout.addWidget(QLabel("Experiment Name:"))
        layout.addWidget(self.name_input)
        layout.addWidget(QLabel("Operator Name:"))
        layout.addWidget(self.operator_input)
        layout.addWidget(QLabel("Comments:"))
        layout.addWidget(self.comment_input)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def save_data(self):
        experiment_name = self.name_input.text().strip()
        operator_name = self.operator_input.text().strip()
        comment = self.comment_input.toPlainText().strip()

        if not experiment_name:
            QMessageBox.warning(self, "Missing Field", "Experiment name is required.")
            return

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{experiment_name}_{timestamp}"
        output_path = os.path.join(".", "output", folder_name)

        os.makedirs(output_path, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(output_path, "data.csv")
        self.df.to_csv(csv_path, index=False)

        # Save height plot
        height_path = os.path.join(output_path, "height_plot.png")
        if self.height_plot_image:
            self.height_plot_image.save(height_path)

        # Save wicking plot
        if self.wicking_plot_image:
            wicking_path = os.path.join(output_path, "wicking_plot.png")
            self.wicking_plot_image.save(wicking_path)

        # Save metadata
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "operator": operator_name,
            "comment": comment
        }

        json_path = os.path.join(output_path, "metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        QMessageBox.information(self, "Saved", f"Experiment saved to: {output_path}")
        self.accept()
