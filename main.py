#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper and Shivam Ghodke
"""

import platform
import sys

def main():
    from processing_modules.calibration import calibration, base_color
    from processing_modules.sliding_window import sliding_window
    from processing_modules.save_data import SaveDialog

    from PyQt5.QtWidgets import QApplication, QMessageBox
    if not QApplication.instance():
        app = QApplication([]) 
    
    import numpy as np

    # Set camera properties
    framerate = 30
    width = 640
    height = 480

    USE_DUMMY_CAMERA = True  # Set to False when real camera is present

    if USE_DUMMY_CAMERA:
        class DummyCamera:
            def start(self):
                print("[INFO] Dummy camera started")

            def stop(self):
                print("[INFO] Dummy camera stopped")

            def capture_array(self):
                return np.zeros((height, width, 3), dtype=np.uint8)

        cam = DummyCamera()

    else:
        from picamera2 import Picamera2
        cam = Picamera2()

        main_config = {"size": (width, height), "format": "RGB888"}
        controls = {"FrameRate": framerate}
        sensor = {"bit_depth": 15, "output_size": (2028, 1520)}
        video_config = cam.create_video_configuration(
            main_config, controls=controls, sensor=sensor
        )
        cam.configure(video_config)

    cam.start()
    print("STATUS: Calibration started", flush=True)

    result = calibration(cam, height, width)
    if result is None:
        print("STATUS: Calibration cancelled", flush=True)
        cam.stop()
        return

    bbox_x, bbox_y, bbox_w, bbox_h, height_in_mm, mm_per_pixel = result
    average_base_color = base_color(cam, bbox_x, bbox_y, bbox_w, bbox_h)
    print("STATUS: Calibration complete", flush=True)

    start_reply = QMessageBox.question(
        None,
        "Start Tracker",
        "Start wicking tracker?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes
    )

    if start_reply != QMessageBox.Yes:
        print("Aborted tracker.")
        cam.stop()
        return
    
    print("STATUS: Wicking tracker started", flush=True)

    df, plot_image = sliding_window(
        cam, bbox_x, bbox_y, bbox_w, bbox_h, height_in_mm, mm_per_pixel, average_base_color)

if __name__ == "__main__":
    if platform.system() == "Linux":
        sys.path.append('/home/pi/opencv/Wicking_Tracker/processing_modules')  # adjust as needed
        main()
    elif platform.system() == "Windows":
        sys.path.append(r'C:/Users/otandel/Downloads/Codes/Wicking/Wicking_Tracker/processing_modules')
        main()
    else:
        print("OS not compatible")
