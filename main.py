#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper and Shivam Ghodke
"""

import platform
from picamera2 import Picamera2
from processing_modules.calibration import calibration, base_color
from processing_modules.sliding_window import sliding_window
from processing_modules.save_data import save_data

def main():
    # Set camera properties
    framerate = 30  # Reduced for better interactivity
    width = 640     # Width of video frame
    height = 480    # Height of video frame

    # Initialize the PiCamera2
    cam = Picamera2()
    main = {"size": (width, height), "format": "RGB888"}
    controls = {"FrameRate": framerate}
    sensor = {"bit_depth": 15, "output_size": (2028, 1520)}
    video_config = cam.create_video_configuration(
        main, controls=controls, sensor=sensor
    )
    cam.configure(video_config)
    
    cam.start()
    print("Starting calibration...")
    bbox_x, bbox_y, bbox_w, bbox_h, height_in_cm, cm_per_pixel = calibration(cam, height, width)
    average_base_color = base_color(cam, bbox_x, bbox_y, bbox_w, bbox_h)
    print("Calibration complete.")

    
    start_sliding_window_input = str(input("Enter y to start wicking tracker: "))
    if start_sliding_window_input.upper() == "Y":
        df, plot_image = sliding_window(cam, bbox_x, bbox_y, bbox_w, bbox_h, height_in_cm, cm_per_pixel, average_base_color)

        save_data_input = str(input("Enter y to save data: "))
        if save_data_input.upper() == "Y":
            save_data(df, plot_image)

    cam.stop()

    print("Program completed successfully")

if __name__ == "__main__":
    if platform.system() == "Linux":
        main()
    else:
        print("OS not compatible")
