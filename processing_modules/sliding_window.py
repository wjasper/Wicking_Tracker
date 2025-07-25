#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
"""
import cv2
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image
from .post_processing_plots import post_process_wicking_rate
from .save_data import SaveDialog
from scipy.interpolate import CubicSpline
from PyQt5.QtWidgets import QMessageBox

def calculate_deltaE(base_color, sliding_window_color):
    """ Calculate the Euclidean distance (delta) between two Lab colors """
    return np.linalg.norm(base_color - sliding_window_color)

def sliding_window(cam, bbox_x, bbox_y, bbox_w, bbox_h,
                   height_in_mm, mm_per_pixel,
                   average_base_color,
                   update_status_func=None):

    df, height_plot_image = None, None,
    # parameters for 
    area_of_interest_offset = 0  # distance from bottom of bounding box to center of area of interest
    area_of_interest_h = 10      # height of area_of_interest (adaptive)
    height = 0                   # wicking height in mm
    delta_time = 0

    # Initialization and setup code: do only once
    cv2.namedWindow("Sliding Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sliding Window", 620, 520)

    start_time = datetime.datetime.now()
    plot_time = start_time + datetime.timedelta(seconds=15)

    # data_list = []
    df = pd.DataFrame(columns=['Time', 'Height','Avg Wicking Rate'])

    # plot setup
    plt.ion()
    fig1 = plt.figure("Height Plot")
    ax1 = fig1.gca()
    ax1.clear()

    max_delta_threshold = 38
    min_delta_threshold = 15
    height_threshold = 50
    current_delta_threshold = max_delta_threshold
    last_height_update_time = start_time
    last_height_value = 0

    while height < 101 and delta_time < 610:

        sliding_color_LAB = []

        if (height > height_threshold):
            area_of_interest_h = 20
            
        area_of_interest_y1 = bbox_y + bbox_h - (area_of_interest_offset + area_of_interest_h//2) # top of area_of_interest
        area_of_interest_y2 = bbox_y + bbox_h - (area_of_interest_offset - area_of_interest_h//2) # bottom of area_of_interest

        pixels_threshold = int(bbox_w * area_of_interest_h * 0.4)

        for _ in range(5):
            frame = cam.capture_array()
            if frame is None:
                break
            region = frame[area_of_interest_y1:area_of_interest_y2, bbox_x:bbox_x + bbox_w]
            sliding_window = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2Lab), axis=(0, 1))
            sliding_color_LAB.append(sliding_window)

        average_sliding_window_color = np.mean(sliding_color_LAB, axis=0)
        delta_E_mean = calculate_deltaE(average_base_color, average_sliding_window_color)

        height = mm_per_pixel*area_of_interest_offset
        now = datetime.datetime.now()
        delta_time = (now - start_time).total_seconds()  # calculate elapsed time

        # Adjust area_of_interest
        if(height < height_threshold):
            if(delta_E_mean > current_delta_threshold):
                print(f"Delta_E greater than threshold ({current_delta_threshold:.2f}), moving AOI up.")
                last_height_value = height     
                area_of_interest_offset += 2  # move the AOI up 2 pixels
                height = mm_per_pixel*area_of_interest_offset
                last_height_update_time = now  # restart the timer
        else:
            #Get a grayscale of the image
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Perform edge detection
            edges = cv2.Canny(gray, 50, 50)

            # Apply Probabilistic Hough Line Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=(bbox_w//2), maxLineGap=30)

            # Convert area_of_interest to HSV
            hsv_image = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 40, 40])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 40, 40])
            upper_red2 = np.array([179, 255, 255])

            lower_pink = np.array([145, 30, 80])     # Adjust these for more sensitivity
            upper_pink = np.array([170, 255, 255])
                                
            # Create a mask
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            mask3 = cv2.inRange(hsv_image, lower_pink, upper_pink)
            red_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
            red_object = cv2.bitwise_and(hsv_image, hsv_image, mask=red_mask)

#                cv2.imshow("red_mask", red_mask)
#                cv2.imshow("red objects", red_object)
#                cv2.imshow("region", region)

            # See if there is red in the region
            pixels = cv2.countNonZero(red_mask)
            print("number of red pixels =", pixels)

            #Draw the lines on the image
            if lines is not None and pixels > 0:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(region, (x1,y1), (x2,y2), (255,0,0), 2)
                    # area_of_interest_offset += max(area_of_interest_h//2 - y2, 0)
                    area_of_interest_offset += min(max(area_of_interest_h//2 - y2, 0), 2)
                    print("y2: ", y2)
                    last_height_value = height     
                    height = mm_per_pixel*area_of_interest_offset
                
#            elif delta_E_mean > current_delta_threshold: 
#                print(f"No edge deteced in image, moving AOI up")
#                last_height_value = height     
#                area_of_interest_offset += 2  # move the AOI up 2 pixels
#                height = mm_per_pixel*area_of_interest_offset
#                last_height_update_time = now  # restart the timer
            else:
                if pixels > pixels_threshold:
                    last_height_value = height     
                    area_of_interest_offset += 2  # move the AOI up 2 pixels
                    height = mm_per_pixel*area_of_interest_offset
                    last_height_update_time = now  # restart the timer
                    print("Red in region")
                else:
                    print("No red in region")

        if round(height) >= height_in_mm:
            print("height greater than height of the bounding box", height)
            break

        # pixels_threshold_reduced = False
        
        if (now - last_height_update_time).total_seconds() > 4:
            new_threshold = delta_E_mean * 0.95
            current_delta_threshold = min(current_delta_threshold, max_delta_threshold, new_threshold)
            current_delta_threshold = max(current_delta_threshold, min_delta_threshold)
            print(f"[INFO] No height change in 4s. Reducing delta threshold to {current_delta_threshold:.2f}")
            print(f"Pixel Threshold {pixels_threshold:.2f}")
            # pixels_threshold = 0.5 * pixels_threshold

            # if not pixels_threshold_reduced:
            #     pixels_threshold = 0.5 * pixels_threshold
            #     pixels_threshold_reduced = True
        
        if len(df) >= 4:
            t_window = df["Time"].iloc[-4:].values
            h_window = df["Height"].iloc[-4:].values

            # Sort by time just in case
            sort_idx = np.argsort(t_window)
            t_window = t_window[sort_idx]
            h_window = h_window[sort_idx]

            # Fit cubic spline on the window
            cs = CubicSpline(t_window, h_window)

            # Create uniform time array within the range of t_window
            t_uniform = np.linspace(t_window[0], t_window[-1], len(t_window))

            # Derivative of the spline gives the rate
            wicking_rate_deri = cs.derivative()(t_uniform)
            wicking_rate = wicking_rate_deri[-1]
        else:
            wicking_rate = 0.0

        # Update data for Average height
        avg_rate = height / delta_time if delta_time > 0 else 0
        df.loc[len(df)] = [delta_time, height, avg_rate] 

        if update_status_func:
            update_status_func(delta_time, delta_E_mean, height, avg_rate, current_delta_threshold)
        # Print live values
        print(f"Time: {delta_time:.2f} s | Delta E: {delta_E_mean:.4f} | Height: {height:.4f} mm | Delta Threshold: {current_delta_threshold:.2f} | Wicking Rate: {wicking_rate:.2f} mm")

        if delta_time > 0:
            print(f"Avg Wicking Rate: {height/delta_time:.4f} mm/s")
        else:
            print("Avg Wicking Rate: N/A (delta_time is zero)")

        # Draw bounding boxes
        cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 255), 2)
        cv2.rectangle(frame, (bbox_x, area_of_interest_y1), (bbox_x + bbox_w, area_of_interest_y2), (0, 255, 0), 1)
        cv2.imshow("Sliding Window", frame)

        # Plot every 10 seconds
        if now > plot_time:
            plot_time += datetime.timedelta(seconds=10)

            # Plot height
            ax1.clear()
            sns.lineplot(ax=ax1, data=df, x="Time", y="Height")
            ax1.set_title("Height Over Time")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Height (mm)")

            fig1.canvas.draw()
            plt.pause(0.1)
            
            #Save Height Plot
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png')
            buf1.seek(0)
            height_plot_image = Image.open(buf1)

        # Exit
        key = cv2.waitKeyEx(40)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

    df, height_plot_image, wicking_plot_image = post_process_wicking_rate(df)

    save_reply = QMessageBox.question(
        None,
        "Save Experiment",
        "Do you want to save the experiment?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes
    )

    if save_reply == QMessageBox.Yes:
        dlg = SaveDialog(df, height_plot_image, wicking_plot_image)
        dlg.exec_()
    
    return df, height_plot_image
