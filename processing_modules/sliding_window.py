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

from PyQt5.QtWidgets import QMessageBox

def calculate_delta(base_color, sliding_window_color):
    """ Calculate the Euclidean distance (delta) between two Lab colors """
    return np.linalg.norm(base_color - sliding_window_color)


def sliding_window(cam, bbox_x, bbox_y, bbox_w, bbox_h, height_in_mm, mm_per_pixel, average_base_color):
    df, height_plot_image = None, None,
    area_of_interest_offset = 0
    height = 0


    cv2.namedWindow("Sliding Window")

    start_time = datetime.datetime.now()
    plot_time = start_time + datetime.timedelta(seconds=15)

    # data_list = []
    df = pd.DataFrame(columns=['Time', 'Height', 'Wicking Rate','Avg Wicking Rate'])

    # plot setup
    plt.ion()
    fig1 = plt.figure("Height Plot")
    ax1 = fig1.gca()
    ax1.clear()

    original_delta_threshold = 38
    current_delta_threshold = original_delta_threshold
    last_height_update_time = start_time
    last_height_value = 0

    while True:
        sliding_color_LAB = []
        area_of_interest_y1 = bbox_y + bbox_h + area_of_interest_offset - 10
        area_of_interest_y2 = bbox_y + bbox_h + area_of_interest_offset

        for _ in range(5):
            frame = cam.capture_array()
            if frame is None:
                break

            region = frame[area_of_interest_y1:area_of_interest_y2, bbox_x:bbox_x + bbox_w]
            sliding_window = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2Lab), axis=(0, 1))
            sliding_color_LAB.append(sliding_window)

        average_sliding_window_color = np.mean(sliding_color_LAB, axis=0)
        delta_E_mean = calculate_delta(average_base_color, average_sliding_window_color)
        

        height = mm_per_pixel * (bbox_y + bbox_h - area_of_interest_y2) -3

        # Adjust AOI
        if(delta_E_mean > current_delta_threshold and height < height_in_mm):
            print(f"Delta greater than threshold ({current_delta_threshold:.2f}), moving AOI up.")
            area_of_interest_offset -= 2

            height = mm_per_pixel * (bbox_y + bbox_h - area_of_interest_y2) -1

        elif height >= height_in_mm:
            area_of_interest_offset = 0

        now = datetime.datetime.now()

        delta_time = (now - start_time).total_seconds()
        if abs(height - last_height_value) > 0.1:
            last_height_update_time = now
            last_height_value = height

        # [ADDED] Reduce threshold if no height change for 20 seconds
        if (now - last_height_update_time).total_seconds() > 20:
            new_threshold = current_delta_threshold * 0.95
            current_delta_threshold = max(new_threshold, 5)
            print(f"[INFO] No height change in 30s. Reducing delta threshold to {current_delta_threshold:.2f}")
            last_height_update_time = now  # reset timer

        # Update data for Average height
        avg_rate = height / delta_time if delta_time > 0 else 0
        df.loc[len(df)] = [delta_time, height, 0, avg_rate] 

        # Print live values
        print(f"Time: {delta_time:.2f} s | Delta E: {delta_E_mean:.4f} | Height: {height:.4f} mm | Delta Threshold: {current_delta_threshold:.2f} mm")

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