import cv2
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
from scipy.signal import savgol_filter

def calculate_delta(base_color, sliding_window_color):
    """ Calculate the Euclidean distance (delta) between two Lab colors """
    return np.linalg.norm(base_color - sliding_window_color)

def sliding_window(cam, bbox_x, bbox_y, bbox_w, bbox_h, height_in_mm, mm_per_pixel, average_base_color):
    df, plot_image = None, None
    area_of_interest_offset = 0
    height = 0
    rate = 0

    cv2.namedWindow("Sliding Window")

    start_time = datetime.datetime.now()
    plot_time = start_time + datetime.timedelta(seconds=15)

    # data_list = []
    df = pd.DataFrame(columns=['Time', 'Height', 'Wicking Rate'])

    # plot setup
    plt.ion()
    fig1 = plt.figure("Height Plot")
    ax1 = fig1.gca()
    ax1.clear()
    fig2 = plt.figure("Wicking Rate Plot")
    ax2 = fig2.gca()
    ax2.clear()

    original_delta_threshold = 40
    current_delta_threshold = original_delta_threshold
    last_height_update_time = start_time
    last_height_value = 0

    while True:
        sliding_window_colors = []
        area_of_interest_y1 = bbox_y + bbox_h + area_of_interest_offset - 2
        area_of_interest_y2 = bbox_y + bbox_h + area_of_interest_offset

        for _ in range(10):
            frame = cam.capture_array()
            if frame is None:
                break
            sliding_window_color = np.mean(cv2.cvtColor(
                frame[area_of_interest_y1:area_of_interest_y2, bbox_x:bbox_x + bbox_w],
                cv2.COLOR_BGR2Lab), axis=(0, 1))
            sliding_window_colors.append(sliding_window_color)

        average_sliding_color = np.mean(sliding_window_colors, axis=0)
        delta_E_mean = calculate_delta(average_base_color, average_sliding_color)
        height = mm_per_pixel * (bbox_y + bbox_h - area_of_interest_y1)

        now = datetime.datetime.now()

        delta_time = (now - start_time).total_seconds()
        if abs(height - last_height_value) > 0.1:
            last_height_update_time = now
            last_height_value = height

        # [ADDED] Reduce threshold if no height change for 30 seconds
        if (now - last_height_update_time).total_seconds() > 30:
            new_threshold = current_delta_threshold * 0.9
            current_delta_threshold = max(new_threshold, 5)
            print(f"[INFO] No height change in 30s. Reducing delta threshold to {current_delta_threshold:.2f}")
            last_height_update_time = now  # reset timer

        # Update data
        # data_list.append([delta_time, height])
        # df = pd.DataFrame(data_list, columns=['Time', 'Height'])
        df.loc[len(df)] = [delta_time, height, 0] 

        # Calculate wicking rate using cubic polynomial (sliding 4-point window)
        if len(df) >= 4:
            t_window = df["Time"].iloc[-4:].values
            h_window = df["Height"].iloc[-4:].values
            coeffs = np.polyfit(t_window, h_window, 3)
            a, b, c, d = coeffs
            t_latest = t_window[-1]
            rate = 3 * a * t_latest**2 + 2 * b * t_latest + c
        else:
            rate = 0

        df.at[df.index[-1], "Wicking Rate"] = rate

        # Print live values
        print(f"Time: {delta_time:.2f} s | Delta E: {delta_E_mean:.4f} | Height: {height:.4f} mm | Wicking Rate: {rate:.4f} mm/s")

        # Adjust AOI
        if delta_E_mean > current_delta_threshold and height < height_in_mm:
            print(f"Delta greater than threshold ({current_delta_threshold:.2f}), moving AOI up.")
            area_of_interest_offset -= 2
        elif height >= height_in_mm:
            area_of_interest_offset = 0

        # Draw bounding boxes
        cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 255), 2)
        cv2.rectangle(frame, (bbox_x, area_of_interest_y1), (bbox_x + bbox_w, area_of_interest_y2), (0, 255, 0), 1)
        cv2.imshow("Sliding Window", frame)

        # Plot every 15 seconds
        if now > plot_time:
            plot_time += datetime.timedelta(seconds=2)

            # Plot height
            ax1.clear()
            sns.lineplot(ax=ax1, data=df, x="Time", y="Height")
            ax1.set_title("Height Over Time")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Height (mm)")

            # Plot wicking rate
            ax2.clear()
            # sns.lineplot(ax=ax2, data=df, x="Time", y="Wicking Rate")
            df_plot = df.copy()

            df_plot["Smoothed Wicking Rate"] = df["Wicking Rate"]
            sns.lineplot(ax=ax2, data=df_plot, x="Time", y="Smoothed Wicking Rate")
            
            ax2.set_title("Wicking Rate Over Time")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Wicking Rate")

            fig1.canvas.draw()
            fig2.canvas.draw()
            plt.pause(0.1)

            # Save Wicking Rate plot
            buf = io.BytesIO()
            fig2.savefig(buf, format='png')
            buf.seek(0)
            plot_image = Image.open(buf)

        # Exit
        key = cv2.waitKeyEx(40)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

    return df, plot_image