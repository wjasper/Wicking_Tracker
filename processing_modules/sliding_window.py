import cv2
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import io
from PIL import Image


def calculate_delta(base_color, sliding_window_color):
    """ Calculate the Euclidean distance (delta) between two Lab colors """
    return np.linalg.norm(base_color - sliding_window_color)

def sliding_window(cam, bbox_x, bbox_y, bbox_w, bbox_h, height_in_cm, cm_per_pixel, average_base_color):
    
    df, plot_image = None, None
    area_of_interest_offset = 0
    height = 0

    cv2.namedWindow("Sliding Window")

    # initialize start time
    start_time = datetime.datetime.now()
    delta_time = 0
    plot_time = start_time + datetime.timedelta(seconds=15)

    data_list = []
    data_list.append([delta_time, height])  # Initial dummy data
    df = pd.DataFrame(data_list, columns=['Time', 'Height'])

    # plot setup
    plt.ion()

    fig1 = plt.figure("Height Plot")
    ax1 = fig1.gca()
    ax1.clear()

    fig2 = plt.figure("Wicking Rate Plot")
    ax2 = fig2.gca()
    ax2.clear()

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
        delta = calculate_delta(average_base_color, average_sliding_color)
        height = cm_per_pixel * (bbox_y + bbox_h - area_of_interest_y1)
        print("Delta:", delta, "Height:", height)

        if delta > 20 and height < height_in_cm:
            print("Delta greater than 20, moving area of interest window up.")
            area_of_interest_offset -= 2
        elif height >= height_in_cm:
            area_of_interest_offset = 0

        # Draw bounding box and AOI
        cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 255), 2)
        cv2.rectangle(frame, (bbox_x, area_of_interest_y1), (bbox_x + bbox_w, area_of_interest_y2), (0, 255, 0), 1)
        cv2.imshow("Sliding Window", frame)

        now = datetime.datetime.now()

        if now > plot_time:
            plot_time += datetime.timedelta(seconds=15)
            delta_time = (now - start_time).total_seconds()

            # append data
            data_list.append([delta_time, height])
            df = pd.DataFrame(data_list, columns=['Time', 'Height'])

            # cubicspline
            if len(df) >= 3:
                cs = CubicSpline(df["Time"], df["Height"])
                df["Wicking Rate"] = cs.derivative()(df["Time"])

                latest_rate = df["Wicking Rate"].iloc[-1]
                print(f"Wicking Rate (spline-based): {latest_rate:.4f} cm/s")
            else:
                df["Wicking Rate"] = 0

            # plot height
            ax1.clear()
            sns.lineplot(ax=ax1, data=df, x="Time", y="Height")
            ax1.set_title("Height Over Time")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Height (cm)")

            # wicking rate
            ax2.clear()
            sns.lineplot(ax=ax2, data=df, x="Time", y="Wicking Rate")
            ax2.set_title("Wicking Rate Over Time")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Wicking Rate")

            # Redraw Plots
            fig1.canvas.draw()
            fig2.canvas.draw()
            plt.pause(0.1)

            # Save latest Wicking Rate plot
            buf = io.BytesIO()
            fig2.savefig(buf, format='png')
            buf.seek(0)
            plot_image = Image.open(buf)

        key = cv2.waitKeyEx(40)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

    return df, plot_image
