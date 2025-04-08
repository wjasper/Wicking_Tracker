import cv2
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image




def calculate_delta(base_color, sliding_window_color):
    """ Calculate the Euclidean distance (delta) between two Lab colors """
    return np.linalg.norm(base_color - sliding_window_color)

def sliding_window(cam, bbox_x, bbox_y, bbox_w, bbox_h, height_in_inches, inch_per_pixel):
    
    df, plot_image = None, None
    
    base_color = None
    sliding_window_color = None
    area_of_interest_offset = 0
    height = 0
    
    cv2.namedWindow("Sliding Window")

    # initialize start time
    start_time = datetime.datetime.now()
    delta_time = 0                                     # elapsed time for sampling
    plot_time = start_time + datetime.timedelta(0,15)  # plot time every 15 seconds

    # Initialize an empty list for the data
    data_list = []
    
    # Append data to the list and create a Pandas DataFrame from it
    data_list.append([delta_time, height])  # initialize the data
    df = pd.DataFrame(data_list, columns = ['Time', 'Height'])

    # Setup the plot
    plt.ion()  # Turn on interactive mode
    
    # Create a seaborn plot
    sns.lineplot(data = df, x = "Time", y = "Height")
    
    # Redraws plot
    plt.draw()
    plt.pause(0.1)  # Pause to allow plot update
    
    base_colors = []

    running = True

    print("Calibrating wicking, this may take a while ...")
    for _ in range(500):  # Loop to capture the color 500 times
        frame = cam.capture_array()
        if frame is None:
            break
        
        # Compute base_color for each iteration
        base_color = np.mean(cv2.cvtColor(
            frame[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w], cv2.COLOR_BGR2Lab), axis=(0, 1))
        base_colors.append(base_color)
    
    # Now calculate the average of all collected base colors
    average_base_color = np.mean(base_colors, axis=0)  # Average across the 100 frames
    print("Average Base Color (Lab):", average_base_color)
        
    while True:
        
        sliding_window_colors = []
        
        area_of_interest_y1 = bbox_y + bbox_h + area_of_interest_offset - 2
        area_of_interest_y2 = bbox_y + bbox_h + area_of_interest_offset
        
        for _ in range(10):
            frame = cam.capture_array()
            if frame is None:
                break
            
            sliding_window_color = np.mean(cv2.cvtColor(
                frame[area_of_interest_y1:area_of_interest_y2, bbox_x:bbox_x + bbox_w], cv2.COLOR_BGR2Lab), axis=(0, 1))
            sliding_window_colors.append(sliding_window_color)
        
        average_sliding_color = np.mean(sliding_window_colors, axis=0)
        
        delta = calculate_delta(average_base_color, average_sliding_color)
        height = inch_per_pixel*(bbox_y + bbox_h - area_of_interest_y1)
        print("Delta:", delta, "Height:", height)
        
        # If delta is greater than 50, move the area of interest up
        if delta > 50 and height < height_in_inches:
            print("Delta greater than 50, moving area of interest window up.")
            area_of_interest_offset -= 2 # Move the area of interest window up (you can adjust this step size)
        elif height >= height_in_inches:
            area_of_interest_offset = 0  # Reset to bottom for testing
            
        # Bounding Box
        cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 255), 2)
        
        # Area of interest
        cv2.rectangle(frame, (bbox_x, area_of_interest_y1), (bbox_x + bbox_w, area_of_interest_y2), (0, 255, 0), 1)
        
        cv2.imshow("Sliding Window", frame)
        
        
        now = datetime.datetime.now()
                      
        if now > plot_time:

            plot_time = plot_time + datetime.timedelta(0,15)  # increment plot_time by 15 seconds
            delta_time = (now - start_time).total_seconds()

            # Append data to the list and create a Pandas DataFrame from it
            data_list.append([delta_time, height])
            df = pd.DataFrame(data_list, columns = ['Time', 'Height'])

            # Clears old plot
            plt.clf()
    
            # Create a seaborn plot
            sns.lineplot(data = df, x = "Time", y = "Height")
            
            plt.ylabel("Height (inches)")
            plt.xlabel("Time (seconds)")
            
            # Redraws plot
            plt.draw()
            plt.pause(0.1)  # Pause to allow plot update
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_image = Image.open(buf)
                    
        key = cv2.waitKeyEx(40)
        if key == ord("q"):
            break

            
    cv2.destroyAllWindows()

    plt.ioff() 
    plt.close()
    
    return df, plot_image