#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper and Shivam Ghodke
"""
# IMPORTS
import cv2
import sys
import datetime
import time
import platform
import libcamera
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from picamera2 import Picamera2
import os
import io
from PIL import Image
import json


# Global variables for the bounding box
bbox_x = 150
bbox_y = 100
bbox_w = 300
bbox_h = 200
dragging = False
drag_start_x = 0
drag_start_y = 0
resize_corner = None
inch_per_pixel = 0
height_in_inches = 17
df = None
plot_image = None

def on_mouse(event, x, y, flags, param):
    global bbox_x, bbox_y, bbox_w, bbox_h, dragging, drag_start_x, drag_start_y, resize_corner
    
    # Define the sensitivity range for grabbing a corner or edge
    sensitivity = 15
    
    # Bounding box corners and edges
    top_left = (bbox_x, bbox_y)
    top_right = (bbox_x + bbox_w, bbox_y)
    bottom_left = (bbox_x, bbox_y + bbox_h)
    bottom_right = (bbox_x + bbox_w, bbox_y + bbox_h)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is near a corner for resizing
        if abs(x - top_left[0]) < sensitivity and abs(y - top_left[1]) < sensitivity:
            resize_corner = "top_left"
            dragging = True
        elif abs(x - top_right[0]) < sensitivity and abs(y - top_right[1]) < sensitivity:
            resize_corner = "top_right"
            dragging = True
        elif abs(x - bottom_left[0]) < sensitivity and abs(y - bottom_left[1]) < sensitivity:
            resize_corner = "bottom_left"
            dragging = True
        elif abs(x - bottom_right[0]) < sensitivity and abs(y - bottom_right[1]) < sensitivity:
            resize_corner = "bottom_right"
            dragging = True
        # Check if click is inside the box for moving
        elif bbox_x < x < bbox_x + bbox_w and bbox_y < y < bbox_y + bbox_h:
            resize_corner = "move"
            dragging = True
            drag_start_x = x - bbox_x
            drag_start_y = y - bbox_y
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Constrain to frame boundaries
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        
        if resize_corner == "top_left":
            # Calculate new width and height
            new_w = bbox_x + bbox_w - x
            new_h = bbox_y + bbox_h - y
            # Only update if size is reasonable
            if new_w > 50 and new_h > 50:
                bbox_w = new_w
                bbox_h = new_h
                bbox_x = x
                bbox_y = y
        elif resize_corner == "top_right":
            new_w = x - bbox_x
            new_h = bbox_y + bbox_h - y
            if new_w > 50 and new_h > 50:
                bbox_w = new_w
                bbox_y = y
                bbox_h = new_h
        elif resize_corner == "bottom_left":
            new_w = bbox_x + bbox_w - x
            new_h = y - bbox_y
            if new_w > 50 and new_h > 50:
                bbox_w = new_w
                bbox_x = x
                bbox_h = new_h
        elif resize_corner == "bottom_right":
            new_w = x - bbox_x
            new_h = y - bbox_y
            if new_w > 50 and new_h > 50:
                bbox_w = new_w
                bbox_h = new_h
        elif resize_corner == "move":
            # Move the entire box
            new_x = x - drag_start_x
            new_y = y - drag_start_y
            # Constrain to frame boundaries
            if new_x >= 0 and new_x + bbox_w <= width:
                bbox_x = new_x
            if new_y >= 0 and new_y + bbox_h <= height:
                bbox_y = new_y
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        dragging = False
        resize_corner = None

def calibration(cam):
    global bbox_x, bbox_y, bbox_w, bbox_h, inch_per_pixel
    global height_in_inches
        
    # Create window and set mouse callback
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse)
    
    # Instructions to display on screen
    instructions = "Drag corners to resize, drag center to move. Press 'q' to quit, 's' to save."
    
    # Try to find initial bounding box
    found_initial_bbox = False
    
    while True:
        frame = cam.capture_array()
        if frame is None:
            break
        
        # Try to detect the white cloth automatically if no manual adjustment yet
        if not found_initial_bbox and not dragging:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Only update if the contour is reasonably sized
                if w > 50 and h > 50:
                    bbox_x, bbox_y, bbox_w, bbox_h = x, y, w, h
                    found_initial_bbox = True
        
        # Draw the bounding box with markers at corners for resizing
        cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 255), 2)
        
        # Draw corner markers
        corner_size = 6
        # Top-left
        cv2.rectangle(frame, (bbox_x - corner_size, bbox_y - corner_size), 
                     (bbox_x + corner_size, bbox_y + corner_size), (255, 0, 0), -1)
        # Top-right
        cv2.rectangle(frame, (bbox_x + bbox_w - corner_size, bbox_y - corner_size), 
                     (bbox_x + bbox_w + corner_size, bbox_y + corner_size), (255, 0, 0), -1)
        # Bottom-left
        cv2.rectangle(frame, (bbox_x - corner_size, bbox_y + bbox_h - corner_size), 
                     (bbox_x + corner_size, bbox_y + bbox_h + corner_size), (255, 0, 0), -1)
        # Bottom-right
        cv2.rectangle(frame, (bbox_x + bbox_w - corner_size, bbox_y + bbox_h - corner_size), 
                     (bbox_x + bbox_w + corner_size, bbox_y + bbox_h + corner_size), (255, 0, 0), -1)
        
        # Display instructions on the frame
        cv2.putText(frame, instructions, (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the current bounding box position
        position_text = f"Box: x={bbox_x}, y={bbox_y}, w={bbox_w}, h={bbox_h}"
        cv2.putText(frame, position_text, (10, height-  40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("Calibration", frame)
        
        
        key = cv2.waitKeyEx(40)


        #52, 50, 54
        
        # Up arrow
        if key == 56:
            if bbox_y > 0:
                bbox_y -= 2
                bbox_h += 2
                

        # Down arrow 
        elif key == 50:
            if bbox_h > 50:  
                bbox_y += 2
                bbox_h -= 2

        # Left arrow 
        elif key == 52:
            if bbox_x > 0:
                bbox_x -= 2
                bbox_w += 2

        # Right arrow 
        elif key == 54:
            if bbox_w > 50: 
                bbox_x += 2
                bbox_w -= 2

                
        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    
    height_in_inches = int(input("Enter reading corresponding to the box in inches: "))
    inch_per_pixel = height_in_inches/bbox_h
    
    return (bbox_x, bbox_y, bbox_w, bbox_h, height_in_inches, inch_per_pixel)

def calculate_delta(base_color, sliding_window_color):
    """ Calculate the Euclidean distance (delta) between two Lab colors """
    return np.linalg.norm(base_color - sliding_window_color)

def sliding_window(cam):
    
    global bbox_x, bbox_y, bbox_w, bbox_h, inch_per_pixel
    global height_in_inches
    global df, plot_image
    
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


def save_data():
    global df, plot_image

    experiment_name = input("Enter Experiment Name: ")

    # Generate folder name with UTC timestamp
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{experiment_name}_{timestamp}"
    output_path = os.path.join("output", folder_name)
 
    os.makedirs(output_path, exist_ok=True)
 
    # Save CSV
    csv_path = os.path.join(output_path, "data.csv")
    df.to_csv(csv_path, index=False)
    
    # Save Image
    image_path = os.path.join(output_path, "plot.png")
    plot_image.save(image_path)
 
    operator_name = input("Enter Operator Name: ")
    comment = input("Enter Comments: ")
    
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "operator": operator_name,
        "comment": comment
    }

    # Save metadata as JSON
    json_path = os.path.join(output_path, "metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

 
    print(f"Data saved to: {csv_path}")
    

def main():
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
    calibration(cam)
    sliding_window(cam)
    cam.stop()
    save_data()

    
# Set camera properties
framerate = 30  # Reduced for better interactivity
width = 640     # Width of video frame
height = 480    # Height of video frame

if __name__ == "__main__":
    if platform.system() == "Linux":
        main()
    else:
        print("OS not compatible")
