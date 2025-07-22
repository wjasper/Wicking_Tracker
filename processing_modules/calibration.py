#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:20:15 2025
@author: Dr. Warren Jasper
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox

class BoundingBox:
    def __init__(self, x=240, y=16, w=132, h=458):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.resize_corner = None
        self.mm_per_pixel = 0
        self.selected_border = None  # Tracks which border is selected


    def handle_mouse(self, event, x, y, flags, param):
        height = param["height"]
        width = param["width"]
        sensitivity = 30

        top_left = (self.x, self.y)
        top_right = (self.x + self.w, self.y)
        bottom_left = (self.x, self.y + self.h)
        bottom_right = (self.x + self.w, self.y + self.h)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Corner resize
            if abs(x - top_left[0]) < sensitivity and abs(y - top_left[1]) < sensitivity:
                self.resize_corner = "top_left"
                self.dragging = True
                self.selected_border = None
            elif abs(x - top_right[0]) < sensitivity and abs(y - top_right[1]) < sensitivity:
                self.resize_corner = "top_right"
                self.dragging = True
                self.selected_border = None
            elif abs(x - bottom_left[0]) < sensitivity and abs(y - bottom_left[1]) < sensitivity:
                self.resize_corner = "bottom_left"
                self.dragging = True
                self.selected_border = None
            elif abs(x - bottom_right[0]) < sensitivity and abs(y - bottom_right[1]) < sensitivity:
                self.resize_corner = "bottom_right"
                self.dragging = True
                self.selected_border = None

            # Border select (prioritized before drag-inside)
            elif abs(y - self.y) < sensitivity and self.x < x < self.x + self.w:
                self.selected_border = "top"
            elif abs(y - (self.y + self.h)) < sensitivity and self.x < x < self.x + self.w:
                self.selected_border = "bottom"
            elif abs(x - self.x) < sensitivity and self.y < y < self.y + self.h:
                self.selected_border = "left"
            elif abs(x - (self.x + self.w)) < sensitivity and self.y < y < self.y + self.h:
                self.selected_border = "right"

            # Drag inside
            elif self.x < x < self.x + self.w and self.y < y < self.y + self.h:
                self.resize_corner = "move"
                self.dragging = True
                self.drag_start_x = x - self.x
                self.drag_start_y = y - self.y
                self.selected_border = None

            else:
                self.selected_border = None

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            x = max(0, min(x, width))
            y = max(0, min(y, height))

            if self.resize_corner == "top_left":
                new_w = self.x + self.w - x
                new_h = self.y + self.h - y
                if new_w > 50 and new_h > 50:
                    self.w = new_w
                    self.h = new_h
                    self.x = x
                    self.y = y
            elif self.resize_corner == "top_right":
                new_w = x - self.x
                new_h = self.y + self.h - y
                if new_w > 50 and new_h > 50:
                    self.w = new_w
                    self.y = y
                    self.h = new_h
            elif self.resize_corner == "bottom_left":
                new_w = self.x + self.w - x
                new_h = y - self.y
                if new_w > 50 and new_h > 50:
                    self.w = new_w
                    self.x = x
                    self.h = new_h
            elif self.resize_corner == "bottom_right":
                new_w = x - self.x
                new_h = y - self.y
                if new_w > 50 and new_h > 50:
                    self.w = new_w
                    self.h = new_h
            elif self.resize_corner == "move":
                new_x = x - self.drag_start_x
                new_y = y - self.drag_start_y
                if new_x >= 0 and new_x + self.w <= width:
                    self.x = new_x
                if new_y >= 0 and new_y + self.h <= height:
                    self.y = new_y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resize_corner = None


def calibration(cam, height, width):
    bbox = BoundingBox(x=240, y=16, w=132, h=458)
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", bbox.handle_mouse, {"height": height, "width": width})
    instructions = "Drag corners to resize, drag center to move. Press 'q' to quit." 
    found_initial_bbox = False

    while True:
        frame = cam.capture_array()
        if frame is None:
            break

    #     if not found_initial_bbox and not bbox.dragging:
    #         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #         _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    #         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #         if contours:
    #             largest = max(contours, key=cv2.contourArea)
    #             x, y, w, h = cv2.boundingRect(largest)
    #             if w > 50 and h > 50:
    #                 bbox.x, bbox.y, bbox.w, bbox.h = x, y, w, h
    #                 found_initial_bbox = True

        # Draw bounding box
        cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (0, 0, 255), 2)

        # Draw corner markers
        cs = 6
        cv2.rectangle(frame, (bbox.x - cs, bbox.y - cs), (bbox.x + cs, bbox.y + cs), (255, 0, 0), -1)  # top-left
        cv2.rectangle(frame, (bbox.x + bbox.w - cs, bbox.y - cs), (bbox.x + bbox.w + cs, bbox.y + cs), (255, 0, 0), -1)  # top-right
        cv2.rectangle(frame, (bbox.x - cs, bbox.y + bbox.h - cs), (bbox.x + cs, bbox.y + bbox.h + cs), (255, 0, 0), -1)  # bottom-left
        cv2.rectangle(frame, (bbox.x + bbox.w - cs, bbox.y + bbox.h - cs), (bbox.x + bbox.w + cs, bbox.y + bbox.h + cs), (255, 0, 0), -1)  # bottom-right

        cv2.putText(frame, instructions, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Box: x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h}", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if bbox.selected_border:
            cv2.putText(frame, f"Selected: {bbox.selected_border}", (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKeyEx(40)

        if key == ord("q"):
            break

        # UP
        if key == 56:  # up arrow
            if bbox.selected_border == "top" and bbox.h > 50:
                bbox.y -= 2
                bbox.h += 2
            elif bbox.selected_border == "bottom" and bbox.h > 50:
                bbox.h -= 2
            elif bbox.selected_border == "move" and bbox.y > 0:
                bbox.y -= 2

        # DOWN
        elif key == 50:
            if bbox.selected_border == "top" and bbox.h > 50:
                bbox.y += 2
                bbox.h -= 2
            elif bbox.selected_border == "bottom":
                bbox.h += 2
            elif bbox.selected_border == "move" and bbox.y + bbox.h < height:
                bbox.y += 2

        # LEFT
        elif key == 52:
            if bbox.selected_border == "left" and bbox.w > 50:
                bbox.x -= 2
                bbox.w += 2
            elif bbox.selected_border == "right" and bbox.w > 50:
                bbox.w -= 2
            elif bbox.selected_border == "move" and bbox.x > 0:
                bbox.x -= 2

        # RIGHT
        elif key == 54:
            if bbox.selected_border == "left" and bbox.w > 50:
                bbox.x += 2
                bbox.w -= 2
            elif bbox.selected_border == "right":
                bbox.w += 2
            elif bbox.selected_border == "move" and bbox.x + bbox.w < width:
                bbox.x += 2

    cv2.destroyAllWindows()

    height_in_mm = None
    while True:
        text, ok = QInputDialog.getText(
            None,
            "Enter Height in mm",
            "Enter reading corresponding to the box height in mm (leave blank to cancel):"
        )

        if not ok or text.strip() == "":
            QMessageBox.information(None, "Cancelled", "Quitting calibration...")
            return None

        try:
            height_in_mm = int(text)
            break
        except ValueError:
            QMessageBox.warning(None, "Invalid Input", "Please enter a valid integer.")
        
#    bbox.mm_per_pixel = 0.449735
    bbox.mm_per_pixel = 160/458
    # height_in_mm = bbox.mm_per_pixel * bbox.h

    print("Hardcoded mm_per_pixel:", bbox.mm_per_pixel)
    print("Calculated height_in_mm:", height_in_mm)

    return (bbox.x, bbox.y, bbox.w, bbox.h, height_in_mm, bbox.mm_per_pixel)


def base_color(cam, bbox_x, bbox_y, bbox_w, bbox_h):
    # cv2.namedWindow("Getting average over 100 frames")
    base_colors = []

    print("STATUS: Calibrating wicking, this may take a while ...", flush=True)
    for _ in range(100):  # Loop to capture the color 100 times
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

    # take one more image
    frame = cam.capture_array()

    return average_base_color
