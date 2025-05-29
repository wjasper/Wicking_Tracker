import cv2
import numpy as np


class BoundingBox:
    def __init__(self, x=150, y=100, w=300, h=200):
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
        sensitivity = 15

        top_left = (self.x, self.y)
        top_right = (self.x + self.w, self.y)
        bottom_left = (self.x, self.y + self.h)
        bottom_right = (self.x + self.w, self.y + self.h)

        if event == cv2.EVENT_LBUTTONDOWN:
            if abs(x - top_left[0]) < sensitivity and abs(y - top_left[1]) < sensitivity:
                self.resize_corner = "top_left"
                self.dragging = True
            elif abs(x - top_right[0]) < sensitivity and abs(y - top_right[1]) < sensitivity:
                self.resize_corner = "top_right"
                self.dragging = True
            elif abs(x - bottom_left[0]) < sensitivity and abs(y - bottom_left[1]) < sensitivity:
                self.resize_corner = "bottom_left"
                self.dragging = True
            elif abs(x - bottom_right[0]) < sensitivity and abs(y - bottom_right[1]) < sensitivity:
                self.resize_corner = "bottom_right"
                self.dragging = True
            elif self.x < x < self.x + self.w and self.y < y < self.y + self.h:
                self.resize_corner = "move"
                self.dragging = True
                self.drag_start_x = x - self.x
                self.drag_start_y = y - self.y

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
    bbox = BoundingBox()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", bbox.handle_mouse, {"height": height, "width": width})
    instructions = "Drag corners to resize, drag center to move. Press 'q' to quit."
    found_initial_bbox = False

    while True:
        frame = cam.capture_array()
        if frame is None:
            break

        if not found_initial_bbox and not bbox.dragging:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                if w > 50 and h > 50:
                    bbox.x, bbox.y, bbox.w, bbox.h = x, y, w, h
                    found_initial_bbox = True

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

        cv2.imshow("Calibration", frame)
        key = cv2.waitKeyEx(40)

        if key == 56 and bbox.y > 0:
            bbox.y -= 2
            bbox.h += 2
        elif key == 50 and bbox.h > 50:
            bbox.y += 2
            bbox.h -= 2
        elif key == 52 and bbox.x > 0:
            bbox.x -= 2
            bbox.w += 2
        elif key == 54 and bbox.w > 50:
            bbox.x += 2
            bbox.w -= 2
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    height_in_mm = int(input("Enter reading corresponding to the box in mm: "))
    bbox.mm_per_pixel = height_in_mm / bbox.h
    print("mm_per_pixel", bbox.mm_per_pixel)

    return (bbox.x, bbox.y, bbox.w, bbox.h, height_in_mm, bbox.mm_per_pixel)


def base_color(cam, bbox_x, bbox_y, bbox_w, bbox_h):
    cv2.namedWindow("Getting average over 500 frames")
    base_colors = []

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

    return average_base_color