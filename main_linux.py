#!/usr/bin/env python3

#IMPORTS
import numpy as np
import cv2
import sys
import time
import platform
import libcamera
from picamera2 import Picamera2


def to_continue():
    """
    Prompts the user to start a new sampling session and returns a Boolean indicating their choice.

    This function displays a prompt asking if the user would like to start a new 5-second sampling session.
    If the user inputs 'y', 'Y', or simply presses Enter without typing anything, the function returns True,
    signaling that the program should continue with a new sampling session. If the user inputs 'n' or any
    other character, the function returns False, halting the sampling process.

    Returns:
        bool: True if the user chooses to start a new sampling session ('y', 'Y', or no input); 
              False otherwise.
    """
    answer = input("Start sampling for 5 seconds [y/n]: ")
    if answer == "y" or answer == "Y" or len(answer) == 0:
        return True
    else:
        return False


def calibration(cam):
    """
    Calibrates the camera by capturing live video frames and displaying a calibration window,
    allowing the user to input minimum and middle distance values in inches from the origin (i.e table).

    This function configures the camera with specific video settings, initiates a live preview with
    overlayed calibration lines, and prompts the user to enter minimum and middle distance values
    (in inches) for calibration purposes. The function returns these user-defined values for further
    use in the application.

    Parameters:
        cam: Camera object
            An object that interfaces with the camera, providing methods for configuring,
            starting, stopping, and capturing frames.

    Configuration:
        - Sets up main video configuration (resolution, format) and sensor properties.
        - Creates a live preview with calibration lines drawn vertically at the start, middle, 
          and end of the frame width.

    Controls:
        - The user can press 'q' to exit the preview window once calibration is complete.

    Returns:
        tuple: (min_value, mid_value)
            min_value (float): The user-defined minimum calibration value in inches.
            mid_value (float): The user-defined middle calibration value in inches.
    """

    main = {"size": (width, height), "format": "RGB888"}
    controls = {"FrameRate": framerate}
    sensor = {"bit_depth": 10, "output_size": (640, 480)}
    video_config = cam.create_video_configuration(
        main, controls=controls, sensor=sensor
    )
    cam.configure(video_config)

    # Start the preview
    cam.start()
    
    # Start timing measurements
    start_time = time.time()
    for frame_count in range(1, 1000):
        
        frame = cam.capture_array()
          
        if frame is None:
            break

        if frame_count % 100 == 0:
          end_time = time.time()
          elapsed_time = end_time - start_time
          print("frames per second: ", int(frame_count/elapsed_time))

    print("\n\nDetermine minimum value (cm or inches) at the red line.")
    print("Determine middle value (cm or inches) at the green line.")
    print("Hit 'q' in the calibration window when done.\n\n")

    # Start Calibration:
    while True:
        frame = cam.capture_array()

        if frame is None:
            break

        # Draw calibration lines
        cv2.line(frame, (0, 0), (0, 480), (0, 0, 255), 3)  # Red line at beginning
        cv2.line(frame, (320, 0), (320, 480), (0, 255, 0), 3)  # Green line in middle
        cv2.line(frame, (640, 0), (640, 480), (255, 0, 0), 3)  # Blue line at end
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(40) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    # Stop the camera
    cam.stop()

    min_value = float(input("Enter the minimum value in inches: "))
    mid_value = float(input("Enter the mid value in inches: "))

    return min_value, mid_value


def shoot_video(cam, min_value, mid_value):
    """
    Captures a 5-second video, processes frames to detect a moving object,
    and estimates the distance of the object's position in inches.

    This function operates the camera in a loop, capturing real-time frames and detecting a ping pong ball.
    It uses a background subtraction technique to isolate the moving object, then draws contours and identifies
    the object's center. After each 5-second session, the function computes and displays the object's horizontal
    position in inches based on the calibration values.

    Parameters:
        cam: Camera object
            Provides methods for starting, stopping, and capturing frames.
        min_value (float): The minimum calibration distance in inches.
        mid_value (float): The mid calibration distance in inches.

    Process:
        - Repeats until user chooses to stop.
        - Captures a grayscale background image for background subtraction.
        - Captures frames in real-time (~90fps), processes them to detect and mark moving objects.
        - Uses contours to locate the object's position and calculates its coordinates if certain
          criteria are met (area and aspect ratio).
        - Outputs processed frames with detected contours to video files.

    Outputs:
        - Three video files in the 'output' directory:
            - "ping_pong_raw_input.avi": Raw video.
            - "ping_pong_with_detection.avi": Video with contours and detection.
            - "ping_pong_bw_with_detection.avi": Black and white video with detection overlay.
        - Prints the estimated distance in inches if the object is successfully tracked.

    Controls:
        - Press 'q' to stop the preview or terminate the recording early.

    Returns:
        None
    """
    while to_continue():

        # Start the camera
        cam.start()

        frames = []

        # Store the background image
        bg_img = cam.capture_array()
        #get the base frame to later compare with 
        #convert the frame to b&w, reducing computational work 
        bg_img_bw = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)

        start_time = time.time()

        # capture images in real time ~ 90fps
        while time.time() - start_time < 5:
            frame = cam.capture_array()
            frames.append(frame)

        for frame in frames:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(40) & 0xFF
            if key == ord("q"):
                break
        
        #POST--PROCESSING
        centers = []

        X, Y = None, None
        y_prev = None
        x_prev = None

        # Define codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        inp = cv2.VideoWriter(
            "output/ping_pong_raw_input.avi", fourcc, framerate, (width, height)
        )
        out = cv2.VideoWriter(
            "output/ping_pong_with_detection.avi", fourcc, framerate, (width, height)
        )
        out_bw = cv2.VideoWriter(
            "output/ping_pong_bw_with_detection.avi", fourcc, framerate, (width, height)
        )

        #Processing all the frames,one-by-one
        for frame in frames:
            og_frame = frame.copy()
            #convert the current frame to b&w, reducing computational work
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #taking absolute difference between the first converted b&w frame with the current frame
            img = cv2.absdiff(gray, bg_img_bw)
            img = cv2.blur(img, (3, 3))
            ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)

            #draw contours around the differences
            contours, _ = cv2.findContours(
                img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            #find the area of all the contours
            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = 100
                
                #draw a circle/track the deformed shape, only if its area and aspect_ratio > threshold
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    max_aspect_ratio = 3.0

                    if aspect_ratio <= max_aspect_ratio:
                        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centers.append((cx, cy))

                            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                            print("Centers:", cx, cy)

                            # Setting the y_prev for the first time
                            if y_prev is None:
                                y_prev = cy
                                x_prev = cx

                            # Changing the y_prev if the y is increasing
                            if cy >= y_prev:
                                y_prev = cy
                                x_prev = cx

                            # Y is decreasing and X and Y both are not been found yet
                            # store values for first minimum in Y
                            elif cy < y_prev and X is None and Y is None:
                                X = x_prev
                                Y = y_prev

            cv2.imshow("Frame", frame)
            cv2.imshow("Image", img)
            cv2.imshow("Raw", og_frame)

            # write images to file
            inp.write(og_frame)
            out.write(frame)
            out_bw.write(img)

            key = cv2.waitKey(40) & 0xFF
            if key == ord("q"):
                break

        if X == None and Y == None:
            print(
                "Was not able to track the ping pong ball.  Try moving the sensor further away from the target"
            )
            print("increasing the lighting or lowering the threshold.")
        else:
            print("Found them:", X, Y)
            x_inches = min_value + (mid_value - min_value) / (width / 2) * X
            print("Distance in inches: ", x_inches)

        cam.stop()
        cv2.destroyAllWindows()
        inp.release()
        out.release()
        out_bw.release()


def main():
    """
    Main function to initialize the camera, perform calibration, and start video capture.

    This function serves as the entry point of the program, initializing the PiCamera2, 
    conducting a calibration process to obtain minimum and middle distance values in inches,
    and initiating a 5-second video capture and processing loop for object detection and tracking.

    Process:
        - Initializes the PiCamera2.
        - Calls the `calibration` function to open a calibration window and prompts the user
          for minimum and mid-point calibration values in inches.
        - Passes these calibration values to the `shoot_video` function, which captures and processes
          real-time frames to track an object and estimate its position in inches.

    Functions:
        - `calibration(cam)`: Calibrates the camera and returns user-defined min and mid values.
        - `shoot_video(cam, min_value, mid_value)`: Captures and processes video using calibration values.

    Returns:
        None
    """
    # Initialize the PiCamera2
    cam = Picamera2()
    min_value, mid_value = calibration(cam)
    
    # Function to shoot the video
    shoot_video(cam, min_value, mid_value)



# Set camera properties
framerate = 90  # Frames per second for video capture
width = 640     # Width of video frame
height = 480    # Height of video frame

if __name__ == "__main__":
    """
    Main script entry point.

    Sets camera properties such as framerate, frame width, and height, then
    checks if the operating system is compatible with the program.
    If running on a Linux system (e.g., Raspberry Pi), it initializes and starts
    the main process with the PiCamera2 for calibration and video capture.
    
    Compatibility:
        - Linux OS required (e.g., Raspberry Pi OS).
        - On non-Linux systems, a message is displayed indicating OS incompatibility.
    
    Functions:
        - main(): Initializes the PiCamera2, performs calibration, and starts video capture.

    """
    if platform.system() == "Linux":
        main()
    else:
        print("OS not compatible")
