#!/usr/bin/env python3
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import cv2
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
from skimage import measure
import csv
import drivers
import time
import os

# Set GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Camera setup
picam2 = Picamera2()

# GPIO button pins
start_button = 16
shutdown_button = 21
relay_pin1 = 24  # Light
relay_pin2 = 25  # LCD

# GPIO setup for buttons
GPIO.setup(start_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(shutdown_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(relay_pin1, GPIO.OUT)
GPIO.setup(relay_pin2, GPIO.OUT)

# Image path
image_path = '/home/celerity/dp2_esd/final/image.jpg'

# Turn on LCD
GPIO.output(relay_pin2, GPIO.HIGH)

# Display setup
display = drivers.Lcd()

# CNN Model
cnn = YOLO('/home/celerity/dp2_esd/final/celerity_cnn.pt')

# SVM Model
svm = joblib.load('/home/celerity/dp2_esd/final/celerity_svm.joblib')
scaler = joblib.load('/home/celerity/dp2_esd/final/celerity_scaler.joblib')

# Camera configuration
camera_config = picam2.create_still_configuration(main={"size": (3280, 2464)})
picam2.configure(camera_config)

def detect_eggs():
    with open("/home/celerity/dp2_esd/final/celerity_coco.txt", "r") as f:
        class_list = f.read().split("\n")

    # Capture image
    GPIO.output(relay_pin1, GPIO.HIGH)  # Light ON
    picam2.start()
    time.sleep(2)
    picam2.capture_file(image_path)
    picam2.stop()
    GPIO.output(relay_pin1, GPIO.LOW)  # Light OFF
    
    # Read captured image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Define partitions
    top_left = image[:height // 2, :width // 2]
    top_right = image[:height // 2, width // 2:]
    bottom_left = image[height // 2:, :width // 2]
    bottom_right = image[height // 2:, width // 2:]
    
    partitions = {
        "Top Left": top_left,
        "Top Right": top_right,
        "Bottom Left": bottom_left,
        "Bottom Right": bottom_right
    }

    display.lcd_clear()
    display.lcd_display_string("  Processing data ", 2)
    
    detected_eggs = {}
    boxes_list = []
    
    for partition_name, partition_image in partitions.items():
        # Image preprocessing
        gray_image = cv2.cvtColor(partition_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 212, 255, cv2.THRESH_BINARY)
        cnn_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        
        # CNN egg detection
        result = cnn.predict(cnn_image)
        boxes = result[0]
        num_boxes = len(boxes)
        if num_boxes > 1:
            num_boxes = 1
        detected_eggs[partition_name] = num_boxes
        boxes_list.append(num_boxes)
    
    boxes_detected = sum(boxes_list)
    if boxes_detected > 4:
        boxes_detected = 4
        
    return top_left, top_right, bottom_left, bottom_right, detected_eggs, boxes_detected

def dsp(partition):    

    # RGB to Binary
    partition = cv2.convertScaleAbs(partition, alpha=1.6, beta=0)
    gray = cv2.cvtColor(partition, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, binary = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY)
    kernel = np.ones((50, 50), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # Erosion + Dilation
    
    # Alignment Correction
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(max_contour)
    angle = ellipse[2]
    
    if angle > 45:
        angle -= 90
    
    rows, cols, _ = partition.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_binary = cv2.warpAffine(binary, rotation_matrix, (cols, rows))
    
    # Contour Extraction
    canny = cv2.Canny(rotated_binary, 30, 150, 10)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    threshold = 400
    bbox_dim = None
    if contour_area > (threshold*10**3):
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(partition, (x, y), (x+w, y+h), (0, 0, 255), 2)
        bbox_dim = (w, h)

    return bbox_dim

def extract_features(w, h):
	si = h / w
	ec = math.sqrt(1 - si ** 2)
	gd = (w * h ** 2) ** (1 / 3)
	sp = (gd / w) * 100
	features = (si, ec, sp)
	return features
	
def predict(features):
    features = np.reshape(features, (1, -1))
    features = scaler.transform(features)
    egg = svm.predict(features)
    if egg == 1:
        egg = 'M'
    else:
        egg = 'F'
    return egg

# Start button
def execute_start_button():
    # LCD
    display.lcd_clear()
    display.lcd_display_string("  Capturing image ", 2)
    
    # Detect eggs
    top_left, top_right, bottom_left, bottom_right, detected_eggs, boxes_detected = detect_eggs()    
    start_time = time.time() # Start time
    
    if all(value == 0 for value in detected_eggs.values()):
        display.lcd_clear()
        display.lcd_display_string("  No Eggs Detected  ", 2)
        display.lcd_display_string("       Restart      ", 4)
    else:
        try:
            if detected_eggs["Top Left"] == 1:
                w1, h1 = dsp(top_left)
                features1 = extract_features(w1, h1)
                egg1 = predict(features1)
                #print(w1, h1)
                #print(features1)
                #print(egg1)
            else:
                egg1 = '0'
        except Exception as e:
            print("Error processing Egg 1", e)
            egg1 = 'E'

        try:
            if detected_eggs["Top Right"] == 1:
                w2, h2 = dsp(top_right)
                features2 = extract_features(w2, h2)
                egg2 = predict(features2)
                #print(w2, h2)
                #print(features2)
                #print(egg2)
            else:
                egg2 = '0'
        except Exception as e:
            print("Error processing Egg 2:", e)
            egg2 = 'E'

        try:
            if detected_eggs["Bottom Left"] == 1:
                w3, h3 = dsp(bottom_left)
                features3 = extract_features(w3, h3)
                egg3 = predict(features3)
                #print(w3, h3)
                #print(features3)
                #print(egg3)
            else:
                egg3 = '0'
        except Exception as e:
            print("Error processing Egg 3:", e)
            egg3 = 'E'

        try:
            if detected_eggs["Bottom Right"] == 1:
                w4, h4 = dsp(bottom_right)
                features4 = extract_features(w4, h4)
                egg4 = predict(features4)
                #print(w4, h4)
                #print(features4)
                #print(egg4)
            else:
                egg4 = '0'
        except Exception as e:
            print("Error processing Egg 4:", e)
            egg4 = 'E'
            
        end_time = time.time() # End time
        inf_time = end_time - start_time # Inference time
        display.lcd_clear()
        display.lcd_display_string("  Eggs detected: " +str(boxes_detected) , 2)
        time.sleep(1)
        
        print("Total Inference time: {:.3f} ms".format((inf_time)*1000))
        
        display.lcd_clear()
        display.lcd_display_string("Results  t: {:.2f}ms".format((inf_time)*1000), 1)
        display.lcd_display_string("Egg 1: " + egg1 + "   Egg 2: " + egg2, 3)
        display.lcd_display_string("Egg 3: " + egg3 + "   Egg 4: " + egg4, 4)

# Shutdown button
def execute_shutdown_button(channel):
    display.lcd_clear()
    display.lcd_display_string("    Shutting down    ", 2)
    time.sleep(2)
    GPIO.output(relay_pin2, GPIO.LOW)
    os.system("sudo shutdown -h now")

try:
    display.lcd_display_string(" Press start button ", 2)

    while True:
        if GPIO.input(start_button) == GPIO.LOW:
            execute_start_button()
            while GPIO.input(start_button) == GPIO.LOW:
                time.sleep(0.1)

        if GPIO.input(shutdown_button) == GPIO.LOW:
            execute_shutdown_button(shutdown_button)

except KeyboardInterrupt:
    GPIO.cleanup()
