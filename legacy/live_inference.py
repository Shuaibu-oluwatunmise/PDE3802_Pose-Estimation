#!/usr/bin/env python3
"""
Simple Live Inference Script - Minimal version
Quick start for real-time detection
"""

import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
CONF_THRESHOLD = 0.5
SOURCE = 1  # 0 for webcam, or path to video file

# Load model
print("Loading model...")
model = YOLO(MODEL_PATH)
print(f"Model loaded! Classes: {model.names}")

# Open video source
cap = cv2.VideoCapture(SOURCE)

print("\nRunning inference... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    
    # Get annotated frame
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('YOLO Live Detection', annotated_frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()