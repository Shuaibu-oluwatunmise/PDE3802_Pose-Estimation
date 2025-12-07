from ultralytics import YOLO
import cv2
import time

try:
    print("Loading NCNN model via Ultralytics...")
    # Point to the DIRECTORY containing .param and .bin
    model = YOLO("runs/detect/yolov8n_detect_V2/weights/best_ncnn_model") 
    print("Model loaded!")
    
    # Create dummy image
    img = cv2.imread("runs/detect/train/val_batch0_labels.jpg") # Try to find a real image or make one
    if img is None:
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    print("Running inference...")
    start = time.time()
    results = model(img)
    end = time.time()
    print(f"Inference time: {(end-start)*1000:.2f}ms")
    
    for r in results:
        print(f"Boxes: {len(r.boxes)}")

except Exception as e:
    print(f"FAILED: {e}")
