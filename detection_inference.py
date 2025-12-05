# Webcam_Gstreamer_Detector.py
import cv2
import threading
import time
import queue
import os
from ultralytics import YOLO
from datetime import datetime

# ==============================================================================
# GStreamer Pipeline Configuration for high-performance camera access
# NOTE: This pipeline string is adapted from your provided file, which typically
# uses 'nvarguscamerasrc' for NVIDIA Jetson CSI cameras. If you are on a standard 
# Raspberry Pi, you may need to modify this string to use 'libcamerasrc' or 
# a 'v4l2src' pipeline for your specific camera module.
# ==============================================================================
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink drop=true"
)

class WebcamDetector:
    """
    Live Object Detection using YOLOv8, leveraging a separate thread for inference
    and a GStreamer pipeline for efficient camera input on embedded systems.
    """
    def __init__(self):
        # Load your trained YOLO detection model
        # UPDATED: Using the user-specified model path
        self.model = YOLO('runs/detect/yolov8n_detect_V2/weights/best.pt') 
        print("INFO: Using custom trained model: 'runs/detect/yolov8n_detect_V2/weights/best.pt'")
        
        # Video settings
        self.cap = None
        self.recording = False
        self.video_writer = None
        # Queue to pass frames from main thread (capture) to worker thread (detection)
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Detection settings
        self.latest_detections = []
        self.last_detection_time = 0
        self.detection_interval = 0.1  # 100ms interval for detection
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processing_time = 0
        
        # Create output directory
        os.makedirs("Webcam_Detections", exist_ok=True)
        
    def setup_webcam(self):
        """
        Initialize webcam connection using the GStreamer pipeline.
        """
        print("Initializing GStreamer webcam pipeline...")
        
        # Use cv2.CAP_GSTREAMER to process the pipeline string
        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam via GStreamer pipeline.")
            print("Check GStreamer plugins, permissions, and pipeline string.")
            return False
        
        # Read a few frames to confirm stability
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"Webcam connected via GStreamer: {w}x{h}")
                return True
            time.sleep(0.1)
        
        print("Error: Webcam connected but cannot read stable frames.")
        return False
    
    def start_recording(self):
        """Start recording session"""
        # Get a frame for dimensions
        ret, frame = self.cap.read() 
        
        if not ret or frame is None:
            print("Error: Could not get stable frame for recording")
            return False
            
        height, width = frame.shape[:2]
        fps = 30.0 # Match the target FPS in the GStreamer pipeline
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Webcam_Detections/session_{timestamp}.mp4"
        
        # Use MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not self.video_writer or not self.video_writer.isOpened():
            print("Error: Could not initialize video writer. Trying alternative codec.")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            if not self.video_writer.isOpened():
                 print("Error: Could not initialize video writer with any standard codec.")
                 self.video_writer = None
                 return False
            
        self.recording = True
        print(f"Recording started: {filename}")
        print(f"Resolution: {width}x{height} at {fps} FPS (Target)")
        return True
    
    def stop_recording(self):
        """Stop recording session"""
        if self.video_writer:
            self.video_writer.release()
            self.recording = False
            self.video_writer = None
            print("Recording stopped and saved")
    
    def detection_worker(self):
        """Separate thread for object detection, running inference on frames from the queue."""
        while True:
            current_time = time.time()
            
            # Only detect at the specified interval
            if current_time - self.last_detection_time >= self.detection_interval:
                try:
                    # Get the most recent frame, discarding older ones
                    if not self.frame_queue.empty():
                        while self.frame_queue.qsize() > 1:
                            self.frame_queue.get_nowait()
                        frame = self.frame_queue.get_nowait()
                        
                        # Run object detection
                        start_time = time.time()
                        results = self.model(frame, verbose=False, persist=True) 
                        self.processing_time = time.time() - start_time
                        
                        # Process detection results (Bounding Box only)
                        detections = []
                        result = results[0]
                        
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()  
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy().astype(int)
                            class_names = self.model.names
                            
                            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                                detections.append({
                                    'box': box,
                                    'confidence': float(conf),
                                    'class_name': class_names[cls_id],
                                    'class_id': int(cls_id)
                                })
                        
                        self.latest_detections = detections
                        self.last_detection_time = current_time
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Detection worker error: {e}")
            
            time.sleep(0.005) # Small sleep to yield CPU
    
    def draw_detections(self, frame):
        """Draw bounding boxes and labels on frame"""
        # Simple color palette
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for detection in self.latest_detections:
            box = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            x1, y1, x2, y2 = map(int, box)
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background
            cv2.rectangle(frame, 
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def add_annotations(self, frame):
        """Add performance info and instructions to frame"""
        height, width = frame.shape[:2]
        font_scale = max(0.8, min(1.5, width / 800))
        thickness = max(2, int(font_scale * 1.5))
        
        # Performance info
        objects_count = len(self.latest_detections)
        perf_text = (f"FPS: {self.fps:.1f} | "
                     f"Detect Rate: {1/self.detection_interval:.1f} Hz | "
                     f"Process: {self.processing_time*1000:.0f}ms | "
                     f"Objects: {objects_count} | "
                     f"Rec: {'ON' if self.recording else 'OFF'}")
        
        # Draw background and text for performance info (Top Left)
        perf_size = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, thickness)[0]
        cv2.rectangle(frame, (10, 10), (20 + perf_size[0], 20 + perf_size[1]), (0, 0, 0), -1)
        cv2.putText(frame, perf_text, (15, 15 + perf_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), thickness)
        
        # Instructions (Bottom Left)
        instr_text = "SPACE: Toggle Recording | R: Restart Recording | Q: Quit"
        instr_size = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness-1)[0]
        cv2.rectangle(frame, (10, height - instr_size[1] - 20), (20 + instr_size[0], height - 10), (0, 0, 0), -1)
        cv2.putText(frame, instr_text, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), thickness-1)
        
        return frame
    
    def calculate_fps(self):
        """Calculate and update FPS (Main Video Loop FPS)"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def run(self):
        """Main application loop"""
        if not self.setup_webcam():
            return
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        
        # Start recording immediately
        self.start_recording()
        
        print("\n" + "="*60)
        print("Webcam Live Object Detection (GStreamer) Started!")
        print(f"Video Stream: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self.cap.get(cv2.CAP_PROP_FPS):.0f} FPS")
        print(f"Detection: Every {self.detection_interval*1000:.0f}ms (Approx. {1/self.detection_interval:.1f} Hz)")
        print("Controls: SPACE to toggle recording | R to restart recording | Q to quit")
        print("="*60 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Error: Could not read frame from webcam. Exiting loop.")
                    break
                
                self.calculate_fps()
                
                # Add current frame to detection queue
                if self.frame_queue.qsize() < 2:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                
                frame_with_detections = self.draw_detections(frame)
                annotated_frame = self.add_annotations(frame_with_detections)
                
                # Write to video file if recording
                if self.recording and self.video_writer and annotated_frame is not None:
                    self.video_writer.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Webcam - Live Object Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # SPACE to toggle recording
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('r'):  # R to restart recording
                    if self.recording:
                        self.stop_recording()
                    self.start_recording()
                elif key == ord('q'):  # Q to quit
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Cleanup
            if self.recording:
                self.stop_recording()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Application cleanup complete. Application closed.")

# Run the application
if __name__ == "__main__":
    detector = WebcamDetector()
    detector.run()