from ultralytics import YOLO
import numpy as np                

class AdaptiveYOLODetector:                
    def __init__(self, model_path, imgsz=384, conf_thres=0.3):                
        print(f"Loading YOLO: {model_path}")                
        self.model = YOLO(model_path)                
        self.model.fuse() # Pi optimization                
        self.imgsz = imgsz                
        self.conf_thres = conf_thres                
        
        # Adaptive Skipping State                
        self.frame_idx = 0                
        self.every_n_search = 1                
        self.every_n_track = 6                
        self.no_det_frames = 0                
        
        self.last_results = (None, None) # (bboxes, confs)                
        
    def detect(self, frame, tracking_established=False):                
        """Adaptive detection loop."""                
        self.frame_idx += 1                
        
        # Define current mode frequency                
        mode_n = self.every_n_track if tracking_established else self.every_n_search                
        
        run_yolo = (self.frame_idx % mode_n == 0)                
        
        # Lost detection safety                
        if self.last_results[0] is None or not any(b is not None for b in self.last_results[0].values()):                
            self.no_det_frames += 1                
        else:                
            self.no_det_frames = 0                
            
        if self.no_det_frames > 10:                
            run_yolo = True                
            self.frame_idx = 0                
            
        if run_yolo:                
            results = self.model(                
                frame, verbose=False, imgsz=self.imgsz,                
                conf=self.conf_thres, iou=0.5                
            )                
            bboxes, confs = self._process_results(results[0])                
            self.last_results = (bboxes, confs)                
            return bboxes, confs, True                
        else:                
            return self.last_results[0], self.last_results[1], False                

    def _process_results(self, result):                
        # ... logic to map box class IDs to target names ...                
        # (This depends on OBJECT_CONFIGS, handled in orchestrator)                
        return {}, {} # Placeholder - orchestration file will pass target maps