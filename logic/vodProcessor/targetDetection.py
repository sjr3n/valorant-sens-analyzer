import cv2
import numpy as np
from ultralytics import YOLO

class TargetDetector:
    """
    Detect player/bot heads in Valorant footage using YOLO.
    """
    
    def __init__(self, modelPath='yolov8n.pt'):
        """
        Initialize YOLO model.
        
        Args:
            modelPath: Path to YOLO model weights
                      'yolov8n.pt' = nano (fastest)
                      'yolov8s.pt' = small
                      'yolov8m.pt' = medium (more accurate)
        """
        self.model = YOLO(modelPath)
        
    def detectHeads(self, frame, confidenceThreshold=0.5):
        """
        Detect heads/people in a frame.
        
        Args:
            frame: BGR image from video
            confidenceThreshold: Minimum confidence for detection
        
        Returns:
            List of (x, y, width, height, confidence) for each detected head
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class (0 = person in COCO dataset)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Filter for 'person' class with sufficient confidence
                if cls == 0 and conf >= confidenceThreshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate center and dimensions
                    centerX = int((x1 + x2) / 2)
                    centerY = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Estimate head position (top 20% of person bounding box)
                    headY = int(y1 + height * 0.15)
                    
                    detections.append({
                        'x': centerX,
                        'y': headY,
                        'width': width,
                        'height': height,
                        'confidence': conf
                    })
        
        return detections
    
    def findNearestTarget(self, crosshairPos, targets, maxDistance=300):
        """
        Find the nearest target to crosshair position.
        
        Args:
            crosshairPos: (x, y) tuple of crosshair
            targets: List of target dictionaries from detectHeads
            maxDistance: Maximum distance to consider a match
        
        Returns:
            Target dict with added 'distance' key, or None
        """
        if not targets:
            return None
        
        nearestTarget = None
        nearestDistance = float('inf')
        
        for target in targets:
            distance = np.sqrt(
                (crosshairPos[0] - target['x'])**2 + 
                (crosshairPos[1] - target['y'])**2
            )
            
            if distance < nearestDistance and distance < maxDistance:
                nearestDistance = distance
                nearestTarget = target.copy()
                nearestTarget['distance'] = distance
        
        return nearestTarget