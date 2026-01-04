import sys
import os
import cv2

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectRoot)

from logic.vodProcessor.targetDetection import TargetDetector
from logic.vodProcessor.videoUtils import loadTrackingData
from logic.analysis.metrics import analyzeFlickAccuracyWithTargets

# Initialize YOLO detector
print("Loading YOLO model...")
detector = TargetDetector('yolov8n.pt')

# Test on one video first
videoPath = "med11vandal.mp4"
trackingPath = "exports/tracking_0.11_vandal.json"

print(f"Loading tracking data from {trackingPath}...")
positions = loadTrackingData(trackingPath)

print(f"Analyzing flick accuracy for {videoPath}...")
stats = analyzeFlickAccuracyWithTargets(videoPath, positions, detector)

if stats:
    print("\n=== FLICK ACCURACY ANALYSIS ===")
    print(f"Total flicks analyzed: {stats['totalFlicks']}")
    print(f"\nOvershoot: {stats['overshootCount']} ({stats['overshootPercent']:.1f}%)")
    print(f"  Average overshoot: {stats['avgOvershootError']:.1f} pixels")
    print(f"\nUndershoot: {stats['undershootCount']} ({stats['undershootPercent']:.1f}%)")
    print(f"  Average undershoot: {stats['avgUndershootError']:.1f} pixels")
    print(f"\nOn-target: {stats['onTargetCount']} ({stats['onTargetPercent']:.1f}%)")
else:
    print("No flicks detected or analyzed")