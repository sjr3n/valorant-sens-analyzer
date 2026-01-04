import sys
import os
import numpy as np

# Add project root to Python path
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectRoot)

from logic.vodProcessor.videoUtils import loadTrackingData
from logic.analysis.metrics import CrosshairMetrics
import json

# Load tracking data
trackingData = loadTrackingData("tracking_data.json")

# Calculate metrics
metrics = CrosshairMetrics(trackingData)

# Get summary
summary = metrics.getSummary()

print("\n=== CROSSHAIR ANALYSIS ===")
print(f"Total frames tracked: {summary['totalFrames']}")
print(f"Total distance traveled: {summary['totalDistance']:.2f} pixels")
print(f"Average velocity: {summary['averageVelocity']:.2f} px/s")
print(f"Max velocity: {summary['maxVelocity']:.2f} px/s")
print(f"Smoothness score: {summary['smoothness']:.2f} (lower = smoother)")

print(f"\n=== FLICKS ===")
print(f"Total flicks detected: {summary['flicks']['count']}")
if summary['flicks']['count'] > 0:
    print(f"Average flick distance: {summary['flicks']['averageDistance']:.2f} pixels")
    print(f"Average flick velocity: {summary['flicks']['averageVelocity']:.2f} px/s")
    print(f"Max flick distance: {summary['flicks']['maxDistance']:.2f} pixels")

print(f"\n=== TRACKING ===")
print(f"Tracking segments detected: {summary['trackingSegmentCount']}")
print(f"Total tracking distance: {summary['totalTrackingDistance']:.2f} pixels")

# NEW: Velocity distribution analysis
print("\n=== VELOCITY DISTRIBUTION ===")
velocities = [m["velocity"] for m in metrics.movements]
if velocities:
    print(f"Min velocity: {min(velocities):.2f} px/s")
    print(f"25th percentile: {np.percentile(velocities, 25):.2f} px/s")
    print(f"Median velocity: {np.percentile(velocities, 50):.2f} px/s")
    print(f"75th percentile: {np.percentile(velocities, 75):.2f} px/s")
    print(f"95th percentile: {np.percentile(velocities, 95):.2f} px/s")
    print(f"Max velocity: {max(velocities):.2f} px/s")

# Check flicks at different thresholds
print("\n=== FLICK DETECTION AT DIFFERENT THRESHOLDS ===")
for threshold in [100, 200, 300, 500, 750, 1000, 2000]:
    flicks = metrics.detectFlicks(velocityThreshold=threshold)
    print(f"Threshold {threshold:4d} px/s: {len(flicks):3d} flicks detected")

# Save summary to JSON
with open("analysis_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to analysis_summary.json")