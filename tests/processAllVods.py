import sys
import os
import numpy as np

# Add project root to Python path
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectRoot)

from logic.vodProcessor.crosshairTracker import trackCrosshairInVideo, smoothPositions, visualizeCrosshairPath
from logic.vodProcessor.videoUtils import saveTrackingData, getVideoMetadata

# Create exports folder if it doesn't exist
exportsFolder = "exports"
if not os.path.exists(exportsFolder):
    os.makedirs(exportsFolder)

# Crosshair color bounds - GREEN
lowerBound = np.array([0, 150, 0])
upperBound = np.array([150, 255, 150])

# VODs to process with their sensitivities and weapons
vods = [
    {"file": "med11vandal.mp4", "sens": 0.11, "weapon": "vandal"},
    {"file": "med11sheriff.mp4", "sens": 0.11, "weapon": "sheriff"},
    {"file": "med145vandal.mp4", "sens": 0.145, "weapon": "vandal"},
    {"file": "med23vandal.mp4", "sens": 0.23, "weapon": "vandal"},
    {"file": "med23sheriff.mp4", "sens": 0.23, "weapon": "sheriff"}
]

# Processing parameters
sampleRate = 1
roiSize = 800

print("=== PROCESSING ALL VODs ===\n")

for vod in vods:
    videoPath = vod["file"]
    sensitivity = vod["sens"]
    weapon = vod["weapon"]
    
    print(f"\n{'='*60}")
    print(f"Processing: {videoPath}")
    print(f"Sensitivity: {sensitivity} | Weapon: {weapon}")
    print(f"{'='*60}")
    
    try:
        # Get video metadata
        metadata = getVideoMetadata(videoPath)
        print(f"Resolution: {metadata['width']}x{metadata['height']}")
        print(f"FPS: {metadata['fps']}")
        print(f"Duration: {metadata['duration']:.2f} seconds\n")
        
        # Track crosshair
        positions = trackCrosshairInVideo(
            videoPath,
            lowerBound,
            upperBound,
            sampleRate=sampleRate,
            roiSize=roiSize
        )
        
        if positions:
            # Apply smoothing
            smoothedPositions = smoothPositions(positions, windowSize=5)
            
            print(f"Tracked {len(smoothedPositions)} frames")
            
            # Save tracking data to exports folder with weapon info
            outputJson = os.path.join(exportsFolder, f"tracking_{sensitivity}_{weapon}.json")
            saveTrackingData(smoothedPositions, outputJson)
            
            # Create visualization in exports folder
            outputVideo = os.path.join(exportsFolder, f"visualized_{sensitivity}_{weapon}.mp4")
            visualizeCrosshairPath(videoPath, smoothedPositions, outputVideo)
            
            print(f"✓ Saved to {outputJson} and {outputVideo}")
        else:
            print(f"✗ No crosshair positions detected!")
    
    except FileNotFoundError:
        print(f"✗ Error: Could not find {videoPath}")
    except Exception as e:
        print(f"✗ Error: {e}")

print(f"\n{'='*60}")
print("All VODs processed!")
print(f"{'='*60}")