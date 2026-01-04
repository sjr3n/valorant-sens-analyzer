import json
import os

def saveTrackingData(positions, outputPath):
    """
    Save tracking data to JSON file.
    
    Args:
        positions: List of position dictionaries from trackCrosshairInVideo
        outputPath: Path to save JSON file
    """
    with open(outputPath, 'w') as f:
        json.dump(positions, f, indent=2)
    
    print(f"Saved tracking data to {outputPath}")


def loadTrackingData(jsonPath):
    """
    Load tracking data from JSON file.
    
    Args:
        jsonPath: Path to JSON file
    
    Returns:
        List of position dictionaries
    """
    if not os.path.exists(jsonPath):
        raise FileNotFoundError(f"Tracking data not found: {jsonPath}")
    
    with open(jsonPath, 'r') as f:
        positions = json.load(f)
    
    print(f"Loaded {len(positions)} positions from {jsonPath}")
    return positions


def getVideoMetadata(videoPath):
    """
    Extract basic metadata from video file.
    
    Args:
        videoPath: Path to video file
    
    Returns:
        Dictionary with video metadata (fps, width, height, duration, frameCount)
    """
    import cv2
    
    cap = cv2.VideoCapture(videoPath)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {videoPath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frameCount / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frameCount": frameCount,
        "duration": duration
    }