import cv2
import numpy as np

def detectCrosshair(frame, crosshairColorLower, crosshairColorUpper, roiSize=200):
    """
    Detect crosshair position in a single frame using color-based detection.
    Only searches in the center region of the frame to avoid gun skin interference.
    
    Args:
        frame: BGR image from video
        crosshairColorLower: Lower bound for crosshair color (BGR tuple)
        crosshairColorUpper: Upper bound for crosshair color (BGR tuple)
        roiSize: Size of the region of interest (square centered on screen)
    
    Returns:
        (x, y) tuple of crosshair center, or None if not found
    """
    height, width = frame.shape[:2]
    centerX = width // 2
    centerY = height // 2
    
    # Define ROI (Region of Interest) - square box around screen center
    halfRoi = roiSize // 2
    x1 = max(0, centerX - halfRoi)
    y1 = max(0, centerY - halfRoi)
    x2 = min(width, centerX + halfRoi)
    y2 = min(height, centerY + halfRoi)
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    # Create mask for crosshair color in ROI only
    mask = cv2.inRange(roi, crosshairColorLower, crosshairColorUpper)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour (most likely the crosshair)
    largestContour = max(contours, key=cv2.contourArea)
    
    # Calculate the center of the contour
    moments = cv2.moments(largestContour)
    if moments["m00"] == 0:
        return None
    
    # Convert ROI coordinates back to full frame coordinates
    roiCenterX = int(moments["m10"] / moments["m00"])
    roiCenterY = int(moments["m01"] / moments["m00"])
    
    # Add ROI offset to get absolute position
    absoluteX = x1 + roiCenterX
    absoluteY = y1 + roiCenterY
    
    return (absoluteX, absoluteY)

def trackCrosshairInVideo(videoPath, crosshairColorLower, crosshairColorUpper, sampleRate=1, roiSize=200):
    """
    Track crosshair throughout entire video.
    
    Args:
        videoPath: Path to video file
        crosshairColorLower: Lower bound for crosshair color (BGR)
        crosshairColorUpper: Upper bound for crosshair color (BGR)
        sampleRate: Process every Nth frame (1 = every frame, 2 = every other frame)
        roiSize: Size of region of interest for crosshair detection
    
    Returns:
        List of dictionaries: [{"frameNumber": int, "x": int, "y": int, "timestamp": float}, ...]
    """
    cap = cv2.VideoCapture(videoPath)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {videoPath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    crosshairPositions = []
    frameNumber = 0
    
    print(f"Processing video at {fps} FPS...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process frames according to sample rate
        if frameNumber % sampleRate == 0:
            position = detectCrosshair(frame, crosshairColorLower, crosshairColorUpper, roiSize)
            
            if position:
                timestamp = frameNumber / fps
                crosshairPositions.append({
                    "frameNumber": frameNumber,
                    "x": position[0],
                    "y": position[1],
                    "timestamp": timestamp
                })
        
        frameNumber += 1
        
        # Progress indicator
        if frameNumber % 100 == 0:
            print(f"Processed {frameNumber} frames...")
    
    cap.release()
    print(f"Complete! Tracked {len(crosshairPositions)} frames with crosshair detected")
    
    return crosshairPositions

def smoothPositions(positions, windowSize=5):
    """
    Smooth crosshair positions using moving average to reduce jitter.
    
    Args:
        positions: List of position dictionaries from trackCrosshairInVideo
        windowSize: Number of frames to average over
    
    Returns:
        Smoothed positions list
    """
    if len(positions) < windowSize:
        return positions
    
    smoothed = []
    
    for i in range(len(positions)):
        # Get window of positions around current frame
        startIdx = max(0, i - windowSize // 2)
        endIdx = min(len(positions), i + windowSize // 2 + 1)
        
        window = positions[startIdx:endIdx]
        
        # Calculate average position
        avgX = sum(p["x"] for p in window) / len(window)
        avgY = sum(p["y"] for p in window) / len(window)
        
        smoothed.append({
            "frameNumber": positions[i]["frameNumber"],
            "x": int(avgX),
            "y": int(avgY),
            "timestamp": positions[i]["timestamp"]
        })
    
    return smoothed
    
def visualizeCrosshairPath(videoPath, crosshairPositions, outputPath):
    """
    Create a video with crosshair path visualized as a trail.
    
    Args:
        videoPath: Original video path
        crosshairPositions: List of crosshair positions from trackCrosshairInVideo
        outputPath: Where to save the visualization video
    """
    cap = cv2.VideoCapture(videoPath)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {videoPath}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))
    
    frameNumber = 0
    positionIndex = 0
    
    # Store recent positions for trail effect
    recentPositions = []
    trailLength = 30  # Show last 30 positions
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find positions for this frame
        while (positionIndex < len(crosshairPositions) and 
               crosshairPositions[positionIndex]["frameNumber"] == frameNumber):
            
            pos = crosshairPositions[positionIndex]
            recentPositions.append((pos["x"], pos["y"]))
            
            # Keep only recent positions
            if len(recentPositions) > trailLength:
                recentPositions.pop(0)
            
            positionIndex += 1
        
        # Draw trail
        for i in range(1, len(recentPositions)):
            # Fade effect: older positions are more transparent
            alpha = i / len(recentPositions)
            color = (0, int(255 * alpha), int(255 * alpha))  # Cyan fade
            thickness = max(1, int(3 * alpha))
            
            cv2.line(frame, recentPositions[i-1], recentPositions[i], color, thickness)
        
        # Draw current position
        if recentPositions:
            cv2.circle(frame, recentPositions[-1], 5, (0, 255, 255), -1)  # Cyan dot
        
        out.write(frame)
        frameNumber += 1
    
    cap.release()
    out.release()
    print(f"Visualization saved to {outputPath}")


if __name__ == "__main__":
    # Test the tracker with a sample video
    # You'll need to adjust these color bounds based on your crosshair
    
    # Common Valorant crosshair colors in BGR:
    # Cyan/Light Blue: lower=(100, 200, 200), upper=(255, 255, 255)
    # Green: lower=(0, 200, 0), upper=(100, 255, 100)
    # White: lower=(200, 200, 200), upper=(255, 255, 255)
    # Yellow: lower=(0, 200, 200), upper=(100, 255, 255)
    
    lowerBound = np.array([174, 145, 27])   # BGR: subtract ~30 from each
    upperBound = np.array([234, 205, 87])   # BGR: add ~30 to each
    
    videoPath = "testVod.mp4"  # Put a test video here
    
    sampleRate = 2
    roiSize = 125  # Changed from 200 to 150 (smaller square)
    
    try:
        # Track crosshair
        positions = trackCrosshairInVideo(
            videoPath, 
            lowerBound, 
            upperBound, 
            sampleRate=sampleRate,
            roiSize=roiSize  # Use the smaller ROI
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Total frames tracked: {len(positions)}")
        
        if positions:
            # Apply smoothing
            smoothedPositions = smoothPositions(positions, windowSize=5)
            
            print(f"First position: Frame {smoothedPositions[0]['frameNumber']} at ({smoothedPositions[0]['x']}, {smoothedPositions[0]['y']})")
            print(f"Last position: Frame {smoothedPositions[-1]['frameNumber']} at ({smoothedPositions[-1]['x']}, {smoothedPositions[-1]['y']})")
            
            # Create visualization with smoothed data
            visualizeCrosshairPath(videoPath, smoothedPositions, "visualized_output.mp4")
        
    except FileNotFoundError:
        print(f"Error: Could not find {videoPath}")
        print("Please record a short Valorant clip and save it as 'testVod.mp4' in the project root")
    except Exception as e:
        print(f"Error: {e}")