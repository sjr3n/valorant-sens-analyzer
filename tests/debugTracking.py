import cv2
import numpy as np

videos = [
    ("11.mp4", 0.11),
    ("145.mp4", 0.145),
    ("23.mp4", 0.23),
    ("43.mp4", 0.43)
]

# Green crosshair bounds
lowerBound = np.array([0, 200, 0])
upperBound = np.array([100, 255, 100])

for videoPath, sens in videos:
    print(f"\n{'='*60}")
    print(f"Analyzing: {videoPath} (Sensitivity: {sens})")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frameNumber = 0
    positions = []
    
    while frameNumber < 2000:  # Check first 2000 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Search ENTIRE frame
        mask = cv2.inRange(frame, lowerBound, upperBound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                positions.append((frameNumber, x, y))
        
        frameNumber += 1
    
    cap.release()
    
    print(f"Tracked {len(positions)} / {frameNumber} frames ({len(positions)/frameNumber*100:.1f}%)")
    
    if len(positions) > 1:
        maxMove = 0
        maxMoveFrames = (0, 0)
        
        for i in range(1, len(positions)):
            prev = positions[i-1]
            curr = positions[i]
            
            dx = curr[1] - prev[1]
            dy = curr[2] - prev[2]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > maxMove:
                maxMove = distance
                maxMoveFrames = (prev[0], curr[0])
        
        timeDiff = (maxMoveFrames[1] - maxMoveFrames[0]) / fps
        velocity = maxMove / timeDiff if timeDiff > 0 else 0
        
        print(f"Largest movement: {maxMove:.2f} pixels at {velocity:.2f} px/s")
        print(f"  Between frames {maxMoveFrames[0]} and {maxMoveFrames[1]}")