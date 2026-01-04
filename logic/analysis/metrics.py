import numpy as np

class CrosshairMetrics:
    """
    Calculate performance metrics from crosshair tracking data.
    """
    
    def __init__(self, positions):
        """
        Args:
            positions: List of position dictionaries from tracking
        """
        self.positions = positions
        self.movements = self._calculateMovements()
    
    def _calculateMovements(self):
        """
        Calculate movement vectors between consecutive positions.
        
        Returns:
            List of movement dictionaries with distance, velocity, direction
        """
        movements = []
        
        for i in range(1, len(self.positions)):
            prev = self.positions[i - 1]
            curr = self.positions[i]
            
            # Calculate distance
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate time difference
            timeDiff = curr["timestamp"] - prev["timestamp"]
            
            # Calculate velocity (pixels per second)
            velocity = distance / timeDiff if timeDiff > 0 else 0
            
            # Calculate direction (angle in degrees)
            direction = np.degrees(np.arctan2(dy, dx))
            
            movements.append({
                "frameStart": prev["frameNumber"],
                "frameEnd": curr["frameNumber"],
                "distance": distance,
                "velocity": velocity,
                "direction": direction,
                "dx": dx,
                "dy": dy,
                "timeDiff": timeDiff
            })
        
        return movements
    
    def getAverageVelocity(self):
        """
        Calculate average crosshair velocity across all movements.
        
        Returns:
            Average velocity in pixels/second
        """
        if not self.movements:
            return 0
        
        velocities = [m["velocity"] for m in self.movements]
        return np.mean(velocities)
    
    def getMaxVelocity(self):
        """
        Get maximum crosshair velocity (fastest movement).
        
        Returns:
            Max velocity in pixels/second
        """
        if not self.movements:
            return 0
        
        velocities = [m["velocity"] for m in self.movements]
        return np.max(velocities)
    
    def getTotalDistance(self):
        """
        Calculate total distance crosshair traveled.
        
        Returns:
            Total distance in pixels
        """
        if not self.movements:
            return 0
        
        return sum(m["distance"] for m in self.movements)
    
    def getSmoothness(self):
        """
        Calculate tracking smoothness (lower = smoother).
        Measures variance in velocity changes.
        
        Returns:
            Smoothness score (standard deviation of velocity changes)
        """
        if len(self.movements) < 2:
            return 0
        
        velocities = [m["velocity"] for m in self.movements]
        velocityChanges = [abs(velocities[i] - velocities[i-1]) for i in range(1, len(velocities))]
        
        return np.std(velocityChanges)
    
    def detectFlicks(self, velocityThreshold=2000):
        """
        Detect flick movements (sudden high-velocity movements).
        
        Args:
            velocityThreshold: Minimum velocity to be considered a flick (pixels/sec)
        
        Returns:
            List of flick dictionaries with frame, distance, velocity
        """
        flicks = []
        
        for movement in self.movements:
            if movement["velocity"] > velocityThreshold:
                flicks.append({
                    "frameStart": movement["frameStart"],
                    "frameEnd": movement["frameEnd"],
                    "distance": movement["distance"],
                    "velocity": movement["velocity"],
                    "direction": movement["direction"]
                })
        
        return flicks
    
    def detectFlicksByDistance(self, velocityThreshold=2000):
        """
        Detect flicks and categorize by distance.
        
        Categories:
        - Small (0-100px): Micro-adjustments, should be clean and fast
        - Medium (100-300px): Standard target acquisition
        - Large (300+px): Wide sweeps, corrections expected
        
        Returns:
            Dictionary with flicks categorized by distance
        """
        flicks = {
            'small': [],   # 0-100px
            'medium': [],  # 100-300px
            'large': []    # 300+px
        }
        
        for movement in self.movements:
            if movement["velocity"] > velocityThreshold:
                distance = movement["distance"]
                
                flickData = {
                    "frameStart": movement["frameStart"],
                    "frameEnd": movement["frameEnd"],
                    "distance": distance,
                    "velocity": movement["velocity"]
                }
                
                if distance < 100:
                    flicks['small'].append(flickData)
                elif distance < 300:
                    flicks['medium'].append(flickData)
                else:
                    flicks['large'].append(flickData)
        
        return flicks
    
    def analyzeFlickCorrections(self, flickFrame, lookAheadFrames=10):
        """
        Analyze corrections after a flick.
        
        Args:
            flickFrame: Frame number where flick occurred
            lookAheadFrames: How many frames to analyze after flick
        
        Returns:
            Dictionary with correction metrics
        """
        # Find the flick position in our data
        flickIndex = None
        for i, pos in enumerate(self.positions):
            if pos["frameNumber"] == flickFrame:
                flickIndex = i
                break
        
        if flickIndex is None or flickIndex + lookAheadFrames >= len(self.positions):
            return None
        
        # Get positions after the flick
        postFlickPositions = self.positions[flickIndex:flickIndex + lookAheadFrames + 1]
        
        # Calculate total correction distance
        correctionDistance = 0
        correctionCount = 0
        directionChanges = 0
        
        for i in range(1, len(postFlickPositions)):
            prev = postFlickPositions[i - 1]
            curr = postFlickPositions[i]
            
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Count as correction if moving more than 2 pixels
            if distance > 2:
                correctionDistance += distance
                correctionCount += 1
        
        # Calculate time to stabilize (when movement drops below threshold)
        stabilizationTime = None
        for i in range(1, len(postFlickPositions)):
            prev = postFlickPositions[i - 1]
            curr = postFlickPositions[i]
            
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Stabilized when movement < 2 pixels for 3 consecutive frames
            if distance < 2:
                if stabilizationTime is None:
                    stabilizationTime = curr["timestamp"] - postFlickPositions[0]["timestamp"]
        
        return {
            "correctionDistance": correctionDistance,
            "correctionCount": correctionCount,
            "stabilizationTime": stabilizationTime if stabilizationTime else (postFlickPositions[-1]["timestamp"] - postFlickPositions[0]["timestamp"])
        }
        
    def getFlickAnalysisByDistance(self, velocityThreshold=2000):
        """
        Get comprehensive flick analysis categorized by distance.
        
        Returns:
            Dictionary with statistics for small, medium, and large flicks
        """
        flicksByDistance = self.detectFlicksByDistance(velocityThreshold)
        
        analysis = {}
        
        for category, flicks in flicksByDistance.items():
            if not flicks:
                analysis[category] = {
                    "count": 0,
                    "avgDistance": 0,
                    "avgVelocity": 0,
                    "avgCorrectionCount": 0,
                    "avgCorrectionDistance": 0,
                    "avgStabilizationTime": 0
                }
                continue
            
            # Analyze corrections for each flick
            corrections = []
            for flick in flicks:
                correctionData = self.analyzeFlickCorrections(flick["frameEnd"])
                if correctionData:
                    corrections.append(correctionData)
            
            # Calculate averages
            avgCorrectionCount = np.mean([c["correctionCount"] for c in corrections]) if corrections else 0
            avgCorrectionDistance = np.mean([c["correctionDistance"] for c in corrections]) if corrections else 0
            avgStabilizationTime = np.mean([c["stabilizationTime"] for c in corrections]) if corrections else 0
            
            analysis[category] = {
                "count": len(flicks),
                "avgDistance": np.mean([f["distance"] for f in flicks]),
                "avgVelocity": np.mean([f["velocity"] for f in flicks]),
                "avgCorrectionCount": avgCorrectionCount,
                "avgCorrectionDistance": avgCorrectionDistance,
                "avgStabilizationTime": avgStabilizationTime
            }
        
        return analysis
    
    def analyzeFlickAccuracyWithTargets(videoPath, crosshairPositions, targetDetector, velocityThreshold=2000):
        """
        Analyze flick accuracy by detecting targets with YOLO.
        
        Args:
            videoPath: Path to video file
            crosshairPositions: List of crosshair tracking data
            targetDetector: TargetDetector instance
            velocityThreshold: Minimum velocity to consider a flick
        
        Returns:
            Dictionary with overshoot/undershoot statistics
        """
        cap = cv2.VideoCapture(videoPath)
        
        flickResults = {
            'overshoot': [],
            'undershoot': [],
            'on-target': []
        }
        
        print("Analyzing flick accuracy with target detection...")
        
        for i in range(1, len(crosshairPositions)):
            prev = crosshairPositions[i-1]
            curr = crosshairPositions[i]
            
            # Calculate velocity
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            distance = np.sqrt(dx**2 + dy**2)
            timeDiff = curr["timestamp"] - prev["timestamp"]
            velocity = distance / timeDiff if timeDiff > 0 else 0
            
            # Check if this is a flick
            if velocity > velocityThreshold:
                # Get frame at flick endpoint
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr["frameNumber"])
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Detect targets in frame
                targets = targetDetector.detectHeads(frame)
                
                if not targets:
                    continue
                
                # Find nearest target to crosshair endpoint
                crosshairPos = (curr["x"], curr["y"])
                nearestTarget = targetDetector.findNearestTarget(crosshairPos, targets)
                
                if not nearestTarget:
                    continue
                
                # Calculate flick accuracy
                # Vector from start position to target
                targetDx = nearestTarget['x'] - prev["x"]
                targetDy = nearestTarget['y'] - prev["y"]
                distanceToTarget = np.sqrt(targetDx**2 + targetDy**2)
                
                # Vector from start position to crosshair endpoint
                flickDx = curr["x"] - prev["x"]
                flickDy = curr["y"] - prev["y"]
                flickDistance = np.sqrt(flickDx**2 + flickDy**2)
                
                # Project crosshair endpoint onto target direction
                if distanceToTarget > 0:
                    dotProduct = (flickDx * targetDx + flickDy * targetDy)
                    projection = dotProduct / distanceToTarget
                    
                    # Error = how much we over/undershot
                    error = projection - distanceToTarget
                    
                    flickData = {
                        'frameNumber': curr["frameNumber"],
                        'error': error,
                        'distanceToTarget': distanceToTarget,
                        'flickDistance': flickDistance,
                        'targetConfidence': nearestTarget['confidence']
                    }
                    
                    # Categorize
                    if error > 15:  # Overshot by >15 pixels
                        flickResults['overshoot'].append(flickData)
                    elif error < -15:  # Undershot by >15 pixels
                        flickResults['undershoot'].append(flickData)
                    else:  # Within 15 pixels = on-target
                        flickResults['on-target'].append(flickData)
                
                # Progress indicator
                if i % 100 == 0:
                    print(f"Processed {i}/{len(crosshairPositions)} positions...")
        
        cap.release()
        
        # Calculate statistics
        totalFlicks = sum(len(v) for v in flickResults.values())
        
        if totalFlicks == 0:
            return None
        
        stats = {
            'totalFlicks': totalFlicks,
            'overshootCount': len(flickResults['overshoot']),
            'undershootCount': len(flickResults['undershoot']),
            'onTargetCount': len(flickResults['on-target']),
            'overshootPercent': len(flickResults['overshoot']) / totalFlicks * 100,
            'undershootPercent': len(flickResults['undershoot']) / totalFlicks * 100,
            'onTargetPercent': len(flickResults['on-target']) / totalFlicks * 100,
            'avgOvershootError': np.mean([f['error'] for f in flickResults['overshoot']]) if flickResults['overshoot'] else 0,
            'avgUndershootError': np.mean([abs(f['error']) for f in flickResults['undershoot']]) if flickResults['undershoot'] else 0
        }
        
        return stats
    
    def getFlickStats(self, velocityThreshold=2000):
        """
        Get statistics about flicks.
        
        Returns:
            Dictionary with flick count, average distance, average velocity
        """
        flicks = self.detectFlicks(velocityThreshold)
        
        if not flicks:
            return {
                "count": 0,
                "averageDistance": 0,
                "averageVelocity": 0,
                "maxDistance": 0,
                "maxVelocity": 0
            }
        
        distances = [f["distance"] for f in flicks]
        velocities = [f["velocity"] for f in flicks]
        
        return {
            "count": len(flicks),
            "averageDistance": np.mean(distances),
            "averageVelocity": np.mean(velocities),
            "maxDistance": np.max(distances),
            "maxVelocity": np.max(velocities)
        }
    
    def getFlickStats(self, velocityThreshold=2000):
        """
        Get statistics about flicks.
        
        Returns:
            Dictionary with flick count, average distance, average velocity
        """
        flicks = self.detectFlicks(velocityThreshold)
        
        if not flicks:
            return {
                "count": 0,
                "averageDistance": 0,
                "averageVelocity": 0,
                "maxDistance": 0,
                "maxVelocity": 0
            }
        
        distances = [f["distance"] for f in flicks]
        velocities = [f["velocity"] for f in flicks]
        
        return {
            "count": len(flicks),
            "averageDistance": np.mean(distances),
            "averageVelocity": np.mean(velocities),
            "maxDistance": np.max(distances),
            "maxVelocity": np.max(velocities)
        }
    
    def getTrackingSegments(self, velocityThreshold=500):
        """
        Identify tracking segments (smooth, continuous movements).
        Tracking = low-to-medium velocity, continuous movement.
        
        Args:
            velocityThreshold: Maximum velocity to be considered tracking
        
        Returns:
            List of tracking segment dictionaries
        """
        trackingSegments = []
        currentSegment = []
        
        for i, movement in enumerate(self.movements):
            if movement["velocity"] <= velocityThreshold and movement["distance"] > 5:
                # This is tracking movement
                currentSegment.append(i)
            else:
                # End of tracking segment
                if len(currentSegment) >= 3:  # At least 3 consecutive tracking movements
                    totalDistance = sum(self.movements[j]["distance"] for j in currentSegment)
                    avgVelocity = np.mean([self.movements[j]["velocity"] for j in currentSegment])
                    
                    trackingSegments.append({
                        "startFrame": self.movements[currentSegment[0]]["frameStart"],
                        "endFrame": self.movements[currentSegment[-1]]["frameEnd"],
                        "duration": len(currentSegment),
                        "distance": totalDistance,
                        "avgVelocity": avgVelocity
                    })
                
                currentSegment = []
        
        # Handle last segment
        if len(currentSegment) >= 3:
            totalDistance = sum(self.movements[j]["distance"] for j in currentSegment)
            avgVelocity = np.mean([self.movements[j]["velocity"] for j in currentSegment])
            
            trackingSegments.append({
                "startFrame": self.movements[currentSegment[0]]["frameStart"],
                "endFrame": self.movements[currentSegment[-1]]["frameEnd"],
                "duration": len(currentSegment),
                "distance": totalDistance,
                "avgVelocity": avgVelocity
            })
        
        return trackingSegments
    
    def getSummary(self):
        """
        Get a complete summary of all metrics.
        
        Returns:
            Dictionary with all calculated metrics
        """
        flickStats = self.getFlickStats(velocityThreshold=2000)  # Changed from 500 to 2000
        trackingSegments = self.getTrackingSegments()
        
        return {
            "totalFrames": len(self.positions),
            "totalDistance": self.getTotalDistance(),
            "averageVelocity": self.getAverageVelocity(),
            "maxVelocity": self.getMaxVelocity(),
            "smoothness": self.getSmoothness(),
            "flicks": flickStats,
            "trackingSegmentCount": len(trackingSegments),
            "totalTrackingDistance": sum(seg["distance"] for seg in trackingSegments) if trackingSegments else 0
        }
        
def analyzeFlickAccuracyWithTargets(videoPath, crosshairPositions, targetDetector, velocityThreshold=2000):
    """
    Analyze flick accuracy by detecting targets with YOLO.
    
    Args:
        videoPath: Path to video file
        crosshairPositions: List of crosshair tracking data
        targetDetector: TargetDetector instance
        velocityThreshold: Minimum velocity to consider a flick
    
    Returns:
        Dictionary with overshoot/undershoot statistics
    """
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(videoPath)
    
    flickResults = {
        'overshoot': [],
        'undershoot': [],
        'on-target': []
    }
    
    print("Analyzing flick accuracy with target detection...")
    
    for i in range(1, len(crosshairPositions)):
        prev = crosshairPositions[i-1]
        curr = crosshairPositions[i]
        
        # Calculate velocity
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        distance = np.sqrt(dx**2 + dy**2)
        timeDiff = curr["timestamp"] - prev["timestamp"]
        velocity = distance / timeDiff if timeDiff > 0 else 0
        
        # Check if this is a flick
        if velocity > velocityThreshold:
            # Get frame at flick endpoint
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr["frameNumber"])
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect targets in frame
            targets = targetDetector.detectHeads(frame)
            
            if not targets:
                continue
            
            # Find nearest target to crosshair endpoint
            crosshairPos = (curr["x"], curr["y"])
            nearestTarget = targetDetector.findNearestTarget(crosshairPos, targets)
            
            if not nearestTarget:
                continue
            
            # Calculate flick accuracy
            # Vector from start position to target
            targetDx = nearestTarget['x'] - prev["x"]
            targetDy = nearestTarget['y'] - prev["y"]
            distanceToTarget = np.sqrt(targetDx**2 + targetDy**2)
            
            # Vector from start position to crosshair endpoint
            flickDx = curr["x"] - prev["x"]
            flickDy = curr["y"] - prev["y"]
            flickDistance = np.sqrt(flickDx**2 + flickDy**2)
            
            # Project crosshair endpoint onto target direction
            if distanceToTarget > 0:
                dotProduct = (flickDx * targetDx + flickDy * targetDy)
                projection = dotProduct / distanceToTarget
                
                # Error = how much we over/undershot
                error = projection - distanceToTarget
                
                flickData = {
                    'frameNumber': curr["frameNumber"],
                    'error': error,
                    'distanceToTarget': distanceToTarget,
                    'flickDistance': flickDistance,
                    'targetConfidence': nearestTarget['confidence']
                }
                
                # Categorize
                if error > 15:  # Overshot by >15 pixels
                    flickResults['overshoot'].append(flickData)
                elif error < -15:  # Undershot by >15 pixels
                    flickResults['undershoot'].append(flickData)
                else:  # Within 15 pixels = on-target
                    flickResults['on-target'].append(flickData)
            
            # Progress indicator
            if i % 100 == 0:
                print(f"Processed {i}/{len(crosshairPositions)} positions...")
    
    cap.release()
    
    # Calculate statistics
    totalFlicks = sum(len(v) for v in flickResults.values())
    
    if totalFlicks == 0:
        return None
    
    stats = {
        'totalFlicks': totalFlicks,
        'overshootCount': len(flickResults['overshoot']),
        'undershootCount': len(flickResults['undershoot']),
        'onTargetCount': len(flickResults['on-target']),
        'overshootPercent': len(flickResults['overshoot']) / totalFlicks * 100,
        'undershootPercent': len(flickResults['undershoot']) / totalFlicks * 100,
        'onTargetPercent': len(flickResults['on-target']) / totalFlicks * 100,
        'avgOvershootError': np.mean([f['error'] for f in flickResults['overshoot']]) if flickResults['overshoot'] else 0,
        'avgUndershootError': np.mean([abs(f['error']) for f in flickResults['undershoot']]) if flickResults['undershoot'] else 0
    }
    
    return stats