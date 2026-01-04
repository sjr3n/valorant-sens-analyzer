import sys
import os
import numpy as np

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectRoot)

from logic.vodProcessor.crosshairTracker import trackCrosshairInVideo, smoothPositions
from logic.vodProcessor.videoUtils import saveTrackingData, getVideoMetadata
from logic.analysis.metrics import CrosshairMetrics

def analyzeTdmGameplay(videoPath, sensitivity):
    """
    Analyze TDM gameplay and provide sensitivity diagnostics.
    
    Returns diagnostic feedback on whether sensitivity is too high/low.
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING TDM GAMEPLAY - Sensitivity: {sensitivity}")
    print(f"{'='*70}\n")
    
    # Crosshair detection bounds
    lowerBound = np.array([0, 150, 0])
    upperBound = np.array([150, 255, 150])
    
    # Get video metadata
    metadata = getVideoMetadata(videoPath)
    print(f"Video: {metadata['duration']:.1f}s at {metadata['fps']} FPS\n")
    
    # Track crosshair
    print("Tracking crosshair movement...")
    positions = trackCrosshairInVideo(
        videoPath,
        lowerBound,
        upperBound,
        sampleRate=1,
        roiSize=800
    )
    
    if not positions:
        print("❌ Could not track crosshair")
        return None
    
    # Smooth positions
    smoothedPositions = smoothPositions(positions, windowSize=5)
    print(f"✓ Tracked {len(smoothedPositions)} frames\n")
    
    # Calculate metrics
    metrics = CrosshairMetrics(smoothedPositions)
    
    # Get movement analysis
    movements = metrics.movements
    velocities = [m["velocity"] for m in movements]
    
    # Detect flicks (high velocity movements)
    flicks = [m for m in movements if m["velocity"] > 2000]
    
    # Detect micro-adjustments (small, slower movements)
    microAdjustments = [m for m in movements if 10 < m["distance"] < 50 and m["velocity"] < 1000]
    
    # Detect corrections (direction reversals after fast movements)
    corrections = []
    for i in range(1, len(movements)):
        prev = movements[i-1]
        curr = movements[i]
        
        # Check if direction reversed after a fast movement
        if prev["velocity"] > 1500:
            prevAngle = np.arctan2(prev["dy"], prev["dx"])
            currAngle = np.arctan2(curr["dy"], curr["dx"])
            angleDiff = abs(prevAngle - currAngle)
            
            if angleDiff > np.pi / 2:  # >90 degree change = correction
                corrections.append({
                    "frame": curr["frameStart"],
                    "correctionDistance": curr["distance"]
                })
    
    # Analyze flick distances
    flickDistances = [f["distance"] for f in flicks]
    smallFlicks = [f for f in flicks if f["distance"] < 100]
    mediumFlicks = [f for f in flicks if 100 <= f["distance"] < 300]
    largeFlicks = [f for f in flicks if f["distance"] >= 300]
    
    # Calculate diagnostics
    diagnostics = {
        "totalFrames": len(smoothedPositions),
        "totalDistance": metrics.getTotalDistance(),
        "avgVelocity": metrics.getAverageVelocity(),
        "maxVelocity": metrics.getMaxVelocity(),
        "smoothness": metrics.getSmoothness(),
        "flickCount": len(flicks),
        "smallFlickCount": len(smallFlicks),
        "mediumFlickCount": len(mediumFlicks),
        "largeFlickCount": len(largeFlicks),
        "avgFlickDistance": np.mean(flickDistances) if flickDistances else 0,
        "microAdjustmentCount": len(microAdjustments),
        "correctionCount": len(corrections),
        "avgCorrectionDistance": np.mean([c["correctionDistance"] for c in corrections]) if corrections else 0,
        "velocityMedian": np.median(velocities),
        "velocity95th": np.percentile(velocities, 95)
    }
    
    return diagnostics


def provideDiagnostics(diagnostics, sensitivity):
    """
    Provide sensitivity recommendations based on gameplay analysis.
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT - Sensitivity: {sensitivity}")
    print(f"{'='*70}\n")
    
    # Display key metrics
    print("📊 MOVEMENT METRICS:")
    print(f"  Total distance traveled: {diagnostics['totalDistance']:.0f} pixels")
    print(f"  Average velocity: {diagnostics['avgVelocity']:.1f} px/s")
    print(f"  Max velocity: {diagnostics['maxVelocity']:.1f} px/s")
    print(f"  Smoothness score: {diagnostics['smoothness']:.1f} (lower = smoother)")
    
    print(f"\n🎯 FLICK ANALYSIS:")
    print(f"  Total flicks: {diagnostics['flickCount']}")
    print(f"    Small (<100px): {diagnostics['smallFlickCount']}")
    print(f"    Medium (100-300px): {diagnostics['mediumFlickCount']}")
    print(f"    Large (300+px): {diagnostics['largeFlickCount']}")
    print(f"  Average flick distance: {diagnostics['avgFlickDistance']:.1f} pixels")
    
    print(f"\n🔧 ADJUSTMENT ANALYSIS:")
    print(f"  Micro-adjustments: {diagnostics['microAdjustmentCount']}")
    print(f"  Corrections (reversals): {diagnostics['correctionCount']}")
    if diagnostics['correctionCount'] > 0:
        print(f"  Avg correction distance: {diagnostics['avgCorrectionDistance']:.1f} pixels")
    
    # Diagnostic logic
    warnings = []
    recommendations = []
    positives = []
    
    # SENSITIVITY TOO LOW indicators
    if diagnostics['maxVelocity'] < 5000:
        warnings.append("⚠️  Low max velocity - you might be struggling to flick to targets")
    
    if diagnostics['avgFlickDistance'] > 200 and diagnostics['velocity95th'] < 4000:
        warnings.append("⚠️  Large flick distances with low velocity - sens might be too low for quick target acquisition")
    
    # SENSITIVITY TOO HIGH indicators
    if diagnostics['correctionCount'] > diagnostics['flickCount'] * 0.6:
        warnings.append("⚠️  High correction rate - you're overshooting frequently (sens might be too high)")
    
    if diagnostics['smoothness'] > 1000:
        warnings.append("⚠️  High smoothness score - movements are jittery (sens might be too high)")
    
    if diagnostics['microAdjustmentCount'] > diagnostics['flickCount'] * 2:
        warnings.append("⚠️  Lots of micro-adjustments - struggling with precise small movements (sens might be too high)")
    
    # GOOD performance indicators
    if 300 <= diagnostics['smoothness'] <= 700:
        positives.append("✅ Good movement smoothness - tracking is stable")
    
    if diagnostics['correctionCount'] < diagnostics['flickCount'] * 0.4:
        positives.append("✅ Low correction rate - your flicks are landing close to target")
    
    if 5000 <= diagnostics['maxVelocity'] <= 12000:
        positives.append("✅ Good flick capability - you can move quickly when needed")
    
    # Print diagnostics
    if warnings:
        print(f"\n{'='*70}")
        print("⚠️  POTENTIAL ISSUES:")
        for warning in warnings:
            print(f"  {warning}")
    
    if positives:
        print(f"\n{'='*70}")
        print("✅ STRENGTHS:")
        for positive in positives:
            print(f"  {positive}")
    
    # Overall recommendation
    print(f"\n{'='*70}")
    print("💡 RECOMMENDATION:")
    
    # Too low indicators
    lowMaxVelocity = diagnostics['maxVelocity'] < 8000
    lowAvgVelocity = diagnostics['avgVelocity'] < 1000
    noLargeFlicks = diagnostics['largeFlickCount'] == 0
    lotsOfSmallFlicks = diagnostics['smallFlickCount'] > diagnostics['flickCount'] * 0.6
    
    # Too high indicators (need to be combined with high velocity)
    highCorrections = diagnostics['correctionCount'] > diagnostics['flickCount'] * 0.6
    highMicroAdjustments = diagnostics['microAdjustmentCount'] > diagnostics['flickCount'] * 2
    highMaxVelocity = diagnostics['maxVelocity'] > 15000
    
    # High smoothness can indicate EITHER too high OR too low sens
    # Combine with velocity to determine which
    highSmoothness = diagnostics['smoothness'] > 1000
    
    # If high smoothness + low velocity = fighting against low sens (choppy from trying to move fast)
    # If high smoothness + high velocity = actually too sensitive (twitchy overcontrol)
    
    if highSmoothness and lowMaxVelocity:
        warnings_detail = "High smoothness + low velocity = choppy micro-movements fighting low sens"
        tooLowScore = 3  # Strong indicator
    elif highSmoothness and highMaxVelocity:
        warnings_detail = "High smoothness + high velocity = twitchy overcontrol"
        tooHighScore = 3  # Strong indicator
    else:
        warnings_detail = None
        tooLowScore = sum([lowMaxVelocity, lowAvgVelocity, noLargeFlicks, lotsOfSmallFlicks])
        tooHighScore = sum([highCorrections and highMaxVelocity, highMicroAdjustments and highMaxVelocity])
    
    if warnings_detail:
        print(f"  ℹ️  {warnings_detail}\n")
    
    if tooLowScore >= 2:
        print(f"  📈 Your sensitivity ({sensitivity}) appears TOO LOW")
        print(f"     → Try increasing to {sensitivity + 0.05:.3f} - {sensitivity + 0.10:.3f}")
        print(f"\n  Why:")
        if lowMaxVelocity:
            print(f"     • Max velocity {diagnostics['maxVelocity']:.0f} px/s is low (good TDM players hit 15k+)")
        if noLargeFlicks:
            print(f"     • Zero large flicks detected - you might be unable to turn fast enough")
        if lotsOfSmallFlicks:
            print(f"     • {diagnostics['smallFlickCount']/diagnostics['flickCount']*100:.0f}% of flicks are small - limited range of motion")
        if highSmoothness and lowMaxVelocity:
            print(f"     • Jittery movement despite low velocity = fighting against low sens")
            
    elif tooHighScore >= 2:
        print(f"  📉 Your sensitivity ({sensitivity}) appears TOO HIGH")
        print(f"     → Try decreasing to {sensitivity - 0.05:.3f} - {sensitivity - 0.10:.3f}")
        print(f"\n  Why:")
        if highCorrections and highMaxVelocity:
            print(f"     • High correction rate ({diagnostics['correctionCount']}) with high velocity = overshooting")
        if highMicroAdjustments and highMaxVelocity:
            print(f"     • Lots of micro-adjustments = struggling with precision")
            
    else:
        print(f"  ✓ Your sensitivity ({sensitivity}) seems REASONABLE")
        print(f"     → No major red flags detected")
        print(f"     → Consider testing slightly higher/lower to fine-tune")
    
    print(f"{'='*70}\n")


# Main execution
if __name__ == "__main__":
    # Test videos
    tests = [
        {"video": "tdm_0.117.mp4", "sens": 0.117},
        {"video": "tdm_0.117LUT.mp4", "sens": 0.117}
        # Add more as you record them
    ]
    
    for test in tests:
        if not os.path.exists(test["video"]):
            print(f"⚠️  Video not found: {test['video']}")
            continue
        
        diagnostics = analyzeTdmGameplay(test["video"], test["sens"])
        
        if diagnostics:
            provideDiagnostics(diagnostics, test["sens"])