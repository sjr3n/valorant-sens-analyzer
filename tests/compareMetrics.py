import sys
import os
import numpy as np
import json

# Add project root to Python path
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectRoot)

from logic.analysis.metrics import CrosshairMetrics
from logic.vodProcessor.videoUtils import loadTrackingData

# Sensitivities to compare
sensitivities = [0.11, 0.145, 0.23, 0.43]

# Exports folder
exportsFolder = "exports"

print("\n" + "="*80)
print("SENSITIVITY COMPARISON ANALYSIS")
print("="*80)

allResults = {}

for sens in sensitivities:
    trackingFile = os.path.join(exportsFolder, f"tracking_{sens}.json")
    
    try:
        # Load tracking data
        positions = loadTrackingData(trackingFile)
        
        # Calculate metrics
        metrics = CrosshairMetrics(positions)
        summary = metrics.getSummary()
        
        # Get velocity distribution
        velocities = [m["velocity"] for m in metrics.movements]
        
        # Get flick analysis by distance
        flickAnalysis = metrics.getFlickAnalysisByDistance(velocityThreshold=2000)
        
        allResults[sens] = {
            "summary": summary,
            "velocities": {
                "min": min(velocities) if velocities else 0,
                "25th": np.percentile(velocities, 25) if velocities else 0,
                "median": np.percentile(velocities, 50) if velocities else 0,
                "75th": np.percentile(velocities, 75) if velocities else 0,
                "95th": np.percentile(velocities, 95) if velocities else 0,
                "max": max(velocities) if velocities else 0
            },
            "flickAnalysis": flickAnalysis
        }
        
        print(f"\n✓ Loaded data for sensitivity {sens}")
    
    except FileNotFoundError:
        print(f"\n✗ Could not find {trackingFile}")

# Print comparison table
print("\n" + "="*80)
print("OVERALL METRICS")
print("="*80)

print(f"\n{'Sensitivity':<12} {'Frames':<10} {'Distance':<12} {'Avg Vel':<12} {'Max Vel':<12} {'Smoothness':<12}")
print("-"*80)

for sens in sensitivities:
    if sens in allResults:
        s = allResults[sens]["summary"]
        print(f"{sens:<12.3f} {s['totalFrames']:<10} {s['totalDistance']:<12.2f} {s['averageVelocity']:<12.2f} {s['maxVelocity']:<12.2f} {s['smoothness']:<12.2f}")

# Flicks comparison
print("\n" + "="*80)
print("FLICKS (threshold: 2000 px/s)")
print("="*80)

print(f"\n{'Sensitivity':<12} {'Count':<10} {'Avg Dist':<12} {'Avg Vel':<12} {'Max Dist':<12}")
print("-"*80)

for sens in sensitivities:
    if sens in allResults:
        f = allResults[sens]["summary"]["flicks"]
        print(f"{sens:<12.3f} {f['count']:<10} {f['averageDistance']:<12.2f} {f['averageVelocity']:<12.2f} {f['maxDistance']:<12.2f}")

# NEW: Flick analysis by distance
print("\n" + "="*80)
print("MICRO-FLICKS ANALYSIS (<100px) - KEY DIAGNOSTIC METRIC")
print("="*80)

print(f"\n{'Sensitivity':<12} {'Count':<10} {'Avg Corr':<12} {'Corr Dist':<12} {'Stab Time':<12}")
print("-"*80)

for sens in sensitivities:
    if sens in allResults:
        small = allResults[sens]["flickAnalysis"]["small"]
        if small["count"] > 0:
            print(f"{sens:<12.3f} {small['count']:<10} {small['avgCorrectionCount']:<12.2f} {small['avgCorrectionDistance']:<12.2f} {small['avgStabilizationTime']:<12.3f}")
        else:
            print(f"{sens:<12.3f} {'0':<10} {'-':<12} {'-':<12} {'-':<12}")

print("\n" + "="*80)
print("MEDIUM FLICKS ANALYSIS (100-300px)")
print("="*80)

print(f"\n{'Sensitivity':<12} {'Count':<10} {'Avg Corr':<12} {'Corr Dist':<12} {'Stab Time':<12}")
print("-"*80)

for sens in sensitivities:
    if sens in allResults:
        medium = allResults[sens]["flickAnalysis"]["medium"]
        if medium["count"] > 0:
            print(f"{sens:<12.3f} {medium['count']:<10} {medium['avgCorrectionCount']:<12.2f} {medium['avgCorrectionDistance']:<12.2f} {medium['avgStabilizationTime']:<12.3f}")
        else:
            print(f"{sens:<12.3f} {'0':<10} {'-':<12} {'-':<12} {'-':<12}")

print("\n" + "="*80)
print("LARGE FLICKS ANALYSIS (300+px)")
print("="*80)

print(f"\n{'Sensitivity':<12} {'Count':<10} {'Avg Corr':<12} {'Corr Dist':<12} {'Stab Time':<12}")
print("-"*80)

for sens in sensitivities:
    if sens in allResults:
        large = allResults[sens]["flickAnalysis"]["large"]
        if large["count"] > 0:
            print(f"{sens:<12.3f} {large['count']:<10} {large['avgCorrectionCount']:<12.2f} {large['avgCorrectionDistance']:<12.2f} {large['avgStabilizationTime']:<12.3f}")
        else:
            print(f"{sens:<12.3f} {'0':<10} {'-':<12} {'-':<12} {'-':<12}")

# Tracking comparison
print("\n" + "="*80)
print("TRACKING")
print("="*80)

print(f"\n{'Sensitivity':<12} {'Segments':<12} {'Total Dist':<12}")
print("-"*80)

for sens in sensitivities:
    if sens in allResults:
        s = allResults[sens]["summary"]
        print(f"{sens:<12.3f} {s['trackingSegmentCount']:<12} {s['totalTrackingDistance']:<12.2f}")

# Velocity distribution comparison
print("\n" + "="*80)
print("VELOCITY DISTRIBUTION (px/s)")
print("="*80)

print(f"\n{'Sensitivity':<12} {'Min':<10} {'25th':<10} {'Median':<10} {'75th':<10} {'95th':<10} {'Max':<10}")
print("-"*80)

for sens in sensitivities:
    if sens in allResults:
        v = allResults[sens]["velocities"]
        print(f"{sens:<12.3f} {v['min']:<10.2f} {v['25th']:<10.2f} {v['median']:<10.2f} {v['75th']:<10.2f} {v['95th']:<10.2f} {v['max']:<10.2f}")

# Analysis and recommendations
print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATIONS")
print("="*80)

if allResults:
    # Find best for different scenarios
    bestSmooth = min(allResults.items(), key=lambda x: x[1]["summary"]["smoothness"])
    bestMaxVel = max(allResults.items(), key=lambda x: x[1]["summary"]["maxVelocity"])
    
    # Find best micro-flick performance
    microFlickPerformance = {}
    for sens, data in allResults.items():
        small = data["flickAnalysis"]["small"]
        if small["count"] > 0:
            # Lower correction count and distance = better
            score = small["avgCorrectionCount"] + (small["avgCorrectionDistance"] / 10)
            microFlickPerformance[sens] = score
    
    if microFlickPerformance:
        bestMicroFlick = min(microFlickPerformance.items(), key=lambda x: x[1])
        
        print(f"\n🎯 BEST MICRO-FLICK CONTROL: {bestMicroFlick[0]} (score: {bestMicroFlick[1]:.2f})")
        print(f"   This sensitivity requires the least correction for small adjustments.")
        print(f"   Micro-flick performance is the best diagnostic for sensitivity fit.")
    
    print(f"\n📊 Smoothest tracking: {bestSmooth[0]} (smoothness score: {bestSmooth[1]['summary']['smoothness']:.2f})")
    print(f"⚡ Highest max velocity: {bestMaxVel[0]} ({bestMaxVel[1]['summary']['maxVelocity']:.2f} px/s)")

# Save comparison to exports folder
comparisonFile = os.path.join(exportsFolder, "sensitivity_comparison.json")
with open(comparisonFile, 'w') as f:
    # Convert to serializable format
    saveData = {}
    for sens, data in allResults.items():
        saveData[str(sens)] = {
            "summary": data["summary"],
            "velocities": data["velocities"],
            "flickAnalysis": data["flickAnalysis"]
        }
    json.dump(saveData, f, indent=2)

print(f"\n💾 Full comparison saved to {comparisonFile}")
print("\n" + "="*80)