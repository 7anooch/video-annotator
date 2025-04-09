#!/usr/bin/env python3
"""
Advanced statistical analysis example using the enhanced analysis tools.

This script demonstrates how to use the enhanced analysis tools for advanced statistical analysis tasks.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so that the script can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the enhanced analysis tools
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

def main():
    """Run the advanced statistical analysis example."""
    print("Advanced Statistical Analysis Example")
    print("===================================")
    
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python advanced_statistical_analysis_example.py <annotation_file>")
        return
    
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Loading annotations from: {file_path}")
    
    # Load annotations
    data = AnnotationData()
    result = data.load_from_file(file_path)
    
    if not result:
        print(f"Error: Failed to load file: {file_path}")
        return
    
    print(f"Loaded {data.get_annotation_count()} annotations")
    print(f"Labels: {', '.join(data.get_labels())}")
    print(f"Frame range: {min(data.get_frames())} - {max(data.get_frames())}")
    
    # Create a statistical analysis object
    stats = StatisticalAnalysis()
    
    # Perform change point detection
    print("\nChange Point Detection:")
    try:
        change_points = stats.detect_change_points(data, method='binary_segmentation', penalty='bic')
        print(f"  Method: {change_points['method']}")
        print(f"  Penalty: {change_points['penalty']}")
        print(f"  Number of change points: {len(change_points['change_point_frames'])}")
        print(f"  Change point frames: {change_points['change_point_frames']}")
        print("  Segments:")
        for i, segment in enumerate(change_points['segments']):
            print(f"    Segment {i+1}:")
            print(f"      Start frame: {segment['start_frame']}")
            print(f"      End frame: {segment['end_frame']}")
            print(f"      Length: {segment['length']}")
            print(f"      Most common label: {segment.get('most_common_label', 'N/A')}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        print("  Note: Change point detection requires the 'ruptures' package.")
        print("  Install it with: pip install ruptures")
    
    # Perform Hidden Markov Model analysis
    print("\nHidden Markov Model Analysis:")
    try:
        hmm_results = stats.hmm_analysis(data, n_states=3)
        print(f"  Number of states: {hmm_results['n_states']}")
        print("  Most likely labels for each state:")
        for state, label in hmm_results['most_likely_labels'].items():
            print(f"    State {state}: {label}")
        print("  Transition matrix:")
        for i, row in enumerate(hmm_results['transition_matrix']):
            print(f"    State {i}: {[round(p, 2) for p in row]}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        print("  Note: HMM analysis requires the 'hmmlearn' package.")
        print("  Install it with: pip install hmmlearn")
    
    # Perform Markov Chain Monte Carlo simulation
    print("\nMarkov Chain Monte Carlo Simulation:")
    try:
        mcmc_results = stats.mcmc_simulation(data, n_samples=1000)
        print(f"  Number of samples: {mcmc_results['n_samples']}")
        print(f"  Unique labels: {mcmc_results['unique_labels']}")
        print("  Transition matrix:")
        for i, row in enumerate(mcmc_results['transition_matrix']):
            label = mcmc_results['unique_labels'][i]
            print(f"    {label}: {[round(p, 2) for p in row]}")
        print("  Generated label counts:")
        for label, count in mcmc_results['generated_label_counts'].items():
            print(f"    {label}: {count}")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Detect anomalies using different methods
    print("\nAnomaly Detection:")
    
    # Z-score method
    print("\nZ-score Method:")
    anomalies = stats.detect_anomalies(data, method="zscore", threshold=3.0)
    if not anomalies:
        print("  No anomalies detected.")
    else:
        print(f"  Detected {len(anomalies)} anomalies:")
        for frame, anomaly in sorted(anomalies.items())[:5]:  # Show only the first 5 anomalies
            print(f"    Frame {frame}:")
            print(f"      Label: {anomaly['label']}")
            print(f"      Duration: {anomaly['duration']}")
            print(f"      Z-score: {anomaly['zscore']:.2f}")
    
    # IQR method
    print("\nIQR Method:")
    anomalies = stats.detect_anomalies(data, method="iqr", threshold=1.5)
    if not anomalies:
        print("  No anomalies detected.")
    else:
        print(f"  Detected {len(anomalies)} anomalies:")
        for frame, anomaly in sorted(anomalies.items())[:5]:  # Show only the first 5 anomalies
            print(f"    Frame {frame}:")
            print(f"      Label: {anomaly['label']}")
            print(f"      Duration: {anomaly['duration']}")
            print(f"      IQR: {anomaly['iqr']:.2f}")
    
    # Isolation Forest method
    print("\nIsolation Forest Method:")
    try:
        anomalies = stats.detect_anomalies(data, method="isolation_forest")
        if not anomalies:
            print("  No anomalies detected.")
        else:
            print(f"  Detected {len(anomalies)} anomalies:")
            for frame, anomaly in sorted(anomalies.items())[:5]:  # Show only the first 5 anomalies
                print(f"    Frame {frame}:")
                print(f"      Label: {anomaly['label']}")
                print(f"      Duration: {anomaly['duration']}")
                print(f"      Anomaly score: {anomaly['anomaly_score']:.2f}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        print("  Note: Isolation Forest method requires scikit-learn.")
        print("  Install it with: pip install scikit-learn")
    
    # DBSCAN method
    print("\nDBSCAN Method:")
    try:
        anomalies = stats.detect_anomalies(data, method="dbscan", threshold=0.5)
        if not anomalies:
            print("  No anomalies detected.")
        else:
            print(f"  Detected {len(anomalies)} anomalies:")
            for frame, anomaly in sorted(anomalies.items())[:5]:  # Show only the first 5 anomalies
                print(f"    Frame {frame}:")
                print(f"      Label: {anomaly['label']}")
                print(f"      Duration: {anomaly['duration']}")
                print(f"      Cluster: {anomaly['cluster']}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        print("  Note: DBSCAN method requires scikit-learn.")
        print("  Install it with: pip install scikit-learn")
    
    # LOF method
    print("\nLOF Method:")
    try:
        anomalies = stats.detect_anomalies(data, method="lof")
        if not anomalies:
            print("  No anomalies detected.")
        else:
            print(f"  Detected {len(anomalies)} anomalies:")
            for frame, anomaly in sorted(anomalies.items())[:5]:  # Show only the first 5 anomalies
                print(f"    Frame {frame}:")
                print(f"      Label: {anomaly['label']}")
                print(f"      Duration: {anomaly['duration']}")
                print(f"      Outlier factor: {anomaly['outlier_factor']:.2f}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        print("  Note: LOF method requires scikit-learn.")
        print("  Install it with: pip install scikit-learn")
    
    print("\nAdvanced statistical analysis completed successfully!")

if __name__ == "__main__":
    main()
