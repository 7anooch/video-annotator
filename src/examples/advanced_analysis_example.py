#!/usr/bin/env python3
"""
Advanced analysis example using the enhanced analysis tools.

This script demonstrates how to use the enhanced analysis tools for advanced analysis tasks.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path so that the script can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the enhanced analysis tools
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

def main():
    """Run the advanced analysis example."""
    print("Advanced Analysis Example")
    print("=======================")
    
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python advanced_analysis_example.py <annotation_file>")
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
    
    # Perform advanced sequence analysis
    print("\nAdvanced Sequence Analysis:")
    sequence_analysis = stats.advanced_sequence_analysis(data)
    
    # Show transition probabilities
    print("\nTransition Probabilities:")
    for (from_label, to_label), prob in sorted(sequence_analysis['transition_probabilities'].items()):
        print(f"  {from_label} -> {to_label}: {prob:.2f}")
    
    # Show n-grams
    print("\nN-grams:")
    for n, ngram_counts in sequence_analysis['ngrams'].items():
        print(f"  {n}-grams:")
        for ngram, count in sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {ngram}: {count}")
    
    # Show complexity measures
    print("\nComplexity Measures:")
    complexity = sequence_analysis['complexity']
    if complexity:
        print(f"  Label entropy: {complexity.get('label_entropy', 0):.2f}")
        print(f"  Bigram entropy: {complexity.get('bigram_entropy', 0):.2f}")
        print(f"  Conditional entropy: {complexity.get('conditional_entropy', 0):.2f}")
    
    # Show recurring patterns
    print("\nRecurring Patterns:")
    for pattern in sequence_analysis['recurring_patterns']:
        print(f"  Pattern: {pattern['pattern']}")
        print(f"    Length: {pattern['length']}")
        print(f"    Occurrences: {pattern['occurrences']}")
        print(f"    Positions: {pattern['positions']}")
    
    # Perform time series analysis
    print("\nTime Series Analysis:")
    time_series = stats.time_series_analysis(data)
    
    # Show windows
    print("\nWindows:")
    for i, (start, end) in enumerate(time_series['windows']):
        print(f"  Window {i+1}: {start} - {end}")
    
    # Show frequencies
    print("\nFrequencies:")
    for label, freq in time_series['frequencies'].items():
        print(f"  {label}: {freq}")
    
    # Show trends
    print("\nTrends:")
    for label, trend in time_series['trends'].items():
        print(f"  {label}:")
        print(f"    Slope: {trend['slope']:.2f}")
        print(f"    Intercept: {trend['intercept']:.2f}")
        print(f"    Increasing: {trend['increasing']}")
        print(f"    Decreasing: {trend['decreasing']}")
    
    # Detect anomalies
    print("\nAnomaly Detection:")
    
    # Z-score method
    print("\nZ-score Method:")
    anomalies = stats.detect_anomalies(data, method="zscore", threshold=3.0)
    if not anomalies:
        print("  No anomalies detected.")
    else:
        for frame, anomaly in sorted(anomalies.items()):
            print(f"  Frame {frame}:")
            print(f"    Label: {anomaly['label']}")
            print(f"    Duration: {anomaly['duration']}")
            print(f"    Z-score: {anomaly['zscore']:.2f}")
            print(f"    Mean duration: {anomaly['mean_duration']:.2f}")
            print(f"    Std duration: {anomaly['std_duration']:.2f}")
    
    # IQR method
    print("\nIQR Method:")
    anomalies = stats.detect_anomalies(data, method="iqr", threshold=1.5)
    if not anomalies:
        print("  No anomalies detected.")
    else:
        for frame, anomaly in sorted(anomalies.items()):
            print(f"  Frame {frame}:")
            print(f"    Label: {anomaly['label']}")
            print(f"    Duration: {anomaly['duration']}")
            print(f"    Q1: {anomaly['q1']:.2f}")
            print(f"    Q3: {anomaly['q3']:.2f}")
            print(f"    IQR: {anomaly['iqr']:.2f}")
            print(f"    Lower bound: {anomaly['lower_bound']:.2f}")
            print(f"    Upper bound: {anomaly['upper_bound']:.2f}")
    
    # Isolation Forest method
    try:
        print("\nIsolation Forest Method:")
        anomalies = stats.detect_anomalies(data, method="isolation_forest")
        if not anomalies:
            print("  No anomalies detected.")
        else:
            for frame, anomaly in sorted(anomalies.items()):
                print(f"  Frame {frame}:")
                print(f"    Label: {anomaly['label']}")
                print(f"    Duration: {anomaly['duration']}")
                print(f"    Anomaly score: {anomaly['anomaly_score']:.2f}")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    print("\nAdvanced analysis completed successfully!")

if __name__ == "__main__":
    main()
