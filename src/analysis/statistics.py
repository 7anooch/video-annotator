#!/usr/bin/env python3
"""
Enhanced statistical analysis for annotations that complements existing tools.

This module provides enhanced statistical analysis functions for annotation data
that complement and extend the functionality in the existing analysis tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel
from src.analysis.data_model import AnnotationData

# Set up logger
logger = setup_logger('statistics')

class StatisticalAnalysis:
    """
    Enhanced statistical analysis for annotations that complements existing tools.

    This class provides enhanced statistical analysis functions for annotation data
    that complement and extend the functionality in the existing analysis tools.

    Attributes:
        logger: The logger instance
        analyze_adapter: Adapter for the analyze.py module
    """

    def __init__(self):
        """Initialize the StatisticalAnalysis."""
        self.logger = setup_logger('statistical_analysis')

        # Import the analyze adapter
        from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
        self.analyze_adapter = AnalyzeAdapter()

    @exception_handler
    def basic_statistics(self, data: AnnotationData) -> Dict[str, Any]:
        """
        Calculate basic statistics for annotation data.

        Args:
            data (AnnotationData): Annotation data

        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frame numbers and labels
        frames = data.get_frames()
        labels = data.get_labels()
        label_counts = data.get_label_counts()

        # Calculate frame statistics
        frame_min = min(frames) if frames else 0
        frame_max = max(frames) if frames else 0
        frame_range = frame_max - frame_min if frames else 0

        # Calculate gaps
        gaps = []
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i-1] - 1
            if gap > 0:
                gaps.append(gap)

        # Calculate statistics
        stats = {
            'total_annotations': data.get_annotation_count(),
            'unique_labels': len(labels),
            'label_counts': label_counts,
            'frame_min': frame_min,
            'frame_max': frame_max,
            'frame_range': frame_range,
            'total_gaps': len(gaps),
            'total_gap_frames': sum(gaps) if gaps else 0,
            'avg_gap_size': np.mean(gaps) if gaps else 0,
            'max_gap_size': max(gaps) if gaps else 0
        }

        self.logger.info(f"Calculated basic statistics for {data.get_annotation_count()} annotations")
        return stats

    @exception_handler
    def label_transitions(self, data: AnnotationData) -> Dict[Tuple[str, str], int]:
        """
        Calculate transitions between labels.

        Args:
            data (AnnotationData): Annotation data

        Returns:
            Dict[Tuple[str, str], int]: Dictionary of transitions with (from_label, to_label) as keys
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frames and sort them
        frames = data.get_frames()

        # Count transitions
        transitions = {}
        prev_label = None

        for frame in frames:
            annotation = data.get_annotation(frame)
            if annotation and 'label' in annotation:
                current_label = annotation['label']

                if prev_label is not None:
                    transition = (prev_label, current_label)
                    transitions[transition] = transitions.get(transition, 0) + 1

                prev_label = current_label

        self.logger.info(f"Calculated {len(transitions)} label transitions")
        return transitions

    @exception_handler
    def label_durations(self, data: AnnotationData) -> Dict[str, List[int]]:
        """
        Calculate durations of continuous label segments.

        Args:
            data (AnnotationData): Annotation data

        Returns:
            Dict[str, List[int]]: Dictionary of durations with labels as keys
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frames and sort them
        frames = data.get_frames()

        # Calculate durations
        durations = {}
        current_label = None
        current_start = None

        for i, frame in enumerate(frames):
            annotation = data.get_annotation(frame)
            if annotation and 'label' in annotation:
                label = annotation['label']

                # If this is a new label or the first annotation
                if label != current_label or current_start is None:
                    # If we were tracking a label, calculate its duration
                    if current_label is not None and current_start is not None:
                        duration = frame - current_start
                        if current_label not in durations:
                            durations[current_label] = []
                        durations[current_label].append(duration)

                    # Start tracking the new label
                    current_label = label
                    current_start = frame

        # Handle the last segment
        if current_label is not None and current_start is not None and frames:
            duration = frames[-1] - current_start + 1
            if current_label not in durations:
                durations[current_label] = []
            durations[current_label].append(duration)

        self.logger.info(f"Calculated durations for {len(durations)} labels")
        return durations

    @exception_handler
    def duration_statistics(self, data: AnnotationData) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for label durations.

        Args:
            data (AnnotationData): Annotation data

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of statistics with labels as keys
        """
        # Get durations
        durations = self.label_durations(data)

        # Calculate statistics
        stats = {}
        for label, duration_list in durations.items():
            if duration_list:
                stats[label] = {
                    'count': len(duration_list),
                    'total_frames': sum(duration_list),
                    'min': min(duration_list),
                    'max': max(duration_list),
                    'mean': np.mean(duration_list),
                    'median': np.median(duration_list),
                    'std': np.std(duration_list)
                }

        self.logger.info(f"Calculated duration statistics for {len(stats)} labels")
        return stats

    @exception_handler
    def compare_annotations(self, data1: AnnotationData, data2: AnnotationData) -> Dict[str, Any]:
        """
        Compare two sets of annotations.

        Args:
            data1 (AnnotationData): First annotation data
            data2 (AnnotationData): Second annotation data

        Returns:
            Dict[str, Any]: Comparison results
        """
        if data1.get_annotation_count() == 0 or data2.get_annotation_count() == 0:
            self.logger.warning("Cannot compare empty annotation sets")
            return {}

        # Get frames
        frames1 = set(data1.get_frames())
        frames2 = set(data2.get_frames())
        common_frames = frames1.intersection(frames2)

        # Calculate agreement
        agreements = 0
        disagreements = 0

        for frame in common_frames:
            annotation1 = data1.get_annotation(frame)
            annotation2 = data2.get_annotation(frame)

            if annotation1 and annotation2:
                label1 = annotation1.get('label')
                label2 = annotation2.get('label')

                if label1 == label2:
                    agreements += 1
                else:
                    disagreements += 1

        # Calculate statistics
        total_common = len(common_frames)
        agreement_rate = agreements / total_common if total_common > 0 else 0

        # Calculate unique frames
        unique_to_1 = frames1 - frames2
        unique_to_2 = frames2 - frames1

        comparison = {
            'total_annotations_1': data1.get_annotation_count(),
            'total_annotations_2': data2.get_annotation_count(),
            'common_frames': total_common,
            'agreements': agreements,
            'disagreements': disagreements,
            'agreement_rate': agreement_rate,
            'unique_to_1': len(unique_to_1),
            'unique_to_2': len(unique_to_2)
        }

        self.logger.info(f"Compared annotations with {total_common} common frames")
        return comparison

    @exception_handler
    def advanced_sequence_analysis(self, data: AnnotationData) -> Dict[str, Any]:
        """
        Perform advanced sequence analysis on annotation data.
        This extends the existing sequence analysis functionality with more sophisticated techniques.

        Args:
            data (AnnotationData): Annotation data

        Returns:
            Dict[str, Any]: Advanced sequence analysis results
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frames and labels
        frames, labels = data.to_frame_label_lists()

        # Use the existing sequence analysis functionality through the adapter
        basic_results = self.analyze_adapter.analyze_sequence(data.annotations)

        # Enhance with additional analysis

        # 1. Calculate transition probabilities
        transition_probs = {}
        prev_label = None

        for label in labels:
            if prev_label is not None:
                transition = (prev_label, label)
                transition_probs[transition] = transition_probs.get(transition, 0) + 1
            prev_label = label

        # Convert counts to probabilities
        for label in set(labels):
            total_transitions_from_label = sum(count for (from_label, _), count in transition_probs.items() if from_label == label)
            if total_transitions_from_label > 0:
                for (from_label, to_label), count in list(transition_probs.items()):
                    if from_label == label:
                        transition_probs[(from_label, to_label)] = count / total_transitions_from_label

        # 2. Calculate n-grams (sequences of length n)
        ngrams = {}
        for n in range(2, 6):  # 2-grams to 5-grams
            if len(labels) >= n:
                ngram_counts = {}
                for i in range(len(labels) - n + 1):
                    ngram = tuple(labels[i:i+n])
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                ngrams[n] = ngram_counts

        # 3. Calculate sequence complexity using entropy
        try:
            from scipy.stats import entropy

            # Calculate label entropy
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            label_probs = [count / len(labels) for count in label_counts.values()]
            label_entropy = entropy(label_probs, base=2)

            # Calculate transition entropy
            if len(labels) > 1:
                bigram_counts = ngrams.get(2, {})
                if bigram_counts:
                    bigram_probs = [count / (len(labels) - 1) for count in bigram_counts.values()]
                    bigram_entropy = entropy(bigram_probs, base=2)
                else:
                    bigram_entropy = 0.0
            else:
                bigram_entropy = 0.0

            # Calculate conditional entropy
            conditional_entropy = max(0, bigram_entropy - label_entropy)

            complexity = {
                'label_entropy': float(label_entropy),
                'bigram_entropy': float(bigram_entropy),
                'conditional_entropy': float(conditional_entropy)
            }
        except ImportError:
            self.logger.warning("scipy is required for entropy calculation")
            complexity = {}
        except Exception as e:
            self.logger.error(f"Error calculating sequence complexity: {str(e)}")
            complexity = {}

        # 4. Identify recurring patterns using suffix trees
        try:
            # Simple implementation of pattern finding
            recurring_patterns = []
            min_pattern_length = 2
            max_pattern_length = min(10, len(labels) // 2)

            for pattern_length in range(min_pattern_length, max_pattern_length + 1):
                patterns = {}
                for i in range(len(labels) - pattern_length + 1):
                    pattern = tuple(labels[i:i+pattern_length])
                    if pattern in patterns:
                        patterns[pattern].append(i)
                    else:
                        patterns[pattern] = [i]

                # Keep patterns that occur more than once
                for pattern, positions in patterns.items():
                    if len(positions) > 1:
                        recurring_patterns.append({
                            'pattern': pattern,
                            'length': pattern_length,
                            'occurrences': len(positions),
                            'positions': positions
                        })

            # Sort by number of occurrences (most frequent first)
            recurring_patterns.sort(key=lambda x: x['occurrences'], reverse=True)

            # Keep only the top 10 patterns
            recurring_patterns = recurring_patterns[:10]
        except Exception as e:
            self.logger.error(f"Error identifying recurring patterns: {str(e)}")
            recurring_patterns = []

        # Combine all results
        results = {
            'basic_analysis': basic_results,
            'transition_probabilities': transition_probs,
            'ngrams': ngrams,
            'complexity': complexity,
            'recurring_patterns': recurring_patterns
        }

        self.logger.info(f"Performed advanced sequence analysis on {len(labels)} labels")
        return results

    @exception_handler
    def correlation_analysis(self, data1: AnnotationData, data2: AnnotationData) -> Dict[str, float]:
        """
        Calculate correlation between two annotation sets.

        Args:
            data1 (AnnotationData): First annotation data
            data2 (AnnotationData): Second annotation data

        Returns:
            Dict[str, float]: Correlation results
        """
        if data1.get_annotation_count() == 0 or data2.get_annotation_count() == 0:
            self.logger.warning("Cannot calculate correlation for empty annotation sets")
            return {}

        # Get all frames from both datasets
        all_frames = sorted(set(data1.get_frames()).union(set(data2.get_frames())))

        # Create DataFrames for both datasets
        df1 = data1.to_dataframe()
        df2 = data2.to_dataframe()

        # Check if 'label' column exists in both DataFrames
        if 'label' not in df1.columns or 'label' not in df2.columns:
            self.logger.warning("Both annotation sets must have a 'label' column for correlation analysis")
            return {}

        # Get all unique labels
        all_labels = sorted(set(df1['label'].unique()).union(set(df2['label'].unique())))

        # Create label mapping
        label_to_index = {label: i for i, label in enumerate(all_labels)}

        # Create arrays for correlation analysis
        values1 = np.zeros(len(all_frames))
        values2 = np.zeros(len(all_frames))

        # Fill arrays with label indices
        for i, frame in enumerate(all_frames):
            annotation1 = data1.get_annotation(frame)
            annotation2 = data2.get_annotation(frame)

            if annotation1 and 'label' in annotation1:
                values1[i] = label_to_index[annotation1['label']]

            if annotation2 and 'label' in annotation2:
                values2[i] = label_to_index[annotation2['label']]

        # Calculate correlation
        try:
            pearson_corr = np.corrcoef(values1, values2)[0, 1]
        except:
            pearson_corr = np.nan

        # Calculate Spearman correlation
        try:
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(values1, values2)
        except:
            spearman_corr = np.nan

        # Calculate Cohen's kappa
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(values1, values2)
        except:
            kappa = np.nan

        results = {
            'pearson_correlation': float(pearson_corr),
            'spearman_correlation': float(spearman_corr),
            'cohen_kappa': float(kappa)
        }

        self.logger.info(f"Calculated correlation between annotation sets")
        return results

    @exception_handler
    def time_series_analysis(self, data: AnnotationData) -> Dict[str, Any]:
        """
        Perform enhanced time series analysis on annotation data.
        This extends the existing functionality by providing more detailed time series analysis.

        Args:
            data (AnnotationData): Annotation data

        Returns:
            Dict[str, Any]: Time series analysis results
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frames and sort them
        frames = data.get_frames()

        # Get all unique labels
        labels = data.get_labels()

        # Create a DataFrame for time series analysis
        df = data.to_dataframe()

        # Check if 'label' column exists
        if 'label' not in df.columns:
            self.logger.warning("Annotation data must have a 'label' column for time series analysis")
            return {}

        # Calculate label frequencies over time
        window_size = max(10, len(frames) // 10)  # Use 10% of frames or at least 10 frames

        # Create windows
        windows = []
        for i in range(0, len(frames), window_size):
            window_frames = frames[i:i+window_size]
            if window_frames:
                windows.append((window_frames[0], window_frames[-1]))

        # Calculate label frequencies for each window
        frequencies = {}
        for label in labels:
            frequencies[label] = []

        for start_frame, end_frame in windows:
            window_data = data.filter_by_frames(start_frame, end_frame)
            counts = window_data.get_label_counts()

            for label in labels:
                frequencies[label].append(counts.get(label, 0))

        # Calculate trend (simple linear regression)
        trends = {}
        for label, freq in frequencies.items():
            if len(freq) > 1:
                x = np.arange(len(freq))
                y = np.array(freq)

                # Calculate slope and intercept
                slope, intercept = np.polyfit(x, y, 1)

                trends[label] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'increasing': slope > 0,
                    'decreasing': slope < 0
                }

        # Calculate autocorrelation
        autocorrelations = {}
        for label, freq in frequencies.items():
            if len(freq) > 1:
                try:
                    from statsmodels.tsa.stattools import acf
                    autocorr = acf(freq, nlags=min(5, len(freq) - 1))
                    autocorrelations[label] = [float(x) for x in autocorr]
                except:
                    autocorrelations[label] = []

        results = {
            'windows': windows,
            'frequencies': frequencies,
            'trends': trends,
            'autocorrelations': autocorrelations
        }

        self.logger.info(f"Performed time series analysis on {data.get_annotation_count()} annotations")
        return results

    @exception_handler
    def detect_change_points(self, data: AnnotationData, method: str = 'binary_segmentation', penalty: str = 'bic') -> Dict[str, Any]:
        """
        Detect change points in annotation data using advanced statistical methods.
        This method identifies points where the statistical properties of the annotation sequence change.

        Args:
            data (AnnotationData): Annotation data
            method (str, optional): Change point detection method ('binary_segmentation', 'window', or 'pelt').
                                    Defaults to 'binary_segmentation'.
            penalty (str, optional): Penalty for adding change points ('bic', 'aic', or 'manual'). Defaults to 'bic'.

        Returns:
            Dict[str, Any]: Change point detection results including change points and segments
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        try:
            import ruptures as rpt
        except ImportError:
            self.logger.error("ruptures is required for change point detection. Install it with: conda install -c conda-forge ruptures")
            return {}

        # Get frames and labels
        frames, labels = data.to_frame_label_lists()

        # Convert labels to numeric values
        unique_labels = sorted(set(labels))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_index[label] for label in labels])

        # Detect change points
        try:
            # Choose the algorithm based on the method parameter
            if method == 'binary_segmentation':
                algo = rpt.Binseg(model="rbf").fit(numeric_labels.reshape(-1, 1))
            elif method == 'window':
                algo = rpt.Window(width=40, model="rbf").fit(numeric_labels.reshape(-1, 1))
            elif method == 'pelt':
                algo = rpt.Pelt(model="rbf").fit(numeric_labels.reshape(-1, 1))
            else:
                self.logger.warning(f"Unknown method: {method}. Using binary segmentation.")
                algo = rpt.Binseg(model="rbf").fit(numeric_labels.reshape(-1, 1))

            # Choose the penalty based on the penalty parameter
            if penalty == 'bic':
                change_points = algo.predict(pen=0.5)
            elif penalty == 'aic':
                change_points = algo.predict(pen=0.1)
            elif penalty == 'manual':
                # Estimate the number of change points using the elbow method
                n_bkps_max = min(20, len(numeric_labels) // 10)  # Maximum number of change points
                bkps, costs = algo.predict_bkps_and_costs(n_bkps_max=n_bkps_max)

                # Find the elbow point (where the cost reduction starts to diminish)
                cost_diffs = np.diff(costs)
                elbow_idx = np.argmax(cost_diffs) + 1
                change_points = algo.predict(n_bkps=elbow_idx)
            else:
                self.logger.warning(f"Unknown penalty: {penalty}. Using BIC.")
                change_points = algo.predict(pen=0.5)

            # Convert change points to frame numbers
            change_point_frames = [frames[cp - 1] if cp < len(frames) else frames[-1] for cp in change_points[:-1]]

            # Create segments
            segments = []
            start_idx = 0
            for cp in change_points[:-1]:
                segment = {
                    'start_frame': frames[start_idx],
                    'end_frame': frames[cp - 1] if cp < len(frames) else frames[-1],
                    'start_index': start_idx,
                    'end_index': cp - 1,
                    'length': cp - start_idx
                }

                # Calculate the most common label in this segment
                segment_labels = labels[start_idx:cp]
                label_counts = {}
                for label in segment_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                if label_counts:
                    most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    segment['most_common_label'] = most_common_label
                    segment['label_counts'] = label_counts

                segments.append(segment)
                start_idx = cp

            # Add the last segment
            if start_idx < len(frames):
                segment = {
                    'start_frame': frames[start_idx],
                    'end_frame': frames[-1],
                    'start_index': start_idx,
                    'end_index': len(frames) - 1,
                    'length': len(frames) - start_idx
                }

                # Calculate the most common label in this segment
                segment_labels = labels[start_idx:]
                label_counts = {}
                for label in segment_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                if label_counts:
                    most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    segment['most_common_label'] = most_common_label
                    segment['label_counts'] = label_counts

                segments.append(segment)

            # Generate a visualization of the change points
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                import io
                import base64
                from matplotlib.patches import Rectangle

                # Create a figure
                fig, ax = plt.subplots(figsize=(12, 6))

                # Create a colormap for the labels
                cmap = plt.cm.get_cmap('tab10', len(unique_labels))
                colors = {label: mcolors.rgb2hex(cmap(i)[:3]) for i, label in enumerate(unique_labels)}

                # Plot the segments
                y_height = 0.8
                y_pos = 0.1

                for segment in segments:
                    start = segment['start_index']
                    end = segment['end_index']
                    length = segment['length']
                    label = segment.get('most_common_label', 'unknown')

                    # Get color for this label
                    color = colors.get(label, 'gray')

                    # Add a rectangle for this segment
                    rect = Rectangle((start, y_pos), length, y_height,
                                    facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)

                    # Add label text if segment is wide enough
                    if length > len(frames) * 0.05:  # Only add text if segment is at least 5% of total length
                        ax.text(start + length/2, y_pos + y_height/2, str(label),
                                ha='center', va='center', fontsize=10, color='black')

                # Add change point lines
                for cp in change_points[:-1]:
                    ax.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
                    # Add frame number annotation
                    frame_num = frames[cp-1] if cp < len(frames) else frames[-1]
                    ax.text(cp, y_pos + y_height + 0.05, f"Frame {frame_num}",
                            ha='center', va='bottom', fontsize=8, rotation=90, color='red')

                # Set axis limits
                ax.set_xlim(0, len(frames))
                ax.set_ylim(0, 1)

                # Remove y-axis ticks and labels
                ax.set_yticks([])

                # Set x-axis ticks and labels
                x_ticks = np.linspace(0, len(frames), min(10, len(frames)))
                x_tick_labels = [frames[int(i)] if i < len(frames) else frames[-1] for i in x_ticks]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_tick_labels)

                # Add title and labels
                ax.set_title(f'Change Point Detection ({method}, {penalty})')
                ax.set_xlabel('Frame Number')

                # Add legend
                legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[label],
                                               edgecolor='black', alpha=0.7, label=str(label))
                                  for label in unique_labels]
                ax.legend(handles=legend_elements, loc='upper right', title='Labels')

                # Save the figure to a base64-encoded string
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)

                # Add the visualization to the results
                visualization = f"data:image/png;base64,{img_str}"
            except Exception as e:
                self.logger.warning(f"Error generating change point visualization: {str(e)}")
                visualization = None

            # Format the results
            results = {
                'method': method,
                'penalty': penalty,
                'change_points': change_points[:-1],  # Exclude the last point (end of sequence)
                'change_point_frames': change_point_frames,
                'segments': segments,
                'visualization': visualization
            }

            self.logger.info(f"Detected {len(change_point_frames)} change points using {method} method")
            return results
        except Exception as e:
            self.logger.error(f"Error in change point detection: {str(e)}")
            return {'error': str(e)}

    @exception_handler
    def mcmc_simulation(self, data: AnnotationData, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform Markov Chain Monte Carlo (MCMC) simulation based on annotation data.
        This advanced method can generate synthetic annotation sequences based on the observed data.

        Args:
            data (AnnotationData): Annotation data
            n_samples (int, optional): Number of samples to generate. Defaults to 1000.

        Returns:
            Dict[str, Any]: MCMC simulation results including transition matrix and generated sequence
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frames and labels
        frames, labels = data.to_frame_label_lists()

        # Calculate transition matrix
        unique_labels = sorted(set(labels))
        n_labels = len(unique_labels)

        # Create a mapping from labels to indices
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        index_to_label = {i: label for i, label in enumerate(unique_labels)}

        # Initialize transition count matrix
        transition_counts = np.zeros((n_labels, n_labels))

        # Count transitions
        for i in range(len(labels) - 1):
            from_label = labels[i]
            to_label = labels[i + 1]
            from_idx = label_to_index[from_label]
            to_idx = label_to_index[to_label]
            transition_counts[from_idx, to_idx] += 1

        # Calculate transition probabilities
        transition_matrix = np.zeros((n_labels, n_labels))
        for i in range(n_labels):
            row_sum = np.sum(transition_counts[i])
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum
            else:
                # If no transitions from this state, use uniform distribution
                transition_matrix[i] = np.ones(n_labels) / n_labels

        # Generate synthetic sequence using MCMC
        try:
            # Start with a random state
            import random
            current_state = random.randint(0, n_labels - 1)

            # Generate sequence
            generated_indices = [current_state]
            for _ in range(n_samples - 1):
                # Sample next state based on transition probabilities
                next_state = np.random.choice(n_labels, p=transition_matrix[current_state])
                generated_indices.append(next_state)
                current_state = next_state

            # Convert indices to labels
            generated_labels = [index_to_label[idx] for idx in generated_indices]

            # Calculate statistics of the generated sequence
            generated_label_counts = {}
            for label in generated_labels:
                generated_label_counts[label] = generated_label_counts.get(label, 0) + 1

            # Calculate transition counts in the generated sequence
            generated_transition_counts = np.zeros((n_labels, n_labels))
            for i in range(len(generated_labels) - 1):
                from_label = generated_labels[i]
                to_label = generated_labels[i + 1]
                from_idx = label_to_index[from_label]
                to_idx = label_to_index[to_label]
                generated_transition_counts[from_idx, to_idx] += 1

            # Format the results
            results = {
                'n_samples': n_samples,
                'unique_labels': unique_labels,
                'transition_matrix': transition_matrix.tolist(),
                'generated_labels': generated_labels,
                'generated_label_counts': generated_label_counts,
                'generated_transition_counts': generated_transition_counts.tolist()
            }

            self.logger.info(f"Performed MCMC simulation with {n_samples} samples")
            return results
        except Exception as e:
            self.logger.error(f"Error in MCMC simulation: {str(e)}")
            return {
                'error': str(e),
                'transition_matrix': transition_matrix.tolist()
            }

    @exception_handler
    def segment_transition_matrix(self, data: AnnotationData) -> Dict[str, Any]:
        """
        Calculate transition matrix for grouped segments (not individual frames).
        This is similar to how the analyze_sequence function in analyze.py works.

        Args:
            data (AnnotationData): Annotation data

        Returns:
            Dict[str, Any]: Transition matrix results including matrix, counts, and visualization
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frames and labels
        frames, labels = data.to_frame_label_lists()

        # Group the labels into segments
        segments = []
        if not labels:
            return {}

        current_label = labels[0]
        start_idx = 0

        # Find segments where the label changes
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append((current_label, start_idx, i-1))
                current_label = labels[i]
                start_idx = i

        # Add the last segment
        segments.append((current_label, start_idx, len(labels)-1))

        # Calculate transitions between segments
        unique_labels = sorted(set(labels))
        n_labels = len(unique_labels)

        # Create a mapping from labels to indices
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        index_to_label = {i: label for i, label in enumerate(unique_labels)}

        # Initialize transition count matrix
        transition_counts = np.zeros((n_labels, n_labels))

        # Count transitions between segments
        for i in range(len(segments) - 1):
            from_label = segments[i][0]
            to_label = segments[i+1][0]
            from_idx = label_to_index[from_label]
            to_idx = label_to_index[to_label]
            transition_counts[from_idx, to_idx] += 1

        # Calculate transition probabilities
        transition_matrix = np.zeros((n_labels, n_labels))
        for i in range(n_labels):
            row_sum = np.sum(transition_counts[i])
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum
            else:
                # If no transitions from this state, use uniform distribution
                transition_matrix[i] = np.ones(n_labels) / n_labels

        # Generate a visualization of the transition matrix
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import io
            import base64
            import seaborn as sns

            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create a heatmap of the transition matrix
            sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                       xticklabels=[str(label) for label in unique_labels],
                       yticklabels=[str(label) for label in unique_labels],
                       ax=ax)

            # Add title and labels
            ax.set_title('Segment Transition Matrix')
            ax.set_xlabel('To Label')
            ax.set_ylabel('From Label')

            # Save the figure to a base64-encoded string
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            # Add the visualization to the results
            visualization = f"data:image/png;base64,{img_str}"
        except Exception as e:
            self.logger.warning(f"Error generating transition matrix visualization: {str(e)}")
            visualization = None

        # Format the results
        results = {
            'unique_labels': unique_labels,
            'transition_counts': transition_counts.tolist(),
            'transition_matrix': transition_matrix.tolist(),
            'segments': [(label, end-start+1) for label, start, end in segments],
            'visualization': visualization
        }

        self.logger.info(f"Calculated segment transition matrix with {len(segments)} segments")
        return results

    @exception_handler
    def hmm_analysis(self, data: AnnotationData, n_states: int = 3) -> Dict[str, Any]:
        """
        Perform Hidden Markov Model (HMM) analysis on annotation data.
        This advanced method can identify hidden states and transitions in the annotation sequence.

        Args:
            data (AnnotationData): Annotation data
            n_states (int, optional): Number of hidden states to identify. Defaults to 3.

        Returns:
            Dict[str, Any]: HMM analysis results including hidden states, transition matrix, and emission matrix
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        try:
            from hmmlearn import hmm
            import numpy as np
        except ImportError:
            self.logger.error("hmmlearn is required for HMM analysis. Install it with: conda install -c conda-forge hmmlearn")
            return {}

        # Get frames and labels
        frames, labels = data.to_frame_label_lists()

        # Convert labels to numeric values
        unique_labels = sorted(set(labels))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_index[label] for label in labels]).reshape(-1, 1)

        # Train the HMM model
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
        model.fit(numeric_labels)

        # Get the hidden states
        hidden_states = model.predict(numeric_labels)

        # Get the transition matrix
        transition_matrix = model.transmat_

        # Get the emission matrix (means and covariances of the Gaussian emissions)
        emission_means = model.means_
        emission_covars = model.covars_

        # Map hidden states to frames
        state_frames = {}
        for i, state in enumerate(hidden_states):
            if state not in state_frames:
                state_frames[state] = []
            state_frames[state].append(frames[i])

        # Map hidden states to labels
        state_labels = {}
        for state in range(n_states):
            state_indices = [i for i, s in enumerate(hidden_states) if s == state]
            state_label_counts = {}
            for i in state_indices:
                label = labels[i]
                state_label_counts[label] = state_label_counts.get(label, 0) + 1
            state_labels[state] = state_label_counts

        # Calculate the most likely label for each state
        most_likely_labels = {}
        for state, label_counts in state_labels.items():
            if label_counts:
                most_likely_label = max(label_counts.items(), key=lambda x: x[1])[0]
                most_likely_labels[state] = most_likely_label

        # Format the results
        results = {
            'n_states': n_states,
            'hidden_states': hidden_states.tolist(),
            'transition_matrix': transition_matrix.tolist(),
            'emission_means': emission_means.tolist(),
            'emission_covars': emission_covars.tolist(),
            'state_frames': state_frames,
            'state_labels': state_labels,
            'most_likely_labels': most_likely_labels
        }

        self.logger.info(f"Performed HMM analysis with {n_states} hidden states")
        return results

    @exception_handler
    def detect_anomalies(self, data: AnnotationData, method: str = 'zscore', threshold: float = 3.0) -> Dict[int, Dict[str, Any]]:
        """
        Detect anomalies in annotation data using advanced statistical methods.
        This extends the existing functionality by providing more sophisticated anomaly detection.

        Args:
            data (AnnotationData): Annotation data
            method (str, optional): Anomaly detection method ('zscore', 'iqr', 'isolation_forest', 'dbscan', or 'lof').
                                    Defaults to 'zscore'.
            threshold (float, optional): Anomaly threshold. Defaults to 3.0.

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary of anomalies with frame numbers as keys
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to analyze")
            return {}

        # Get frames and sort them
        frames = data.get_frames()

        # Get durations
        durations = self.label_durations(data)

        # Calculate mean and standard deviation for each label
        duration_stats = {}
        for label, duration_list in durations.items():
            if duration_list:
                duration_stats[label] = {
                    'mean': np.mean(duration_list),
                    'std': np.std(duration_list)
                }

        # Detect anomalies
        anomalies = {}

        if method == 'zscore':
            # Detect anomalies using Z-score
            current_label = None
            current_start = None

            for i, frame in enumerate(frames):
                annotation = data.get_annotation(frame)
                if annotation and 'label' in annotation:
                    label = annotation['label']

                    # If this is a new label or the first annotation
                    if label != current_label or current_start is None:
                        # If we were tracking a label, check if it's an anomaly
                        if current_label is not None and current_start is not None:
                            duration = frame - current_start

                            if current_label in duration_stats:
                                stats = duration_stats[current_label]
                                mean = stats['mean']
                                std = stats['std']

                                if std > 0:
                                    zscore = abs(duration - mean) / std

                                    if zscore > threshold:
                                        anomalies[current_start] = {
                                            'label': current_label,
                                            'duration': duration,
                                            'zscore': float(zscore),
                                            'mean_duration': float(mean),
                                            'std_duration': float(std),
                                            'method': 'zscore'
                                        }

                        # Start tracking the new label
                        current_label = label
                        current_start = frame

            # Handle the last segment
            if current_label is not None and current_start is not None and frames:
                duration = frames[-1] - current_start + 1

                if current_label in duration_stats:
                    stats = duration_stats[current_label]
                    mean = stats['mean']
                    std = stats['std']

                    if std > 0:
                        zscore = abs(duration - mean) / std

                        if zscore > threshold:
                            anomalies[current_start] = {
                                'label': current_label,
                                'duration': duration,
                                'zscore': float(zscore),
                                'mean_duration': float(mean),
                                'std_duration': float(std),
                                'method': 'zscore'
                            }

        elif method == 'iqr':
            # Detect anomalies using Interquartile Range (IQR)
            # This is a more robust method than Z-score for non-normal distributions
            current_label = None
            current_start = None

            # Calculate IQR for each label
            iqr_stats = {}
            for label, duration_list in durations.items():
                if duration_list:
                    q1 = np.percentile(duration_list, 25)
                    q3 = np.percentile(duration_list, 75)
                    iqr = q3 - q1
                    iqr_stats[label] = {
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr
                    }

            # Detect anomalies
            for i, frame in enumerate(frames):
                annotation = data.get_annotation(frame)
                if annotation and 'label' in annotation:
                    label = annotation['label']

                    # If this is a new label or the first annotation
                    if label != current_label or current_start is None:
                        # If we were tracking a label, check if it's an anomaly
                        if current_label is not None and current_start is not None:
                            duration = frame - current_start

                            if current_label in iqr_stats:
                                stats = iqr_stats[current_label]
                                q1 = stats['q1']
                                q3 = stats['q3']
                                iqr = stats['iqr']

                                # Check if duration is an outlier (outside q1 - threshold*iqr or q3 + threshold*iqr)
                                lower_bound = q1 - threshold * iqr
                                upper_bound = q3 + threshold * iqr

                                if duration < lower_bound or duration > upper_bound:
                                    anomalies[current_start] = {
                                        'label': current_label,
                                        'duration': duration,
                                        'q1': float(q1),
                                        'q3': float(q3),
                                        'iqr': float(iqr),
                                        'lower_bound': float(lower_bound),
                                        'upper_bound': float(upper_bound),
                                        'method': 'iqr'
                                    }

                        # Start tracking the new label
                        current_label = label
                        current_start = frame

            # Handle the last segment
            if current_label is not None and current_start is not None and frames:
                duration = frames[-1] - current_start + 1

                if current_label in iqr_stats:
                    stats = iqr_stats[current_label]
                    q1 = stats['q1']
                    q3 = stats['q3']
                    iqr = stats['iqr']

                    # Check if duration is an outlier
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr

                    if duration < lower_bound or duration > upper_bound:
                        anomalies[current_start] = {
                            'label': current_label,
                            'duration': duration,
                            'q1': float(q1),
                            'q3': float(q3),
                            'iqr': float(iqr),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'method': 'iqr'
                        }

        elif method == 'isolation_forest':
            # Detect anomalies using Isolation Forest
            # This is a machine learning approach that works well for complex patterns
            try:
                from sklearn.ensemble import IsolationForest

                # Prepare data for Isolation Forest
                X = []
                frame_map = []

                # For each label, collect durations and corresponding frames
                for label in data.get_labels():
                    label_durations = []
                    label_frames = []

                    current_label = None
                    current_start = None

                    for frame in frames:
                        annotation = data.get_annotation(frame)
                        if annotation and 'label' in annotation:
                            current_annotation_label = annotation['label']

                            # If this is a new label or the first annotation
                            if current_annotation_label != current_label or current_start is None:
                                # If we were tracking a label and it matches the current label we're analyzing
                                if current_label == label and current_start is not None:
                                    duration = frame - current_start
                                    label_durations.append(duration)
                                    label_frames.append(current_start)

                                # Start tracking the new label
                                current_label = current_annotation_label
                                current_start = frame

                    # Handle the last segment
                    if current_label == label and current_start is not None and frames:
                        duration = frames[-1] - current_start + 1
                        label_durations.append(duration)
                        label_frames.append(current_start)

                    # Add to the overall data
                    for duration, frame in zip(label_durations, label_frames):
                        X.append([duration])
                        frame_map.append((frame, label))

                # If we have enough data, run Isolation Forest
                if len(X) > 10:  # Need a reasonable amount of data
                    # Train Isolation Forest
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    clf.fit(X)

                    # Predict anomalies
                    y_pred = clf.predict(X)
                    scores = clf.decision_function(X)

                    # Collect anomalies
                    for i, (pred, score) in enumerate(zip(y_pred, scores)):
                        if pred == -1:  # -1 indicates an anomaly
                            frame, label = frame_map[i]
                            duration = X[i][0]

                            anomalies[frame] = {
                                'label': label,
                                'duration': duration,
                                'anomaly_score': float(score),
                                'method': 'isolation_forest'
                            }
            except ImportError:
                self.logger.warning("scikit-learn is required for Isolation Forest anomaly detection")
            except Exception as e:
                self.logger.error(f"Error in Isolation Forest anomaly detection: {str(e)}")

        elif method == 'dbscan':
            # Detect anomalies using DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
            # This is a density-based clustering algorithm that can identify outliers
            try:
                from sklearn.cluster import DBSCAN

                # Prepare data for DBSCAN
                X = []
                frame_map = []

                # For each label, collect durations and corresponding frames
                for label in data.get_labels():
                    label_durations = []
                    label_frames = []

                    current_label = None
                    current_start = None

                    for frame in frames:
                        annotation = data.get_annotation(frame)
                        if annotation and 'label' in annotation:
                            current_annotation_label = annotation['label']

                            # If this is a new label or the first annotation
                            if current_annotation_label != current_label or current_start is None:
                                # If we were tracking a label and it matches the current label we're analyzing
                                if current_label == label and current_start is not None:
                                    duration = frame - current_start
                                    label_durations.append(duration)
                                    label_frames.append(current_start)

                                # Start tracking the new label
                                current_label = current_annotation_label
                                current_start = frame

                    # Handle the last segment
                    if current_label == label and current_start is not None and frames:
                        duration = frames[-1] - current_start + 1
                        label_durations.append(duration)
                        label_frames.append(current_start)

                    # Add to the overall data
                    for duration, frame in zip(label_durations, label_frames):
                        X.append([duration])
                        frame_map.append((frame, label))

                # If we have enough data, run DBSCAN
                if len(X) > 10:  # Need a reasonable amount of data
                    # Normalize the data
                    X_array = np.array(X)
                    mean = np.mean(X_array, axis=0)
                    std = np.std(X_array, axis=0)
                    X_norm = (X_array - mean) / std

                    # Run DBSCAN
                    # eps is the maximum distance between two samples for them to be considered as in the same neighborhood
                    # min_samples is the number of samples in a neighborhood for a point to be considered as a core point
                    dbscan = DBSCAN(eps=threshold, min_samples=3)
                    labels = dbscan.fit_predict(X_norm)

                    # Collect anomalies (points labeled as -1 are outliers)
                    for i, cluster_label in enumerate(labels):
                        if cluster_label == -1:  # -1 indicates an outlier
                            frame, label = frame_map[i]
                            duration = X[i][0]

                            anomalies[frame] = {
                                'label': label,
                                'duration': duration,
                                'cluster': int(cluster_label),
                                'method': 'dbscan'
                            }
            except ImportError:
                self.logger.warning("scikit-learn is required for DBSCAN anomaly detection")
            except Exception as e:
                self.logger.error(f"Error in DBSCAN anomaly detection: {str(e)}")

        elif method == 'lof':
            # Detect anomalies using LOF (Local Outlier Factor)
            # This is a density-based algorithm that computes the local density deviation of a point with respect to its neighbors
            try:
                from sklearn.neighbors import LocalOutlierFactor

                # Prepare data for LOF
                X = []
                frame_map = []

                # For each label, collect durations and corresponding frames
                for label in data.get_labels():
                    label_durations = []
                    label_frames = []

                    current_label = None
                    current_start = None

                    for frame in frames:
                        annotation = data.get_annotation(frame)
                        if annotation and 'label' in annotation:
                            current_annotation_label = annotation['label']

                            # If this is a new label or the first annotation
                            if current_annotation_label != current_label or current_start is None:
                                # If we were tracking a label and it matches the current label we're analyzing
                                if current_label == label and current_start is not None:
                                    duration = frame - current_start
                                    label_durations.append(duration)
                                    label_frames.append(current_start)

                                # Start tracking the new label
                                current_label = current_annotation_label
                                current_start = frame

                    # Handle the last segment
                    if current_label == label and current_start is not None and frames:
                        duration = frames[-1] - current_start + 1
                        label_durations.append(duration)
                        label_frames.append(current_start)

                    # Add to the overall data
                    for duration, frame in zip(label_durations, label_frames):
                        X.append([duration])
                        frame_map.append((frame, label))

                # If we have enough data, run LOF
                if len(X) > 10:  # Need a reasonable amount of data
                    # Run LOF
                    # n_neighbors is the number of neighbors to consider
                    # contamination is the expected proportion of outliers in the data set
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                    y_pred = lof.fit_predict(X)

                    # Get the negative outlier factor (higher values indicate more anomalous points)
                    negative_outlier_factor = lof.negative_outlier_factor_

                    # Collect anomalies (points with negative_outlier_factor < -threshold are outliers)
                    for i, (pred, score) in enumerate(zip(y_pred, negative_outlier_factor)):
                        if pred == -1:  # -1 indicates an outlier
                            frame, label = frame_map[i]
                            duration = X[i][0]

                            anomalies[frame] = {
                                'label': label,
                                'duration': duration,
                                'outlier_factor': float(-score),  # Convert to positive value for easier interpretation
                                'method': 'lof'
                            }
            except ImportError:
                self.logger.warning("scikit-learn is required for LOF anomaly detection")
            except Exception as e:
                self.logger.error(f"Error in LOF anomaly detection: {str(e)}")

        self.logger.info(f"Detected {len(anomalies)} anomalies using {method} method")
        return anomalies
