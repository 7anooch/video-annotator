#!/usr/bin/env python3
"""
Memory usage tracking utilities for the Video Annotator.

This module provides tools for tracking memory usage in the application.
"""

import os
import sys
import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel

# Set up logger
logger = setup_logger('memory_tracker')

class MemorySnapshot:
    """
    A snapshot of memory usage at a specific point in time.
    
    Attributes:
        timestamp (float): The time when the snapshot was taken
        rss (int): Resident Set Size (RSS) in bytes
        vms (int): Virtual Memory Size (VMS) in bytes
        shared (int): Shared memory size in bytes
        data (int): Data memory size in bytes
        percent (float): Memory usage as a percentage of total system memory
    """
    
    def __init__(self, process: Optional[psutil.Process] = None):
        """
        Initialize a MemorySnapshot.
        
        Args:
            process (psutil.Process, optional): The process to snapshot. 
                                               If None, the current process is used.
        """
        self.timestamp = time.time()
        
        if process is None:
            process = psutil.Process(os.getpid())
        
        # Get memory info
        mem_info = process.memory_info()
        
        # Store memory metrics
        self.rss = mem_info.rss  # Resident Set Size
        self.vms = mem_info.vms  # Virtual Memory Size
        
        # These may not be available on all platforms
        self.shared = getattr(mem_info, 'shared', 0)
        self.data = getattr(mem_info, 'data', 0)
        
        # Get memory usage as percentage
        self.percent = process.memory_percent()
    
    def __str__(self) -> str:
        """Return a string representation of the memory snapshot."""
        return (f"Memory Snapshot at {self.timestamp}:\n"
                f"  RSS: {self.format_bytes(self.rss)}\n"
                f"  VMS: {self.format_bytes(self.vms)}\n"
                f"  Shared: {self.format_bytes(self.shared)}\n"
                f"  Data: {self.format_bytes(self.data)}\n"
                f"  Percent: {self.percent:.2f}%")
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """
        Format bytes as a human-readable string.
        
        Args:
            bytes_value (int): The number of bytes
            
        Returns:
            str: A human-readable string representation of the bytes
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        """
        Convert the snapshot to a dictionary.
        
        Returns:
            dict: A dictionary representation of the snapshot
        """
        return {
            'timestamp': self.timestamp,
            'rss': self.rss,
            'vms': self.vms,
            'shared': self.shared,
            'data': self.data,
            'percent': self.percent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Union[float, int]]) -> 'MemorySnapshot':
        """
        Create a MemorySnapshot from a dictionary.
        
        Args:
            data (dict): A dictionary representation of a snapshot
            
        Returns:
            MemorySnapshot: A new MemorySnapshot instance
        """
        snapshot = cls.__new__(cls)
        snapshot.timestamp = data['timestamp']
        snapshot.rss = data['rss']
        snapshot.vms = data['vms']
        snapshot.shared = data['shared']
        snapshot.data = data['data']
        snapshot.percent = data['percent']
        return snapshot

class MemoryTracker:
    """
    Track memory usage over time.
    
    Attributes:
        snapshots (List[MemorySnapshot]): List of memory snapshots
        tracking (bool): Whether tracking is currently active
        tracking_thread (threading.Thread): Thread for continuous tracking
        tracking_interval (float): Interval between snapshots in seconds
    """
    
    def __init__(self):
        """Initialize a MemoryTracker."""
        self.logger = setup_logger('memory_tracker')
        self.snapshots: List[MemorySnapshot] = []
        self.tracking = False
        self.tracking_thread: Optional[threading.Thread] = None
        self.tracking_interval = 1.0  # Default: 1 second
    
    def take_snapshot(self) -> MemorySnapshot:
        """
        Take a snapshot of the current memory usage.
        
        Returns:
            MemorySnapshot: A snapshot of the current memory usage
        """
        snapshot = MemorySnapshot()
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_tracking(self, interval: float = 1.0) -> None:
        """
        Start continuous memory tracking.
        
        Args:
            interval (float, optional): Interval between snapshots in seconds. 
                                       Defaults to 1.0.
        """
        if self.tracking:
            self.logger.warning("Memory tracking is already active")
            return
        
        self.tracking_interval = interval
        self.tracking = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        self.logger.info(f"Started memory tracking with interval {interval} seconds")
    
    def stop_tracking(self) -> None:
        """Stop continuous memory tracking."""
        if not self.tracking:
            self.logger.warning("Memory tracking is not active")
            return
        
        self.tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
            self.tracking_thread = None
        
        self.logger.info(f"Stopped memory tracking. Collected {len(self.snapshots)} snapshots")
    
    def _tracking_loop(self) -> None:
        """Internal method for continuous tracking."""
        while self.tracking:
            try:
                self.take_snapshot()
                time.sleep(self.tracking_interval)
            except Exception as e:
                self.logger.error(f"Error in memory tracking loop: {str(e)}")
                break
    
    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self.snapshots.clear()
        self.logger.info("Cleared all memory snapshots")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of memory usage.
        
        Returns:
            dict: A summary of memory usage statistics
        """
        if not self.snapshots:
            return {
                'count': 0,
                'duration': 0,
                'rss': {'min': 0, 'max': 0, 'avg': 0, 'start': 0, 'end': 0},
                'vms': {'min': 0, 'max': 0, 'avg': 0, 'start': 0, 'end': 0},
                'percent': {'min': 0, 'max': 0, 'avg': 0, 'start': 0, 'end': 0}
            }
        
        # Extract data
        timestamps = [s.timestamp for s in self.snapshots]
        rss_values = [s.rss for s in self.snapshots]
        vms_values = [s.vms for s in self.snapshots]
        percent_values = [s.percent for s in self.snapshots]
        
        # Calculate statistics
        duration = max(timestamps) - min(timestamps)
        
        return {
            'count': len(self.snapshots),
            'duration': duration,
            'rss': {
                'min': min(rss_values),
                'max': max(rss_values),
                'avg': np.mean(rss_values),
                'start': self.snapshots[0].rss,
                'end': self.snapshots[-1].rss,
                'diff': self.snapshots[-1].rss - self.snapshots[0].rss
            },
            'vms': {
                'min': min(vms_values),
                'max': max(vms_values),
                'avg': np.mean(vms_values),
                'start': self.snapshots[0].vms,
                'end': self.snapshots[-1].vms,
                'diff': self.snapshots[-1].vms - self.snapshots[0].vms
            },
            'percent': {
                'min': min(percent_values),
                'max': max(percent_values),
                'avg': np.mean(percent_values),
                'start': self.snapshots[0].percent,
                'end': self.snapshots[-1].percent,
                'diff': self.snapshots[-1].percent - self.snapshots[0].percent
            }
        }
    
    def print_summary(self) -> None:
        """Print a summary of memory usage."""
        summary = self.get_summary()
        
        if summary['count'] == 0:
            print("No memory snapshots available")
            return
        
        print(f"Memory Usage Summary ({summary['count']} snapshots over {summary['duration']:.2f} seconds):")
        print(f"  RSS (Resident Set Size):")
        print(f"    Min: {MemorySnapshot.format_bytes(summary['rss']['min'])}")
        print(f"    Max: {MemorySnapshot.format_bytes(summary['rss']['max'])}")
        print(f"    Avg: {MemorySnapshot.format_bytes(summary['rss']['avg'])}")
        print(f"    Start: {MemorySnapshot.format_bytes(summary['rss']['start'])}")
        print(f"    End: {MemorySnapshot.format_bytes(summary['rss']['end'])}")
        print(f"    Diff: {MemorySnapshot.format_bytes(summary['rss']['diff'])}")
        
        print(f"  VMS (Virtual Memory Size):")
        print(f"    Min: {MemorySnapshot.format_bytes(summary['vms']['min'])}")
        print(f"    Max: {MemorySnapshot.format_bytes(summary['vms']['max'])}")
        print(f"    Avg: {MemorySnapshot.format_bytes(summary['vms']['avg'])}")
        print(f"    Start: {MemorySnapshot.format_bytes(summary['vms']['start'])}")
        print(f"    End: {MemorySnapshot.format_bytes(summary['vms']['end'])}")
        print(f"    Diff: {MemorySnapshot.format_bytes(summary['vms']['diff'])}")
        
        print(f"  Memory Usage (Percent):")
        print(f"    Min: {summary['percent']['min']:.2f}%")
        print(f"    Max: {summary['percent']['max']:.2f}%")
        print(f"    Avg: {summary['percent']['avg']:.2f}%")
        print(f"    Start: {summary['percent']['start']:.2f}%")
        print(f"    End: {summary['percent']['end']:.2f}%")
        print(f"    Diff: {summary['percent']['diff']:.2f}%")

@exception_handler(level=ErrorLevel.WARNING, show_dialog=False)
def memory_usage_decorator(func: Callable) -> Callable:
    """
    Decorator to track memory usage during function execution.
    
    Args:
        func (Callable): The function to track
        
    Returns:
        Callable: The decorated function
    """
    def wrapper(*args, **kwargs):
        # Create a memory tracker
        tracker = MemoryTracker()
        
        # Take a snapshot before execution
        tracker.take_snapshot()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Take a snapshot after execution
        tracker.take_snapshot()
        
        # Print the summary
        tracker.print_summary()
        
        return result
    
    return wrapper

# Global memory tracker instance
_memory_tracker = MemoryTracker()

def get_memory_tracker() -> MemoryTracker:
    """
    Get the global memory tracker instance.
    
    Returns:
        MemoryTracker: The global memory tracker instance
    """
    return _memory_tracker

def take_memory_snapshot() -> MemorySnapshot:
    """
    Take a memory snapshot using the global tracker.
    
    Returns:
        MemorySnapshot: A snapshot of the current memory usage
    """
    return _memory_tracker.take_snapshot()

def start_memory_tracking(interval: float = 1.0) -> None:
    """
    Start memory tracking using the global tracker.
    
    Args:
        interval (float, optional): Interval between snapshots in seconds. 
                                   Defaults to 1.0.
    """
    _memory_tracker.start_tracking(interval)

def stop_memory_tracking() -> None:
    """Stop memory tracking using the global tracker."""
    _memory_tracker.stop_tracking()

def get_memory_summary() -> Dict[str, Any]:
    """
    Get a memory usage summary using the global tracker.
    
    Returns:
        dict: A summary of memory usage statistics
    """
    return _memory_tracker.get_summary()

def print_memory_summary() -> None:
    """Print a memory usage summary using the global tracker."""
    _memory_tracker.print_summary()

def clear_memory_snapshots() -> None:
    """Clear all memory snapshots in the global tracker."""
    _memory_tracker.clear_snapshots()
