#!/usr/bin/env python3
"""
Cache management utilities for the Video Annotator.

This module provides tools for caching data in memory to improve performance.
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, Callable, Tuple, List, TypeVar, Generic, Union
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel

# Set up logger
logger = setup_logger('cache_manager')

# Type variables for generic caching
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type

class CacheItem(Generic[V]):
    """
    An item in the cache.
    
    Attributes:
        value: The cached value
        timestamp (float): When the item was added to the cache
        access_count (int): How many times the item has been accessed
        last_access (float): When the item was last accessed
        size (int): Size of the item in bytes (if known)
    """
    
    def __init__(self, value: V, size: int = 0):
        """
        Initialize a CacheItem.
        
        Args:
            value: The value to cache
            size (int, optional): Size of the item in bytes. Defaults to 0.
        """
        self.value = value
        self.timestamp = time.time()
        self.access_count = 0
        self.last_access = self.timestamp
        self.size = size
    
    def access(self) -> None:
        """Record an access to this cache item."""
        self.access_count += 1
        self.last_access = time.time()

class LRUCache(Generic[K, V]):
    """
    A Least Recently Used (LRU) cache.
    
    Attributes:
        capacity (int): Maximum number of items in the cache
        max_size_bytes (int): Maximum size of the cache in bytes
        items (OrderedDict): The cached items
        current_size_bytes (int): Current size of the cache in bytes
        hits (int): Number of cache hits
        misses (int): Number of cache misses
        lock (threading.RLock): Lock for thread safety
    """
    
    def __init__(self, capacity: int = 100, max_size_bytes: int = 0):
        """
        Initialize an LRUCache.
        
        Args:
            capacity (int, optional): Maximum number of items in the cache. Defaults to 100.
            max_size_bytes (int, optional): Maximum size of the cache in bytes. 
                                           Defaults to 0 (no limit).
        """
        self.capacity = capacity
        self.max_size_bytes = max_size_bytes
        self.items: OrderedDict[K, CacheItem[V]] = OrderedDict()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: K) -> Optional[V]:
        """
        Get an item from the cache.
        
        Args:
            key: The key to look up
            
        Returns:
            The cached value, or None if not found
        """
        with self.lock:
            if key in self.items:
                # Move the item to the end (most recently used)
                item = self.items.pop(key)
                self.items[key] = item
                
                # Record the access
                item.access()
                self.hits += 1
                
                return item.value
            else:
                self.misses += 1
                return None
    
    def put(self, key: K, value: V, size: int = 0) -> None:
        """
        Add an item to the cache.
        
        Args:
            key: The key for the item
            value: The value to cache
            size (int, optional): Size of the item in bytes. Defaults to 0.
        """
        with self.lock:
            # If the key already exists, remove it first
            if key in self.items:
                old_item = self.items.pop(key)
                self.current_size_bytes -= old_item.size
            
            # Check if we need to make room
            while (self.capacity > 0 and len(self.items) >= self.capacity) or \
                  (self.max_size_bytes > 0 and self.current_size_bytes + size > self.max_size_bytes):
                # Remove the least recently used item
                if not self.items:
                    break
                _, oldest_item = self.items.popitem(last=False)
                self.current_size_bytes -= oldest_item.size
            
            # Add the new item
            item = CacheItem(value, size)
            self.items[key] = item
            self.current_size_bytes += size
    
    def remove(self, key: K) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: The key to remove
        """
        with self.lock:
            if key in self.items:
                item = self.items.pop(key)
                self.current_size_bytes -= item.size
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.items.clear()
            self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            return {
                'capacity': self.capacity,
                'max_size_bytes': self.max_size_bytes,
                'item_count': len(self.items),
                'current_size_bytes': self.current_size_bytes,
                'hits': self.hits,
                'misses': self.misses,
                'hit_ratio': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            }
    
    def get_keys(self) -> List[K]:
        """
        Get all keys in the cache.
        
        Returns:
            list: List of keys
        """
        with self.lock:
            return list(self.items.keys())
    
    def contains(self, key: K) -> bool:
        """
        Check if a key is in the cache.
        
        Args:
            key: The key to check
            
        Returns:
            bool: True if the key is in the cache, False otherwise
        """
        with self.lock:
            return key in self.items
    
    def get_size(self) -> int:
        """
        Get the number of items in the cache.
        
        Returns:
            int: Number of items in the cache
        """
        with self.lock:
            return len(self.items)
    
    def get_size_bytes(self) -> int:
        """
        Get the size of the cache in bytes.
        
        Returns:
            int: Size of the cache in bytes
        """
        with self.lock:
            return self.current_size_bytes

class FrameCache(LRUCache[int, Any]):
    """
    A cache specifically for video frames.
    
    This extends LRUCache with methods specific to caching video frames.
    """
    
    def __init__(self, capacity: int = 100, max_size_bytes: int = 1024 * 1024 * 1024):
        """
        Initialize a FrameCache.
        
        Args:
            capacity (int, optional): Maximum number of frames in the cache. Defaults to 100.
            max_size_bytes (int, optional): Maximum size of the cache in bytes. 
                                           Defaults to 1 GB.
        """
        super().__init__(capacity, max_size_bytes)
        self.logger = setup_logger('frame_cache')
        self.logger.info(f"Initialized frame cache with capacity {capacity} frames, "
                        f"max size {max_size_bytes / (1024 * 1024):.2f} MB")
    
    def get_frame(self, frame_number: int) -> Optional[Any]:
        """
        Get a frame from the cache.
        
        Args:
            frame_number (int): The frame number to retrieve
            
        Returns:
            The cached frame, or None if not found
        """
        return self.get(frame_number)
    
    def put_frame(self, frame_number: int, frame: Any, size: int = 0) -> None:
        """
        Add a frame to the cache.
        
        Args:
            frame_number (int): The frame number
            frame: The frame data
            size (int, optional): Size of the frame in bytes. Defaults to 0.
        """
        # Estimate size if not provided
        if size == 0 and hasattr(frame, 'nbytes'):
            size = frame.nbytes
        
        self.put(frame_number, frame, size)
    
    def prefetch_frames(self, frame_numbers: List[int], loader: Callable[[int], Any]) -> None:
        """
        Prefetch frames into the cache.
        
        Args:
            frame_numbers (List[int]): List of frame numbers to prefetch
            loader (Callable[[int], Any]): Function to load a frame
        """
        for frame_number in frame_numbers:
            if not self.contains(frame_number):
                try:
                    frame = loader(frame_number)
                    size = frame.nbytes if hasattr(frame, 'nbytes') else 0
                    self.put_frame(frame_number, frame, size)
                except Exception as e:
                    self.logger.warning(f"Failed to prefetch frame {frame_number}: {str(e)}")
    
    def prefetch_range(self, start_frame: int, end_frame: int, loader: Callable[[int], Any]) -> None:
        """
        Prefetch a range of frames into the cache.
        
        Args:
            start_frame (int): First frame to prefetch
            end_frame (int): Last frame to prefetch
            loader (Callable[[int], Any]): Function to load a frame
        """
        frame_numbers = list(range(start_frame, end_frame + 1))
        self.prefetch_frames(frame_numbers, loader)
    
    def get_frame_with_loader(self, frame_number: int, loader: Callable[[int], Any]) -> Any:
        """
        Get a frame from the cache, loading it if not present.
        
        Args:
            frame_number (int): The frame number to retrieve
            loader (Callable[[int], Any]): Function to load the frame if not cached
            
        Returns:
            The frame data
        """
        frame = self.get_frame(frame_number)
        if frame is None:
            frame = loader(frame_number)
            size = frame.nbytes if hasattr(frame, 'nbytes') else 0
            self.put_frame(frame_number, frame, size)
        return frame

# Global frame cache instance
_frame_cache = FrameCache()

def get_frame_cache() -> FrameCache:
    """
    Get the global frame cache instance.
    
    Returns:
        FrameCache: The global frame cache instance
    """
    return _frame_cache

def get_cached_frame(frame_number: int) -> Optional[Any]:
    """
    Get a frame from the global cache.
    
    Args:
        frame_number (int): The frame number to retrieve
        
    Returns:
        The cached frame, or None if not found
    """
    return _frame_cache.get_frame(frame_number)

def cache_frame(frame_number: int, frame: Any, size: int = 0) -> None:
    """
    Add a frame to the global cache.
    
    Args:
        frame_number (int): The frame number
        frame: The frame data
        size (int, optional): Size of the frame in bytes. Defaults to 0.
    """
    _frame_cache.put_frame(frame_number, frame, size)

def configure_frame_cache(capacity: int = 100, max_size_bytes: int = 1024 * 1024 * 1024) -> None:
    """
    Configure the global frame cache.
    
    Args:
        capacity (int, optional): Maximum number of frames in the cache. Defaults to 100.
        max_size_bytes (int, optional): Maximum size of the cache in bytes. 
                                       Defaults to 1 GB.
    """
    global _frame_cache
    _frame_cache = FrameCache(capacity, max_size_bytes)

def clear_frame_cache() -> None:
    """Clear the global frame cache."""
    _frame_cache.clear()

def get_frame_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the global frame cache.
    
    Returns:
        dict: Cache statistics
    """
    return _frame_cache.get_stats()
