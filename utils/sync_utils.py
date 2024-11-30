import logging
import time
from datetime import datetime, timedelta
from queue import Queue
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class TimestampedQueue:
    """Queue that maintains items with timestamps"""
    def __init__(self, maxsize=0, max_age=timedelta(seconds=5)):
        self.queue = Queue(maxsize)
        self.max_age = max_age
        self.buffer = deque()

    def put(self, item, timestamp=None):
        """Put an item in the queue with a timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        self.queue.put((timestamp, item))

    def get(self):
        """Get an item and its timestamp from the queue"""
        if not self.queue.empty():
            return self.queue.get()
        return None

    def get_latest(self):
        """Get the most recent item from the queue"""
        latest_item = None
        latest_timestamp = None
        
        while not self.queue.empty():
            timestamp, item = self.queue.get()
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_item = item
        
        return latest_timestamp, latest_item if latest_timestamp else None

    def clear_old(self, current_time=None):
        """Remove items older than max_age"""
        if current_time is None:
            current_time = datetime.now()
            
        while not self.queue.empty():
            timestamp, item = self.queue.get()
            if current_time - timestamp <= self.max_age:
                self.queue.put((timestamp, item))

class StreamSynchronizer:
    """Synchronizes multiple data streams based on timestamps"""
    def __init__(self, sync_window=timedelta(milliseconds=500)):
        self.sync_window = sync_window
        self.streams = {}
        self.buffer_size = 100

    def register_stream(self, stream_name):
        """Register a new data stream"""
        if stream_name not in self.streams:
            self.streams[stream_name] = TimestampedQueue(
                maxsize=self.buffer_size
            )
            logger.info(f"Registered stream: {stream_name}")

    def add_data(self, stream_name, data, timestamp=None):
        """Add data to a stream"""
        if stream_name in self.streams:
            self.streams[stream_name].put(data, timestamp)
        else:
            logger.warning(f"Stream not registered: {stream_name}")

    def get_synchronized_data(self, current_time=None):
        """Get synchronized data from all streams within the sync window"""
        if current_time is None:
            current_time = datetime.now()
            
        # Clear old data from all streams
        for stream in self.streams.values():
            stream.clear_old(current_time)
        
        # Collect data from all streams within sync window
        sync_data = {}
        for name, stream in self.streams.items():
            timestamp, data = stream.get_latest()
            if timestamp and current_time - timestamp <= self.sync_window:
                sync_data[name] = {
                    'timestamp': timestamp,
                    'data': data
                }
        
        return sync_data if sync_data else None

class TimeAligner:
    """Aligns timestamps between different data streams"""
    def __init__(self):
        self.reference_time = None
        self.time_offsets = {}

    def set_reference_stream(self, stream_name, timestamp):
        """Set the reference stream for time alignment"""
        self.reference_time = timestamp
        self.time_offsets[stream_name] = timedelta(0)
        logger.info(f"Set reference stream: {stream_name}")

    def register_stream(self, stream_name, timestamp):
        """Register a stream and calculate its offset from reference"""
        if self.reference_time is None:
            raise ValueError("Reference stream not set")
            
        offset = timestamp - self.reference_time
        self.time_offsets[stream_name] = offset
        logger.info(f"Registered stream: {stream_name} with offset: {offset}")

    def align_timestamp(self, stream_name, timestamp):
        """Align a timestamp to the reference time"""
        if stream_name not in self.time_offsets:
            raise ValueError(f"Stream not registered: {stream_name}")
            
        return timestamp - self.time_offsets[stream_name]

class LatencyTracker:
    """Tracks processing latency for different components"""
    def __init__(self, window_size=100):
        self.latencies = {}
        self.window_size = window_size

    def start_operation(self, operation_name):
        """Start timing an operation"""
        if operation_name not in self.latencies:
            self.latencies[operation_name] = deque(maxlen=self.window_size)
        return time.time()

    def end_operation(self, operation_name, start_time):
        """End timing an operation and record latency"""
        latency = time.time() - start_time
        self.latencies[operation_name].append(latency)
        return latency

    def get_stats(self, operation_name):
        """Get latency statistics for an operation"""
        if operation_name not in self.latencies:
            return None
            
        latencies = np.array(self.latencies[operation_name])
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }

    def reset_stats(self, operation_name=None):
        """Reset latency statistics"""
        if operation_name:
            if operation_name in self.latencies:
                self.latencies[operation_name].clear()
        else:
            for queue in self.latencies.values():
                queue.clear()

class DataBuffer:
    """Circular buffer for storing timestamped data"""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, data, timestamp=None):
        """Add data to buffer"""
        if timestamp is None:
            timestamp = datetime.now()
        self.buffer.append((timestamp, data))

    def get_range(self, start_time, end_time):
        """Get data within a time range"""
        return [
            (ts, data) for ts, data in self.buffer
            if start_time <= ts <= end_time
        ]

    def get_latest(self, n=1):
        """Get the n most recent items"""
        return list(self.buffer)[-n:]

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
