import logging
from datetime import datetime
from typing import Dict, List, Optional
from .confidence_timestamped import TimestampedConfidenceCalculator
from utils.sync_utils import LatencyTracker

logger = logging.getLogger(__name__)

class TimestampedFuser:
    """
    Fuses timestamped transcriptions from Whisper and VSR models.
    Aligns segments by timestamp and makes fusion decisions based on confidence.
    """
    
    def __init__(self):
        self.confidence_calc = TimestampedConfidenceCalculator()
        self.latency_tracker = LatencyTracker()
        
        # Configure thresholds
        self.whisper_threshold = 0.4  # Primary decision threshold
        self.vsr_threshold = 0.1      # Secondary threshold for VSR fallback
        
        # Configure timestamp matching
        self.time_tolerance = 0.5  # Allow 0.5s difference in timestamps
        
    def process_segments(
        self,
        whisper_segments: List[Dict],
        vsr_segments: List[Dict]
    ) -> List[Dict]:
        """
        Process and fuse timestamped segments from both models
        
        Args:
            whisper_segments: List of Whisper segments with timestamps
            vsr_segments: List of VSR segments with timestamps
            
        Returns:
            List of fused segments with timestamps
        """
        try:
            # Start timing
            start_time = self.latency_tracker.start_operation('fusion')
            
            # Reset confidence calculator state
            self.confidence_calc.reset()
            
            # Handle empty inputs
            if not whisper_segments and not vsr_segments:
                return []
                
            # Align and fuse segments
            fused_segments = []
            
            # Track processed segments
            whisper_idx = 0
            vsr_idx = 0
            
            while whisper_idx < len(whisper_segments) or vsr_idx < len(vsr_segments):
                # Get current segments if available
                whisper_seg = whisper_segments[whisper_idx] if whisper_idx < len(whisper_segments) else None
                vsr_seg = vsr_segments[vsr_idx] if vsr_idx < len(vsr_segments) else None
                
                # If one stream is exhausted, process remaining segments from other stream
                if whisper_seg is None:
                    fused_segments.append(self._process_single_segment(None, vsr_seg))
                    vsr_idx += 1
                    continue
                    
                if vsr_seg is None:
                    fused_segments.append(self._process_single_segment(whisper_seg, None))
                    whisper_idx += 1
                    continue
                
                # Get segment timestamps
                w_start, w_end = whisper_seg['timestamp']
                v_start, v_end = vsr_seg['timestamp']
                
                # Check if segments overlap
                if self._segments_overlap(w_start, w_end, v_start, v_end):
                    # Process overlapping segments
                    fused_seg = self._fuse_overlapping_segments(whisper_seg, vsr_seg)
                    fused_segments.append(fused_seg)
                    whisper_idx += 1
                    vsr_idx += 1
                    
                # If segments don't overlap, process the earlier one
                elif w_start < v_start:
                    fused_segments.append(self._process_single_segment(whisper_seg, None))
                    whisper_idx += 1
                else:
                    fused_segments.append(self._process_single_segment(None, vsr_seg))
                    vsr_idx += 1
            
            # Track latency
            self.latency_tracker.end_operation('fusion', start_time)
            
            return fused_segments
            
        except Exception as e:
            logger.error(f"Error in timestamped fusion: {str(e)}")
            return []
    
    def _segments_overlap(self, w_start, w_end, v_start, v_end):
        """Check if two segments overlap within tolerance"""
        # Add tolerance to account for slight timestamp differences
        return (
            (w_start - self.time_tolerance <= v_end) and 
            (w_end + self.time_tolerance >= v_start)
        )
    
    def _fuse_overlapping_segments(self, whisper_seg, vsr_seg):
        """Fuse two overlapping segments"""
        try:
            # Calculate confidence scores
            whisper_conf = self.confidence_calc.get_whisper_confidence(whisper_seg)
            vsr_conf = self.confidence_calc.get_vsr_confidence(vsr_seg)
            
            # Use Whisper if confidence exceeds threshold
            if whisper_conf > self.whisper_threshold:
                return {
                    'text': whisper_seg['text'],
                    'source': 'whisper',
                    'confidence': whisper_conf,
                    'timestamp': whisper_seg['timestamp']
                }
            
            # Try VSR if available and confident
            if vsr_conf > self.vsr_threshold:
                return {
                    'text': vsr_seg['text'],
                    'source': 'vsr',
                    'confidence': vsr_conf,
                    'timestamp': vsr_seg['timestamp']
                }
            
            # If neither is confident enough, use Whisper by default
            return {
                'text': whisper_seg['text'],
                'source': 'whisper',
                'confidence': whisper_conf,
                'timestamp': whisper_seg['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error fusing segments: {str(e)}")
            # Return Whisper segment as fallback
            return {
                'text': whisper_seg['text'],
                'source': 'whisper',
                'confidence': 0.0,
                'timestamp': whisper_seg['timestamp']
            }
    
    def _process_single_segment(self, whisper_seg, vsr_seg):
        """Process a single segment from either source"""
        try:
            if whisper_seg:
                conf = self.confidence_calc.get_whisper_confidence(whisper_seg)
                return {
                    'text': whisper_seg['text'],
                    'source': 'whisper',
                    'confidence': conf,
                    'timestamp': whisper_seg['timestamp']
                }
            
            if vsr_seg:
                conf = self.confidence_calc.get_vsr_confidence(vsr_seg)
                if conf > self.vsr_threshold:
                    return {
                        'text': vsr_seg['text'],
                        'source': 'vsr',
                        'confidence': conf,
                        'timestamp': vsr_seg['timestamp']
                    }
            
            # If neither segment is usable, return inaudible with appropriate timestamp
            seg = whisper_seg or vsr_seg
            return {
                'text': '[inaudible]',
                'source': 'none',
                'confidence': 0.0,
                'timestamp': seg['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error processing single segment: {str(e)}")
            # Return inaudible with segment timestamp
            seg = whisper_seg or vsr_seg
            return {
                'text': '[error]',
                'source': 'none',
                'confidence': 0.0,
                'timestamp': seg['timestamp']
            }
    
    def get_latency_stats(self) -> Dict:
        """Get fusion operation latency statistics"""
        return self.latency_tracker.get_stats('fusion')
    
    def reset_stats(self):
        """Reset all statistics"""
        self.latency_tracker.reset_stats()
        self.confidence_calc.reset()
