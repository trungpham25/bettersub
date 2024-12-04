import logging
from datetime import datetime
from typing import Dict, List, Optional
from .confidence import ConfidenceCalculator
from utils.sync_utils import LatencyTracker

logger = logging.getLogger(__name__)

class TimestampedFuser:
    """
    Fuses timestamped transcriptions from Whisper and VSR models.
    Aligns segments by timestamp and makes fusion decisions per segment.
    """
    
    def __init__(self):
        self.confidence_calc = ConfidenceCalculator()
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
        # Calculate confidence scores
        whisper_conf = self.confidence_calc.get_whisper_confidence(whisper_seg)
        vsr_conf = self.confidence_calc.get_vsr_confidence(
            vsr_seg.get('text', ''),
            None  # No previous text needed for basic fusion
        )
        
        # Make fusion decision
        result = self._make_decision(
            whisper_seg.get('text', ''),
            vsr_seg.get('text', ''),
            whisper_conf,
            vsr_conf
        )
        
        # Use average of timestamps for fused segment
        w_start, w_end = whisper_seg['timestamp']
        v_start, v_end = vsr_seg['timestamp']
        
        start_time = min(w_start, v_start)
        end_time = max(w_end, v_end)
        
        return {
            'text': result['text'],
            'source': result['source'],
            'confidence': result['confidence'],
            'timestamp': [start_time, end_time]
        }
    
    def _process_single_segment(self, whisper_seg, vsr_seg):
        """Process a single segment from either source"""
        if whisper_seg:
            conf = self.confidence_calc.get_whisper_confidence(whisper_seg)
            if conf > self.whisper_threshold:
                return {
                    'text': whisper_seg['text'],
                    'source': 'whisper',
                    'confidence': conf,
                    'timestamp': whisper_seg['timestamp']
                }
        
        if vsr_seg:
            conf = self.confidence_calc.get_vsr_confidence(vsr_seg.get('text', ''), None)
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
    
    def _make_decision(
        self,
        whisper_text: str,
        vsr_text: str,
        whisper_conf: float,
        vsr_conf: float
    ) -> Dict:
        """Make decision on which transcription to use based on confidence scores"""
        # Clean up input texts
        whisper_text = whisper_text.strip()
        vsr_text = vsr_text.strip()
        
        # Case 1: Use Whisper if confidence exceeds threshold
        if whisper_conf > self.whisper_threshold and whisper_text:
            return {
                'text': whisper_text,
                'source': 'whisper',
                'confidence': whisper_conf
            }
        
        # Case 2: Try VSR if available and confident
        if vsr_conf > self.vsr_threshold and vsr_text:
            return {
                'text': vsr_text,
                'source': 'vsr',
                'confidence': vsr_conf
            }
        
        # Case 3: If both texts are empty, mark as inaudible
        if not whisper_text and not vsr_text:
            return {
                'text': '[inaudible]',
                'source': 'none',
                'confidence': 0.0
            }
            
        # Case 4: Use any available text with its confidence
        if whisper_text:
            return {
                'text': whisper_text,
                'source': 'whisper',
                'confidence': whisper_conf
            }
        if vsr_text:
            return {
                'text': vsr_text,
                'source': 'vsr',
                'confidence': vsr_conf
            }
            
        # Fallback case
        return {
            'text': '[inaudible]',
            'source': 'none',
            'confidence': 0.0
        }
    
    def get_latency_stats(self) -> Dict:
        """Get fusion operation latency statistics"""
        return self.latency_tracker.get_stats('fusion')
    
    def reset_stats(self):
        """Reset all statistics"""
        self.latency_tracker.reset_stats()
