import numpy as np
import logging

logger = logging.getLogger(__name__)

class TimestampedConfidenceCalculator:
    """Calculates confidence scores for timestamped segments"""
    
    def __init__(self):
        # Configure thresholds
        self.min_word_length = 3  # Minimum words for full length confidence
        self.overlap_threshold = 0.5  # Minimum overlap for temporal consistency
        
        # State tracking
        self.last_whisper_segment = None
        self.last_vsr_segment = None
    
    def get_whisper_confidence(self, segment):
        """
        Calculate confidence score for Whisper segment
        
        Args:
            segment: Dict containing 'text' and 'timestamp'
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            if not segment or 'text' not in segment:
                return 0.0
                
            text = segment['text'].strip()
            if not text:
                return 0.0
                
            # Length-based confidence
            words = text.split()
            length_conf = min(len(words) / self.min_word_length, 1.0)
            
            # Temporal consistency with previous segment
            consistency_conf = 0.0
            if self.last_whisper_segment:
                consistency_conf = self._calculate_temporal_consistency(
                    segment, self.last_whisper_segment
                )
            
            # Update state
            self.last_whisper_segment = segment
            
            # Combine scores
            confidence = 0.7 * length_conf + 0.3 * consistency_conf
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating Whisper confidence: {str(e)}")
            return 0.0
    
    def get_vsr_confidence(self, segment):
        """
        Calculate confidence score for VSR segment
        
        Args:
            segment: Dict containing 'text' and 'timestamp'
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            if not segment or 'text' not in segment:
                return 0.0
                
            text = segment['text'].strip()
            if not text:
                return 0.0
                
            # Length-based confidence
            words = text.split()
            length_conf = min(len(words) / self.min_word_length, 1.0)
            
            # Temporal consistency with previous segment
            consistency_conf = 0.0
            if self.last_vsr_segment:
                consistency_conf = self._calculate_temporal_consistency(
                    segment, self.last_vsr_segment
                )
            
            # Update state
            self.last_vsr_segment = segment
            
            # Combine scores
            # Weight length more heavily for VSR as temporal consistency is less reliable
            confidence = 0.8 * length_conf + 0.2 * consistency_conf
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating VSR confidence: {str(e)}")
            return 0.0
    
    def _calculate_temporal_consistency(self, current_segment, prev_segment):
        """Calculate temporal consistency between segments"""
        try:
            # Get timestamps
            curr_start, curr_end = current_segment['timestamp']
            prev_start, prev_end = prev_segment['timestamp']
            
            # Check for temporal overlap
            overlap_start = max(curr_start, prev_start)
            overlap_end = min(curr_end, prev_end)
            
            if overlap_end > overlap_start:
                # Calculate overlap ratio
                overlap_duration = overlap_end - overlap_start
                total_duration = max(curr_end, prev_end) - min(curr_start, prev_start)
                overlap_ratio = overlap_duration / total_duration
                
                # Calculate text similarity for overlapping period
                prev_words = set(prev_segment['text'].lower().split())
                curr_words = set(current_segment['text'].lower().split())
                
                if prev_words and curr_words:
                    word_overlap = len(prev_words.intersection(curr_words))
                    word_ratio = word_overlap / max(len(prev_words), len(curr_words))
                    
                    # Combine temporal and text consistency
                    return 0.5 * overlap_ratio + 0.5 * word_ratio
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating temporal consistency: {str(e)}")
            return 0.0
    
    def reset(self):
        """Reset state tracking"""
        self.last_whisper_segment = None
        self.last_vsr_segment = None
