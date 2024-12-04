import logging
from datetime import datetime
from typing import Dict, Optional

from .confidence import ConfidenceCalculator
from utils.sync_utils import LatencyTracker

logger = logging.getLogger(__name__)

class TranscriptionFuser:
    """
    Fuses transcriptions from Whisper and VSR models based on confidence scores.
    Primary logic: Use Whisper when confidence > 0.4, otherwise try VSR.
    """
    
    def __init__(self):
        self.confidence_calc = ConfidenceCalculator()
        self.latency_tracker = LatencyTracker()
        
        # Configure thresholds
        self.whisper_threshold = 0.4  # Primary decision threshold
        self.vsr_threshold = 0.1      # Secondary threshold for VSR fallback
        
    def process_transcriptions(
        self,
        whisper_data: Dict,
        vsr_data: Dict,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Process and fuse transcriptions from both models
        
        Args:
            whisper_data: Whisper transcription data
            vsr_data: VSR transcription data
            timestamp: Optional timestamp for synchronization
            
        Returns:
            Dict containing:
                - text: Final transcription text
                - source: Source of transcription ('whisper', 'vsr', or 'none')
                - confidence: Confidence score of chosen transcription
        """
        try:
            # Start timing
            start_time = self.latency_tracker.start_operation('fusion')
            
            # Handle None inputs
            if whisper_data is None or vsr_data is None:
                return {
                    'text': '[error]',
                    'source': 'none',
                    'confidence': 0.0
                }
            
            # Calculate confidence scores
            whisper_conf = self.confidence_calc.get_whisper_confidence(whisper_data)
            vsr_conf = self.confidence_calc.get_vsr_confidence(
                vsr_data.get('text', ''),
                None  # No previous text needed for basic fusion
            )
            
            # Make fusion decision
            result = self._make_decision(
                whisper_data.get('text', ''),
                vsr_data.get('text', ''),
                whisper_conf,
                vsr_conf
            )
            
            # Track latency
            self.latency_tracker.end_operation('fusion', start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transcription fusion: {str(e)}")
            return {
                'text': '[error]',
                'source': 'none',
                'confidence': 0.0
            }
    
    def _make_decision(
        self,
        whisper_text: str,
        vsr_text: str,
        whisper_conf: float,
        vsr_conf: float
    ) -> Dict:
        """
        Make decision on which transcription to use based on confidence scores
        
        Primary logic:
        1. If Whisper confidence > 0.4 -> Use Whisper
        2. If VSR confidence > 0.6 -> Use VSR
        3. Otherwise mark as inaudible
        """
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
