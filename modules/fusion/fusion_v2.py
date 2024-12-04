import logging
from datetime import datetime
from typing import Dict, Optional

from .confidence import ConfidenceCalculator
from utils.sync_utils import LatencyTracker

logger = logging.getLogger(__name__)

class TranscriptionFuser:
    """
    Enhanced fusion of transcriptions from Whisper and VSR models.
    Improved handling of VSR-only scenarios and muted sections.
    """
    
    def __init__(self):
        self.confidence_calc = ConfidenceCalculator()
        self.latency_tracker = LatencyTracker()
        
        # Configure thresholds
        self.whisper_threshold = 0.4  # Primary decision threshold
        self.vsr_threshold = 0.1      # Lower threshold for VSR to handle muted sections
        self.last_vsr_text = None     # Track last successful VSR output
        
    def process_transcriptions(
        self,
        whisper_data: Dict,
        vsr_data: Dict,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Process and fuse transcriptions from both models with enhanced VSR handling
        
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
            if whisper_data is None and vsr_data is None:
                return {
                    'text': '[error]',
                    'source': 'none',
                    'confidence': 0.0
                }
            
            # Special handling for VSR-only mode (muted sections)
            if whisper_data is None or not whisper_data.get('text', '').strip():
                vsr_text = vsr_data.get('text', '') if vsr_data else ''
                if vsr_text.strip():
                    self.last_vsr_text = vsr_text
                    return {
                        'text': vsr_text,
                        'source': 'vsr',
                        'confidence': self.confidence_calc.get_vsr_confidence(vsr_text, None)
                    }
                elif self.last_vsr_text:
                    # Use last successful VSR text if current is empty
                    return {
                        'text': self.last_vsr_text,
                        'source': 'vsr',
                        'confidence': self.vsr_threshold  # Minimum confidence
                    }
            
            # Calculate confidence scores
            whisper_conf = self.confidence_calc.get_whisper_confidence(whisper_data) if whisper_data else 0.0
            vsr_conf = self.confidence_calc.get_vsr_confidence(
                vsr_data.get('text', ''),
                self.last_vsr_text  # Use last VSR text for consistency check
            ) if vsr_data else 0.0
            
            # Make fusion decision
            result = self._make_decision(
                whisper_data.get('text', '') if whisper_data else '',
                vsr_data.get('text', '') if vsr_data else '',
                whisper_conf,
                vsr_conf
            )
            
            # Track latency
            self.latency_tracker.end_operation('fusion', start_time)
            
            # Update last VSR text if current decision uses VSR
            if result['source'] == 'vsr' and result['text'].strip():
                self.last_vsr_text = result['text']
            
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
        Make decision on which transcription to use based on confidence scores.
        Enhanced logic for VSR-only scenarios and muted sections.
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
        
        # Case 2: Use VSR if confidence exceeds threshold
        if vsr_conf > self.vsr_threshold and vsr_text:
            return {
                'text': vsr_text,
                'source': 'vsr',
                'confidence': vsr_conf
            }
        
        # Case 3: If VSR has text but low confidence, still use it in muted sections
        if not whisper_text and vsr_text:
            return {
                'text': vsr_text,
                'source': 'vsr',
                'confidence': max(vsr_conf, self.vsr_threshold)  # Ensure minimum confidence
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
            
        # Case 5: No valid text available
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
        self.last_vsr_text = None  # Also reset last VSR text
