import numpy as np
import logging

logger = logging.getLogger(__name__)

class ConfidenceCalculator:
    """Calculates confidence scores for Whisper and VSR transcriptions"""
    
    @staticmethod
    def get_whisper_confidence(prediction):
        """
        Calculate confidence score for Whisper transcription
        
        Args:
            prediction: Whisper prediction object containing logprobs
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            # Extract logprobs if available
            logprobs = prediction.get('logprobs', [])
            if isinstance(logprobs, (list, np.ndarray)) and len(logprobs) > 0:
                # Convert to numpy array if needed
                if isinstance(logprobs, list):
                    logprobs = np.array(logprobs)
                # Convert log probabilities to probabilities and average
                probs = np.exp(logprobs)
                return float(np.mean(probs))
            
            # Fallback to token-level probabilities if available
            token_probs = prediction.get('token_probs', [])
            if isinstance(token_probs, (list, np.ndarray)) and len(token_probs) > 0:
                if isinstance(token_probs, list):
                    token_probs = np.array(token_probs)
                return float(np.mean(token_probs))
                
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Whisper confidence: {str(e)}")
            return 0.0
    
    @staticmethod
    def get_vsr_confidence(transcript, prev_transcript=None):
        """
        Calculate confidence score for VSR transcription
        
        Args:
            transcript: Current VSR transcript
            prev_transcript: Previous VSR transcript for consistency check
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            if not transcript:
                return 0.0
                
            # Length-based confidence (longer transcripts are generally more reliable)
            words = transcript.split()
            length_conf = min(len(words) / 3, 1.0)  # Cap at 3 words
            
            # Consistency with previous transcript
            consistency_conf = 0.0
            if prev_transcript:
                prev_words = set(prev_transcript.lower().split())
                current_words = set(transcript.lower().split())
                if prev_words and current_words:
                    overlap = len(prev_words.intersection(current_words))
                    consistency_conf = overlap / max(len(prev_words), len(current_words))
            
            # Combine confidence scores
            # Weight length more heavily as it's a more reliable indicator
            confidence = 0.7 * length_conf + 0.3 * consistency_conf
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating VSR confidence: {str(e)}")
            return 0.0
