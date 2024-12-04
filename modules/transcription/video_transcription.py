import os
import logging
import subprocess
import numpy as np
import torch
from datetime import datetime
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer,
    pipeline
)

logger = logging.getLogger(__name__)

class VideoTranscriptionEngine:
    def __init__(self):
        """Initialize the video transcription engine"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_name = "openai/whisper-large-v3-turbo"
        self.sample_rate = 16000
        self.temp_dir = "temp_audio"
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def initialize(self):
        """Initialize the Whisper model"""
        try:
            logger.info("Initializing Whisper model...")
            
            # Initialize model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)

            # Initialize processor and tokenizer
            processor = AutoProcessor.from_pretrained(self.model_name)
            tokenizer = WhisperTokenizer.from_pretrained(self.model_name, language="en")

            # Create pipeline
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,  # Process 30-second chunks
                stride_length_s=5,   # 5-second overlap between chunks
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            logger.info("Video transcription engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize video transcription engine: {str(e)}")
            return False

    def extract_audio(self, video_path):
        """Extract audio from video file"""
        try:
            # Generate unique filename for audio
            audio_filename = os.path.join(
                self.temp_dir,
                f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            # Use FFmpeg to extract audio
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                audio_filename
            ]
            
            subprocess.run(command, check=True)
            logger.info(f"Audio extracted to {audio_filename}")
            
            return audio_filename
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            return None

    def transcribe_video(self, video_path):
        """Transcribe video file"""
        try:
            # Extract audio from video
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                raise Exception("Failed to extract audio from video")

            try:
                # Transcribe audio
                logger.info("Starting transcription...")
                result = self.pipe(
                    audio_path,
                    return_timestamps=True,  # Get word-level timestamps
                    generate_kwargs={"language": "en"}
                )

                # Process results
                transcription = []
                for chunk in result["chunks"]:
                    transcription.append({
                        'timestamp': [chunk['timestamp'][0], chunk['timestamp'][1]],
                        'text': chunk['text'].strip()
                    })

                return {
                    'status': 'success',
                    'segments': transcription,
                    'video_path': video_path
                }

            finally:
                # Clean up temporary audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'video_path': video_path
            }

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up video transcription engine...")
        try:
            # Clear pipeline
            self.pipe = None
            
            # Remove temp directory if empty
            if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                os.rmdir(self.temp_dir)
            
            logger.info("Video transcription engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
