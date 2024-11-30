import torch
import logging
import numpy as np
import sounddevice as sd
from queue import Queue
import threading
import scipy.io.wavfile
import uuid
import os
from datetime import datetime
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer,
    pipeline
)

class TranscriptionEngine:
    def __init__(self):
        """Initialize the transcription engine"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.audio_queue = Queue()
        self.is_running = False
        self.sample_rate = 16000
        self.chunk_duration = 2  # seconds (matches stream_every in original)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_name = "openai/whisper-large-v3-turbo"
        self.previous_transcription = ""

    def initialize(self):
        """Initialize the Whisper model and audio capture"""
        try:
            self.logger.info("Initializing Whisper model...")
            
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
                max_new_tokens=25,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

            # Initialize audio capture thread
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.is_running = True
            self.audio_thread.start()
            
            self.logger.info("Transcription engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize transcription engine: {str(e)}")
            return False

    def _audio_capture_loop(self):
        """Continuous audio capture loop"""
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream"""
            if status:
                self.logger.warning(f"Audio capture status: {status}")
            
            # Add new audio data to queue
            self.audio_queue.put((self.sample_rate, indata.copy()))

        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * self.chunk_duration),
                dtype=np.int16
            ):
                self.logger.info("Audio capture started")
                while self.is_running:
                    sd.sleep(100)  # Sleep for 100ms
                    
        except Exception as e:
            self.logger.error(f"Audio capture error: {str(e)}")
            self.is_running = False

    def process_audio(self, audio_data=None):
        """Process audio chunks and return transcription"""
        try:
            # If no audio_data provided, get from queue
            if audio_data is None:
                if self.audio_queue.empty():
                    return None
                audio_data = self.audio_queue.get()

            # Save audio to temporary file
            filename = f"{uuid.uuid4().hex}.wav"
            try:
                sample_rate, audio = audio_data
                scipy.io.wavfile.write(filename, sample_rate, audio)

                # Transcribe audio
                result = self.pipe(filename)
                transcription = result["text"]
                
                # Update and return full transcription
                self.previous_transcription += transcription

                return {
                    'timestamp': datetime.now().isoformat(),
                    'text': self.previous_transcription,
                    'segment': transcription
                }

            finally:
                # Clean up temporary file
                if os.path.exists(filename):
                    os.remove(filename)

        except Exception as e:
            self.logger.error(f"Audio processing error: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up transcription engine...")
        try:
            self.is_running = False
            if hasattr(self, 'audio_thread'):
                self.audio_thread.join(timeout=1.0)
            
            # Clear queue
            while not self.audio_queue.empty():
                self.audio_queue.get()
            
            # Clear previous transcription
            self.previous_transcription = ""
            
            # Clear pipeline
            self.pipe = None
            
            self.logger.info("Transcription engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def reset_transcription(self):
        """Reset the transcription history"""
        self.previous_transcription = ""
