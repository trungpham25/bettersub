import numpy as np
import librosa
import logging
from queue import Queue
import threading
import sounddevice as sd
from scipy import signal
from datetime import datetime
from sklearn.mixture import GaussianMixture

class AudioDiarization:
    def __init__(self):
        """Initialize audio diarization system"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.audio_queue = Queue()
        self.is_running = False
        self.sample_rate = 16000
        self.n_mics = 2  # Number of microphones
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.speaker_models = {}
        self.current_speaker_id = 0
        self.min_speech_duration = 0.5  # seconds

    def initialize(self):
        """Initialize audio diarization system"""
        try:
            self.logger.info("Initializing audio diarization system...")
            
            # Initialize speaker models
            self._initialize_speaker_models()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.is_running = True
            self.processing_thread.start()
            
            self.logger.info("Audio diarization system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio diarization: {str(e)}")
            return False

    def _initialize_speaker_models(self):
        """Initialize GMM models for speaker identification"""
        try:
            # Initialize empty GMM models
            self.gmm_models = {}
            self.feature_scaler = None
            self.logger.info("Speaker models initialized")
            
        except Exception as e:
            self.logger.error(f"Speaker model initialization error: {str(e)}")
            raise

    def _processing_loop(self):
        """Main audio processing loop"""
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    self._process_audio_chunk(audio_data)
            except Exception as e:
                self.logger.error(f"Processing loop error: {str(e)}")

    def _process_audio_chunk(self, audio_data):
        """Process a chunk of audio data"""
        try:
            # Extract features
            features = self._extract_features(audio_data)
            
            # Perform speaker diarization
            speaker_id = self._identify_speaker(features)
            
            # Update speaker models if needed
            self._update_speaker_model(speaker_id, features)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'speaker_id': speaker_id,
                'confidence': self._get_speaker_confidence(features, speaker_id)
            }
            
        except Exception as e:
            self.logger.error(f"Audio chunk processing error: {str(e)}")
            return None

    def _extract_features(self, audio_data):
        """Extract audio features for speaker identification"""
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=20,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )

            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )

            # Extract pitch features
            f0, voiced_flag, _ = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            pitch = np.array([f0[voiced_flag].mean() if voiced_flag.any() else 0])

            # Combine features
            features = np.concatenate([
                mfcc.mean(axis=1),
                spectral_centroids.mean(axis=1),
                spectral_rolloff.mean(axis=1),
                pitch
            ])

            return features.reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            return None

    def _identify_speaker(self, features):
        """Identify speaker from features"""
        if features is None:
            return None

        if not self.gmm_models:
            return self._register_new_speaker(features)
        
        # Calculate likelihood for each speaker
        likelihoods = {}
        for speaker_id, gmm in self.gmm_models.items():
            likelihoods[speaker_id] = gmm.score_samples(features)[0]
        
        # Return most likely speaker
        return max(likelihoods.items(), key=lambda x: x[1])[0]

    def _register_new_speaker(self, features):
        """Register a new speaker"""
        speaker_id = self.current_speaker_id
        self.current_speaker_id += 1
        
        # Initialize new GMM model
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm.fit(features)
        self.gmm_models[speaker_id] = gmm
        
        return speaker_id

    def _update_speaker_model(self, speaker_id, features):
        """Update speaker model with new features"""
        if speaker_id in self.gmm_models and features is not None:
            # For simplicity, we're not updating the model in real-time
            # In a production system, you might want to implement online learning
            pass

    def _get_speaker_confidence(self, features, speaker_id):
        """Calculate confidence score for speaker identification"""
        if speaker_id not in self.gmm_models or features is None:
            return 0.0
        
        score = self.gmm_models[speaker_id].score_samples(features)[0]
        # Convert log-likelihood to probability-like score
        return np.exp(score) / (1 + np.exp(score))

    def process_audio(self, audio_data):
        """Process audio data and return speaker information"""
        try:
            self.audio_queue.put(audio_data)
            return self._process_audio_chunk(audio_data)
        except Exception as e:
            self.logger.error(f"Audio processing error: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up audio diarization system...")
        try:
            self.is_running = False
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            
            # Clear queue
            while not self.audio_queue.empty():
                self.audio_queue.get()
            
            # Clear models
            self.gmm_models.clear()
            self.speaker_models.clear()
            
            self.logger.info("Audio diarization cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
