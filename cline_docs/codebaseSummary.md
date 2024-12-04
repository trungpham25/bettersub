# Codebase Summary

## Project Structure

### Core Components
1. Main Applications
   - `main.py`: Primary application entry point
   - `main_av.py`: Audio-visual processing entry point

2. Modules
   - `modules/transcription/`
     - `realtime_transcription.py`: Real-time transcription functionality
     - `video_transcription.py`: Video file transcription with timestamps

3. Auto-AVSR
   - `auto_avsr/`: Visual Speech Recognition implementation
     - `realtime_vsr_v12.py`: Current VSR model implementation
     - Multiple supporting modules for face detection and processing

4. Utilities
   - `utils/`
     - `audio_utils.py`: Audio processing functions
     - `video_utils.py`: Video processing functions
     - `av_utils.py`: Audio-visual synchronization utilities
     - `sync_utils.py`: Timing and synchronization functions

5. Testing
   - `tests/`
     - `test_transcription.py`: Transcription module tests
     - `test_video_transcription.py`: Video transcription tests

## Key Components and Their Interactions

### Transcription Pipeline
1. Audio Processing
   - Whisper model integration
   - Real-time audio capture
   - Video audio extraction and processing
   - Timestamp preservation

2. Visual Processing
   - VSR model implementation
   - Face and lip movement detection
   - Frame extraction and processing

3. Synchronization (Planned)
   - Audio-visual alignment
   - Timestamp management
   - Output merging logic

## Data Flow
1. Input Handling
   - Real-time mode: Camera and microphone input
   - Video mode: File upload and processing

2. Processing Pipeline
   - Audio extraction and processing
   - Visual feature extraction
   - Model inference
   - Result synchronization (planned)

3. Output Generation
   - Transcription formatting
   - Subtitle file generation (planned)
   - UI display

## Recent Changes
1. Video Transcription Implementation
   - Added FFmpeg integration for audio extraction
   - Implemented timestamp preservation
   - Added unit tests for video transcription
   - Successfully completed Ticket 1

2. Development Status
   - Ticket 1 (Video Input for Whisper) completed
   - Ticket 2 (VSR Timestamp Sync) on hold
   - Moving to Ticket 3 (Fusion Logic)

## Development Guidelines
1. Code Organization
   - Maintain modular structure
   - Clear separation of concerns
   - Consistent naming conventions

2. Testing
   - Unit tests for core functionality
   - Integration tests for pipelines
   - Regular test updates with new features

3. Documentation
   - Keep documentation in sync with code
   - Document all major components
   - Maintain clear API documentation

## External Dependencies
- See `requirements.txt` for detailed package requirements
- Key dependencies:
  - PyTorch for model operations
  - FFmpeg for media processing
  - Gradio for UI
  - Face alignment libraries
