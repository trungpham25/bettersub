# Codebase Summary

## System Architecture Overview

### 1. Frontend Layer (`main.py`, `main_av.py`)
- Entry points for application modes
- Gradio interface implementation
- Mode switching logic
- File upload handling
- Results display and export

### 2. Backend Processing Pipeline

#### Audio Processing (`modules/transcription/`)
- `realtime_transcription.py`: Real-time Whisper integration
- Confidence scoring implementation
- Timestamp generation
- Audio preprocessing utilities

#### Visual Processing (`auto_avsr/`)
- `realtime_vsr_v12.py`: Current VSR model implementation
- Face detection and tracking
- Lip movement analysis
- Visual confidence metrics

#### Utility Layer (`utils/`)
- `audio_utils.py`: Audio processing functions
- `video_utils.py`: Video frame extraction and processing
- `av_utils.py`: Audio-visual synchronization
- `sync_utils.py`: Timestamp management

#### Testing Framework (`tests/`)
- `test_transcription.py`: Transcription validation
- Integration test suites
- Performance benchmarks

## Component Interactions

### 1. Data Flow
```
User Input (Video/Realtime)
    ↓
Preprocessing (utils/)
    ↓
Parallel Processing
    ├→ Whisper (transcription/)
    └→ VSR (auto_avsr/)
    ↓
Fusion Engine (av_utils.py)
    ↓
Output Generation
```

### 2. Fusion Algorithm Implementation
- Located in `av_utils.py`
- Confidence threshold management
- Source selection logic
- Timestamp synchronization
- Output composition

### 3. Key Integration Points
1. Input Processing
   - Video file handling
   - Real-time stream management
   - Format validation

2. Model Integration
   - Whisper initialization and inference
   - VSR model setup and processing
   - Resource management

3. Output Generation
   - SRT/VTT formatting
   - Source marking
   - Timestamp alignment

## Current Development Status

### Active Components
- Basic Whisper integration
- VSR model implementation
- Utility functions
- Testing framework

### Under Development
- Fusion algorithm implementation
- UI enhancements
- Export functionality
- LLM integration (planned)

## Development Guidelines

### 1. Code Organization
- Maintain modular architecture
- Clear component boundaries
- Consistent error handling
- Comprehensive logging

### 2. Testing Requirements
- Unit tests for core functions
- Integration tests for pipelines
- Performance benchmarks
- Error case validation

### 3. Documentation Standards
- Inline documentation
- API specifications
- Component interaction diagrams
- Setup and deployment guides

## External Dependencies
See `requirements.txt` for complete list
- Core Dependencies:
  - PyTorch: Model operations
  - FFmpeg: Media processing
  - Gradio: UI framework
  - Face alignment libraries
