# Current Development Tasks

## Active Tickets

### Ticket 1: Refactor Whisper for Video Input ✓
- [x] Implement video file audio extraction using ffmpeg
- [x] Connect extracted audio to Whisper pipeline
- [x] Ensure timestamp preservation
- [x] Add unit tests for video file transcription
- [x] Test with sample videos

### Ticket 2: Add Timestamp Synchronization in VSR (On Hold)
- [ ] Extract video frames with timestamps
- [ ] Update VSR pipeline for timestamped output
- [ ] Validate output with sample videos
- [ ] Add timestamp validation tests
Note: This ticket is temporarily on hold pending further investigation of the existing VSR implementation.

### Ticket 3: Implement Whisper + VSR Fusion Logic
- [ ] Define confidence thresholds
- [ ] Create Whisper prioritization logic
- [ ] Implement VSR fallback mechanism
- [ ] Add "inaudible" segment marking

### Ticket 4: Integrate Whisper and VSR for Video Transcription
- [ ] Create unified pipeline
- [ ] Implement source indication system
- [ ] Add timestamp-based output merging
- [ ] Test with various video scenarios

### Ticket 5: UI Enhancements
- [ ] Add video upload widget
- [ ] Implement progress tracking
- [ ] Create transcription preview/edit interface

### Ticket 6: Export Functionality
- [ ] Implement SRT export
- [ ] Implement VTT export
- [ ] Add download functionality
- [ ] Validate with video players

### Ticket 7: Mode Toggle Implementation
- [ ] Create mode selection UI
- [ ] Implement mode-specific routing
- [ ] Test mode switching

### Ticket 8: LLM Integration (Stretch Goal)
- [ ] Set up LLM API integration
- [ ] Implement refinement logic
- [ ] Add configuration options
- [ ] Test with real-world scenarios

## Current Focus
- Moving to Ticket 3: Implementing Whisper + VSR Fusion Logic
- Planning fusion algorithm design

## Next Steps
1. Design confidence scoring system for both models
2. Implement prioritization logic
3. Create fallback mechanism
4. Add inaudible segment marking

## Technical Notes
- Successfully implemented video transcription with FFmpeg integration
- Whisper model properly handles video input through audio extraction
- Timestamp preservation working in transcription output
- Unit tests confirm functionality
- Ticket 2 (VSR timestamp synchronization) put on hold to focus on fusion logic first
