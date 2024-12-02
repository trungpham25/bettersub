# Current Development Tasks

## Active Tickets

### Ticket 1: Refactor Whisper for Video Input
- [ ] Implement video file audio extraction using ffmpeg
- [ ] Connect extracted audio to Whisper pipeline
- [ ] Ensure timestamp preservation
- [ ] Add unit tests for video file transcription

### Ticket 2: Add Timestamp Synchronization in VSR
- [ ] Implement frame extraction with timestamps
- [ ] Update VSR pipeline for timestamped output
- [ ] Add timestamp validation tests
- [ ] Test with sample videos

### Ticket 3: Implement Whisper + VSR Fusion Logic
- [ ] Define confidence threshold system
- [ ] Implement Whisper prioritization logic
- [ ] Create VSR fallback mechanism
- [ ] Add "inaudible" segment marking

### Ticket 4: Integrate Whisper and VSR Pipeline
- [ ] Create unified video processing pipeline
- [ ] Implement source indication system
- [ ] Add timestamp-based output merging
- [ ] Test with various video scenarios

### Ticket 5: UI Enhancements
- [ ] Add video upload functionality
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
- Setting up version control
- Organizing codebase for ticket implementation
- Planning integration strategy

## Next Steps
1. Complete version control setup
2. Begin work on Ticket 1 (Whisper refactoring)
3. Set up testing infrastructure
4. Plan incremental integration approach

## Dependencies
- Ticket 4 depends on completion of Tickets 1-3
- UI enhancements (Ticket 5) can proceed in parallel
- Export functionality (Ticket 6) requires working pipeline
- LLM integration (Ticket 8) is optional and can be done last

## Technical Notes
- Maintain separation between real-time and video processing modes
- Ensure proper error handling throughout
- Focus on maintainable, testable code
- Document all major components and interfaces
