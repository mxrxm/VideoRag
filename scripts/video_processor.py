import whisper
import os
from pathlib import Path
import json
from typing import List, Dict
import subprocess
from audio_extractor import AudioExtractor
from ocr_extractor import OCRExtractor
from frame_extractor import FrameExtractor
from transcriber import AudioTranscriber
from video_downloader import VideoDownloader

class VideoProcessor:
    def __init__(self, whisper_model="base", project_root=".", delete_original=False, use_ocr=True, ocr_languages=['en']):
        """
        Initialize the Video RAG processor
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            project_root: Root directory of the project
            delete_original: Whether to delete original video after processing
            use_ocr: Whether to run OCR on frames
            ocr_languages: List of languages for OCR
        """
        self.project_root = Path(project_root)
        self.delete_original = delete_original
        self.use_ocr = use_ocr
        
        # Directory setup
        self.videos_dir = self.project_root / "videos"
        self.extracted_dir = self.project_root / "extracted"
        self.audio_dir = self.extracted_dir / "audio"
        self.frames_dir = self.extracted_dir / "frames"
        self.transcripts_dir = self.project_root / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.audio_extractor = AudioExtractor(project_root)
        self.frame_extractor = FrameExtractor(project_root)
        self.transcriber = AudioTranscriber(model_name=whisper_model)
        self.downloader = VideoDownloader(project_root)
        
        if use_ocr:
            self.ocr_extractor = OCRExtractor(languages=ocr_languages, project_root=project_root)
    

    def process_video(self, video_filename=None, video_source=None, fps=1, language=None, ocr_confidence=0.5, custom_filename=None):
        """
        Complete pipeline: download (if URL), extract audio, frames, transcribe, and OCR
        
        Args:
            video_filename: Video filename in videos/ directory OR a URL (for backward compatibility)
            video_source: Video filename in videos/ directory OR a URL (alternative parameter name)
            fps: Frame extraction rate
            language: Language for transcription or None for auto-detect
            ocr_confidence: Minimum confidence threshold for OCR (0-1)
            custom_filename: Custom filename for downloaded video (optional)
            
        Returns:
            Dictionary with all processed data
        """
        # Support both parameter names for backward compatibility
        if video_source is None and video_filename is None:
            raise ValueError("Either video_filename or video_source must be provided")
        
        # Use video_source if provided, otherwise use video_filename
        video_input = video_source if video_source is not None else video_filename
        
        # Check if video_input is a URL
        if self.downloader.is_url(video_input):
            print(f"\n{'='*60}")
            print(f"Downloading video from URL...")
            print(f"{'='*60}\n")
            
            # Download the video
            video_path = self.downloader.download(video_input, output_filename=custom_filename)
            is_downloaded = True
        else:
            # It's a local file
            video_path = self.videos_dir / video_input
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            is_downloaded = False
        
        video_name = video_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract audio
        print("STEP 1: Extracting audio...")
        audio_path = self.audio_extractor.extract(video_path)
        
        # Step 2: Extract frames
        print("\nSTEP 2: Extracting frames...")
        frames_dir, frames = self.frame_extractor.extract(video_path, fps=fps)
        
        # Step 3: Transcribe audio
        print("\nSTEP 3: Transcribing audio...")
        transcription_result = self.transcriber.transcribe_with_timestamps(
            str(audio_path),
            language=language
        )
                
        # Step 4: Run OCR on frames (if enabled)
        ocr_results = None
        
        if self.use_ocr:
            print("\nSTEP 4: Running OCR on frames...")
            try:
                ocr_data = self.ocr_extractor.extract_from_frames(
                    frames,
                    video_name,
                    confidence_threshold=ocr_confidence
                )
                ocr_results = ocr_data["results"]
                
            except Exception as e:
                print(f"   OCR failed: {e}")
                ocr_results = None
        
        print("\nSTEP 6: Saving Data...")
        segments_with_ocr = self.add_ocr_to_segments(
            transcription_result["segments"],
            ocr_results,
            fps=fps
        )

        # self.transcriber.save_transcription(
        #     transcription_result,
        #     self.transcripts_dir / f"{video_name}.json",
        # )
        
        results = {
            "video_name": video_name,
            "video_source": str(video_input),
            "language": transcription_result["language"],
            "full_transcript": transcription_result["text"],
            "segments": segments_with_ocr,
        }
        
        
        # Save complete results
        results_path = self.transcripts_dir / f"{video_name}_complete.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Video source: {video_input}")
        if is_downloaded:
            print(f"Downloaded to: {video_path.relative_to(self.project_root)}")
        print(f"Audio: {audio_path}")
        print(f"Frames: {len(frames)} in {frames_dir}")
        print(f"Language: {results['language']}")
        print(f"Complete results: {results_path.relative_to(self.project_root)}")
        
        if self.delete_original and is_downloaded:
            video_path.unlink()
            print(f"\nDeleted downloaded video: {video_path.name}")
        
        return results_path,results
    
    def add_ocr_to_segments(self, segments, ocr_results, fps=1):
        """
        Add OCR text directly to transcript segments
        
        Args:
            segments: Transcript segments with timestamps
            ocr_results: OCR results list or None
            fps: Frame extraction rate
            
        Returns:
            Segments with OCR text added
        """
        if not ocr_results:
            # No OCR, just add empty ocr_text to all segments
            for seg in segments:
                seg["ocr_text"] = ""
            return segments
        
        # For each segment, find all frames that overlap with it
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Find frames within this segment's time range
            ocr_texts = []
            
            for ocr_result in ocr_results:
                frame_id = ocr_result.get("frame_id", 0)
                frame_time = frame_id / fps
                
                # If frame is within segment time range
                if seg_start <= frame_time <= seg_end:
                    ocr_text = ocr_result.get("full_text", "")
                    if ocr_text:
                        ocr_texts.append(ocr_text)
            
            # Combine all OCR texts for this segment (remove duplicates)
            seg["ocr_text"] = " ".join(list(dict.fromkeys(ocr_texts)))
        
        return segments
    