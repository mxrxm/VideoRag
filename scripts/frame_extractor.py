import os
import subprocess
from pathlib import Path


class FrameExtractor:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.frames_dir = self.project_root / "extracted" / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(self, video_path, fps=1, quality=2):
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create subdirectory for this video's frames
        video_name = video_path.stem
        video_frames_dir = self.frames_dir / video_name
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output pattern
        frame_output_pattern = str(video_frames_dir / f"{video_name}_frame_%04d.jpg")
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",        # Frame rate filter
            "-q:v", str(quality),       # JPEG quality
            "-y",                       # Overwrite if exists
            frame_output_pattern
        ]
        
        print(f" Extracting frames from {video_path.name} at {fps} fps...")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Get list of extracted frames
            frames = sorted(video_frames_dir.glob(f"{video_name}_frame_*.jpg"))
            
            print(f"Extracted {len(frames)} frames to: {video_frames_dir.relative_to(self.project_root)}")
            
            return video_frames_dir, [str(f) for f in frames]
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frames: {e.stderr.decode()}")
            raise
    
    