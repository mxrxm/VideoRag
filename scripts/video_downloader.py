import os
import subprocess
import requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import re

class VideoDownloader:
    def __init__(self, project_root="."):
        """
        Initialize video downloader
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.videos_dir = self.project_root / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
    
    def is_url(self, path):
        """
        Check if the given path is a URL
        
        Args:
            path: String to check
            
        Returns:
            Boolean indicating if it's a URL
        """
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def sanitize_filename(self, filename):
        """
        Sanitize filename to remove invalid characters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*#%]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        return filename
    
    def download_with_ytdlp(self, url, output_filename=None):
        """
        Download video using yt-dlp (supports YouTube, Vimeo, etc.)
        
        Args:
            url: Video URL
            output_filename: Optional custom output filename
            
        Returns:
            Path to downloaded video
        """
        print(f" Downloading video from: {url}")
        
        # Determine output path
        if output_filename:
            output_template = str(self.videos_dir / output_filename)
        else:
            output_template = str(self.videos_dir / "%(title)s.%(ext)s")
        
        # yt-dlp command
        cmd = [
            "yt-dlp",
            "-f", "best",  # Download best quality
            "-o", output_template,
            "--no-playlist",  # Don't download playlists
            "--no-warnings",
            url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            # Find the downloaded file
            if output_filename:
                downloaded_file = self.videos_dir / output_filename
            else:
                # Parse yt-dlp output to find filename
                output = result.stdout.decode() + result.stderr.decode()
                # Look for the downloaded file in the videos directory
                video_files = sorted(
                    self.videos_dir.iterdir(),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                downloaded_file = video_files[0] if video_files else None
                self.sanitize_filename(downloaded_file.name)
            if downloaded_file and downloaded_file.exists():
                print(f"Downloaded: {downloaded_file.name}")
                return downloaded_file
            else:
                raise FileNotFoundError("Downloaded file not found")
            
        except subprocess.CalledProcessError as e:
            print(f"yt-dlp error: {e.stderr.decode()}")
            raise
        except FileNotFoundError:
            print("yt-dlp not found.")
            raise
    
    def download_direct(self, url, output_filename=None):
        """
        Download video from direct URL using requests
        
        Args:
            url: Direct video URL
            output_filename: Optional custom output filename
            
        Returns:
            Path to downloaded video
        """
        print(f"Downloading video from: {url}")
        print(f"   Using: Direct download")
        
        # Determine output filename
        if not output_filename:
            # Extract filename from URL
            parsed_url = urlparse(url)
            output_filename = os.path.basename(parsed_url.path)
            
            # If no filename in URL, use a default
            if not output_filename or '.' not in output_filename:
                output_filename = "downloaded_video.mp4"
        
        # Sanitize filename
        output_filename = self.sanitize_filename(output_filename)
        output_path = self.videos_dir / output_filename
        
        try:
            # Download with progress
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='')
            
            print(f"\nDownloaded: {output_path.name}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            print(f" Download error: {e}")
            if output_path.exists():
                output_path.unlink()
            raise
    
    def download(self, url, output_filename=None, force_direct=False):
        """
        Download video from URL (auto-detect method)
        
        Args:
            url: Video URL
            output_filename: Optional custom output filename
            force_direct: Force direct download instead of yt-dlp
            
        Returns:
            Path to downloaded video
        """
        if not self.is_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        # Check if it's a direct video link
        parsed_url = urlparse(url)
        is_direct = parsed_url.path.endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'))
        
        if force_direct or is_direct:
            return self.download_direct(url, output_filename)
        else:
            # Try yt-dlp first (supports many platforms)
            try:
                return self.download_with_ytdlp(url, output_filename)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f" yt-dlp failed, trying direct download...")
                return self.download_direct(url, output_filename)

