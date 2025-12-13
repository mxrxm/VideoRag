"""
Utility functions for the RAG library core module.
Re-exports commonly used loaders for backward compatibility.
"""

# Re-export the video JSON loader from loaders module for backward compatibility
from ..loaders.video_json_Loader import load_asr_ocr_segments_from_json

__all__ = ["load_asr_ocr_segments_from_json"]
