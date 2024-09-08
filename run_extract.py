# run_extract.py
import sys
from extract_frame import extract_frames  # Adjust the import based on your file structure

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else '../briefs_video.MP4'
    extract_frames(video_path)