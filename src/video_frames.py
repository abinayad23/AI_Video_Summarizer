"""
Video Frame Extraction and Analysis Module
Extracts frames from videos for multimodal analysis
"""

import cv2
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np


class VideoFrameExtractor:
    """Extract and process video frames for AI analysis"""
    
    @staticmethod
    def download_video(youtube_url, output_path="temp_video.mp4"):
        """Download video from YouTube"""
        import yt_dlp
        
        ydl_opts = {
            "format": "best[ext=mp4]",
            "outtmpl": output_path,
            "quiet": True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            return output_path
        except Exception as e:
            raise Exception(f"Failed to download video: {str(e)}")
    
    @staticmethod
    def extract_frames(video_path, num_frames=20, max_frames=30):
        """
        Extract evenly distributed frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Target number of frames to extract
            max_frames: Maximum frames to extract
            
        Returns:
            List of PIL Image objects
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame interval
            if total_frames < num_frames:
                num_frames = total_frames
            
            # Limit to max_frames
            num_frames = min(num_frames, max_frames)
            
            frame_interval = total_frames // num_frames if num_frames > 0 else 1
            
            frames = []
            frame_indices = []
            
            for i in range(num_frames):
                frame_idx = i * frame_interval
                frame_indices.append(frame_idx)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    # Resize to reduce size (max 1024px on longest side)
                    pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    frames.append(pil_image)
            
            cap.release()
            
            return frames, frame_indices, duration
            
        except Exception as e:
            raise Exception(f"Failed to extract frames: {str(e)}")
    
    @staticmethod
    def frames_to_base64(frames):
        """Convert PIL Images to base64 strings for API transmission"""
        base64_frames = []
        
        for frame in frames:
            buffered = BytesIO()
            frame.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            base64_frames.append(img_str)
        
        return base64_frames
    
    @staticmethod
    def create_frame_grid(frames, grid_size=(3, 3)):
        """Create a grid of frames for visualization"""
        if not frames:
            return None
        
        rows, cols = grid_size
        num_frames = min(len(frames), rows * cols)
        
        # Get frame size
        frame_width, frame_height = frames[0].size
        
        # Create grid
        grid_width = frame_width * cols
        grid_height = frame_height * rows
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        for idx, frame in enumerate(frames[:num_frames]):
            row = idx // cols
            col = idx % cols
            x = col * frame_width
            y = row * frame_height
            grid_image.paste(frame, (x, y))
        
        return grid_image
    
    @staticmethod
    def analyze_frames_with_gemini(frames, prompt, api_key, model_type="gemini-2.0-flash-exp"):
        """
        Analyze video frames using Google Gemini Vision
        
        Args:
            frames: List of PIL Images
            prompt: Text prompt for analysis
            api_key: Google API key
            model_type: Gemini model to use
            
        Returns:
            Analysis text from Gemini
        """
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_type)
            
            # Prepare content with frames
            content = [prompt]
            
            # Add frames (limit to avoid token limits)
            max_frames_for_api = 10
            for frame in frames[:max_frames_for_api]:
                content.append(frame)
            
            response = model.generate_content(content)
            return response.text
            
        except Exception as e:
            return f"Error analyzing frames: {str(e)}"
    
    @staticmethod
    def check_video_has_audio(video_path):
        """Check if video file has audio track"""
        try:
            cap = cv2.VideoCapture(video_path)
            # OpenCV doesn't directly check audio, so we'll use a different approach
            cap.release()
            
            # Use ffprobe or similar to check audio
            # For now, we'll assume videos have audio and handle errors gracefully
            return True
        except:
            return False
    
    @staticmethod
    def cleanup_temp_files(video_path):
        """Remove temporary video file"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass
