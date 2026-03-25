from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
import requests
import re

class GetVideo:
    @staticmethod
    def Id(link):
        pattern = r"(?:v=|youtu\.be\/|shorts\/|embed\/)([0-9A-Za-z_-]{11})"
        match = re.search(pattern, link)

        if not match:
            raise ValueError(f"Could not extract video ID from URL: {link}")

        return match.group(1)

    @staticmethod
    def title(link):
        """Gets the title of a YouTube video."""
        try:
            r = requests.get(link, timeout=10) 
            s = BeautifulSoup(r.text, "html.parser") 
            title = s.find("meta", itemprop="name")["content"]
            return title
        except Exception as e:
            return "⚠️ Could not fetch video title. Please check the link."
        
    @staticmethod
    def transcript(link):
        """Gets the transcript of a YouTube video."""
        video_id = GetVideo.Id(link)
        try:
            # Create an instance and get transcript list
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try to get English transcript first
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # If no English, get the first available
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Get any available transcript
                    for t in transcript_list:
                        transcript = t
                        break
            
            # Fetch returns a list of transcript snippets
            transcript_data = transcript.fetch()
            
            # Handle different return types
            if isinstance(transcript_data, list):
                final_transcript = " ".join(item.get('text', '') if isinstance(item, dict) else str(item.text) for item in transcript_data)
            else:
                # If it's an iterator or generator
                final_transcript = " ".join(str(item.text) if hasattr(item, 'text') else item.get('text', '') for item in transcript_data)
            
            return final_transcript
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise Exception(f"No transcript available for this video")
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")

    @staticmethod
    def transcript_time(link):
        """Gets the transcript of a YouTube video with timestamps."""
        video_id = GetVideo.Id(link)
        try:
            # Create an instance and get transcript list
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try to get English transcript first
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # If no English, get the first available
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Get any available transcript
                    for t in transcript_list:
                        transcript = t
                        break
            
            # Fetch returns a list of transcript snippets
            transcript_data = transcript.fetch()
            
            final_transcript = ""
            for item in transcript_data:
                # Handle both dict and object formats
                if isinstance(item, dict):
                    text = item.get('text', '')
                    start = item.get('start', 0)
                else:
                    text = item.text if hasattr(item, 'text') else ''
                    start = item.start if hasattr(item, 'start') else 0
                
                timevar = round(float(start))
                hours = int(timevar // 3600)
                timevar %= 3600
                minutes = int(timevar // 60)
                timevar %= 60
                timevex = f"{hours:02d}:{minutes:02d}:{timevar:02d}"
                final_transcript += f'{text} "time:{timevex}" '
            
            return final_transcript
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise Exception(f"No transcript available for this video")
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")
