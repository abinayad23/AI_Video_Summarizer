class Prompt:
    @staticmethod
    def prompt1(ID=0, summary_length=250):
        if ID == 0:
            prompt_text = f"""Your task: Condense a video transcript into a captivating and informative summary that highlights key points and engages viewers.

CRITICAL REQUIREMENT - WORD COUNT:
- Your summary MUST be EXACTLY {summary_length} words (±10 words maximum)
- Count your words carefully before responding
- If you go over, cut content. If you're under, add relevant details
- This is a strict requirement - do not exceed {summary_length + 10} words or go below {summary_length - 10} words

Guidelines:
    Focus on essential information: Prioritize the video's core messages, condensing them into point-wise sections.
    Maintain clarity and conciseness: Craft your summary using accessible language, ensuring it's easily understood by a broad audience.
    Capture the essence of the video: Go beyond mere listings. Integrate key insights and interesting aspects to create a narrative that draws readers in.
    Word count: MUST be {summary_length} words (±10 words tolerance)

Structure:
    Setting the Stage: Briefly introduce the video's topic and context.
    Key Points:
        Point A: Describe the first crucial aspect with clarity and depth.
        Point B: Elaborate on a second significant point.
        (Continue listing and describing key points based on available word count)
    Conclusions: Summarize the video's main takeaways.

Additional Tips:
    Tailor the tone: Adjust your language to resonate with the video's intended audience and overall style.
    Weave in storytelling elements: Employ vivid descriptions and engaging transitions to make the summary more memorable.
    Proofread carefully: Ensure your final summary is free of grammatical errors and typos.

REMINDER: Your response must be {summary_length} words (±10 words). Count carefully!

Input transcript:"""

        elif ID == "timestamp":
            prompt_text = """
            Generate timestamps for main chapter/topics in a YouTube video transcript.
            Given text segments with their time, generate timestamps for main topics discussed in the video. Format timestamps as hh:mm:ss and provide clear and concise topic titles.  
           
            Instructions:
            1. List only topic titles and timestamps.
            2. Do not explain the titles.
            3. Include clickable URLs.
            4. Provide output in Markdown format.

            Markdown for output:
            1. [hh:mm:ss](%VIDEO_URL?t=seconds) %TOPIC TITLE 1%
            2. [hh:mm:ss](%VIDEO_URL?t=seconds) %TOPIC TITLE 2%
            - and so on

            Markdown Example:
            1. [00:05:23](https://youtu.be/hCaXor?t=323) Introduction
            2. [00:10:45](https://youtu.be/hCaXor?t=645) Main Topic 1
            3. [00:25:17](https://youtu.be/hCaXor?t=1517) Main Topic 2
            - and so on

            The %VIDEO_URL% (YouTube video link) and transcript are provided below :
            """
            
        elif ID == "transcript":
            prompt_text = """
            """
        
        elif ID == "qa":
            prompt_text = """You are an AI assistant helping users understand video content. 
Based on the video transcript provided below, answer the user's question accurately and concisely.

Instructions:
1. Answer based ONLY on the information in the transcript
2. If the answer is not in the transcript, say "This information is not covered in the video"
3. Be clear, concise, and helpful
4. Quote relevant parts if needed
5. Keep your answer focused on the question

Video Transcript:
"""
        
        elif ID == "qa_multimodal":
            prompt_text = """You are an AI assistant helping users understand video content using BOTH visual frames and audio transcript.

Instructions:
1. Answer based on BOTH the visual frames shown AND the audio transcript
2. For visual questions (e.g., "what is she holding?", "what's on the screen?"), analyze the frames
3. For audio questions (e.g., "what did he say?"), use the transcript
4. Combine both sources when needed for complete answers
5. If the answer is not in either frames or transcript, say "This information is not visible or mentioned in the video"
6. Be specific and reference what you see in the frames
7. Keep your answer focused on the question

Video Frames:
[Frames will be provided]

Video Transcript:
"""
        
        elif ID == "visual_analysis":
            prompt_text = """Analyze the visual content of this video based on the frames provided.

Your task:
1. Describe what you see in the video frames
2. Identify key visual elements, scenes, and activities
3. Note any text, graphics, or important visual information
4. Describe the overall theme and context
5. Mention any notable changes or progression across frames

Provide a detailed visual description that captures the essence of the video content.

Video Frames:
"""
        
        elif ID == "multimodal_summary":
            prompt_text = f"""Create a comprehensive summary of this video using both the visual content (frames) and audio content (transcript).

Your task:
1. Analyze the visual frames to understand what's shown
2. Analyze the transcript to understand what's said
3. Combine both sources to create a complete picture
4. Highlight information that appears in visuals but not in audio
5. Create a cohesive summary that integrates both modalities

CRITICAL REQUIREMENT - WORD COUNT:
- Your summary MUST be EXACTLY {summary_length} words (±10 words maximum)
- Count your words carefully before responding
- If you go over, cut content. If you're under, add relevant details

Provide a rich, multimodal summary that captures both what is seen and heard in the video.

Visual Content (Frames):
[Frames will be provided]

Audio Content (Transcript):
"""

        else:
            prompt_text = "NA" 

        return prompt_text
