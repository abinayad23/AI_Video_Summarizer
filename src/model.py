import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

class Model:
    def __init__(self):
        load_dotenv()

    @staticmethod
    def google_gemini(transcript, prompt, extra="", model_type="gemini-2.5-flash", conversation_history=None):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
        model = genai.GenerativeModel(model_type)
        try:
            if conversation_history:
                # For Q&A with conversation history
                chat = model.start_chat(history=conversation_history)
                response = chat.send_message(prompt + extra + transcript)
            else:
                response = model.generate_content(prompt + extra + transcript)
            return response.text
        except Exception as e:
            response_error = "⚠️ There is a problem with the API key or with python module."
            return response_error, str(e)
    
    @staticmethod
    def openai_chatgpt(transcript, prompt, extra="", conversation_history=None):
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))
        model = "gpt-3.5-turbo"
        
        if conversation_history:
            # For Q&A with conversation history
            messages = conversation_history + [{"role": "user", "content": prompt + extra + transcript}]
        else:
            messages = [{"role": "system", "content": prompt + extra + transcript}]
        
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            response_error = "⚠️ There is a problem with the API key or with python module."
            return response_error, str(e)
    
    @staticmethod
    def groq_ai(transcript, prompt, extra="", conversation_history=None, model_id="llama-3.3-70b-versatile"):
        load_dotenv()
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        if conversation_history:
            messages = conversation_history + [{"role": "user", "content": prompt + extra + transcript}]
        else:
            messages = [{"role": "user", "content": prompt + extra + transcript}]
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            response_error = "There is a problem with the Groq API key or model."
            return response_error, str(e)

    # ---- Free open-source models via Groq (no extra API key needed) ----

    @staticmethod
    def mistral_groq(transcript, prompt, extra=""):
        """Mistral 7B via Groq — free, fast, open-source"""
        return Model.groq_ai(transcript, prompt, extra, model_id="mistral-saba-24b")

    @staticmethod
    def gemma_groq(transcript, prompt, extra=""):
        """Google Gemma 2 9B via Groq — free, open-source"""
        return Model.groq_ai(transcript, prompt, extra, model_id="gemma2-9b-it")

    @staticmethod
    def deepseek_groq(transcript, prompt, extra=""):
        """DeepSeek R1 Distill Llama 70B via Groq — free, open-source"""
        return Model.groq_ai(transcript, prompt, extra, model_id="deepseek-r1-distill-llama-70b")

    @staticmethod
    def llama3_groq(transcript, prompt, extra=""):
        """Meta Llama 3.1 8B via Groq — free, lightweight, open-source"""
        return Model.groq_ai(transcript, prompt, extra, model_id="llama-3.1-8b-instant")

    @staticmethod
    def qwen_groq(transcript, prompt, extra=""):
        """Qwen QwQ 32B via Groq — free, open-source reasoning model"""
        return Model.groq_ai(transcript, prompt, extra, model_id="qwen-qwq-32b")
    
    @staticmethod
    def facebook_bart(transcript, prompt, extra=""):
        """Uses Facebook's BART model for summarization (local, no API key needed)"""
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            
            # Load BART model and tokenizer
            model_name = "facebook/bart-large-cnn"
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
            
            # Prepare the transcript
            transcript_text = transcript
            
            # BART has a max input length of 1024 tokens
            max_length = 1024
            
            # Tokenize and truncate if needed
            inputs = tokenizer(
                transcript_text, 
                max_length=max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Add context from prompt if needed
            if "timestamp" in prompt.lower():
                return f"📝 Summary:\n{summary}\n\n⚠️ Note: BART model doesn't generate timestamps. Use Groq for timestamps."
            
            return f"📝 Summary:\n{summary}"
            
        except Exception as e:
            response_error = "⚠️ There is a problem with the BART model."
            return response_error, str(e)
    
    @staticmethod
    def google_gemini_vision(frames, transcript, prompt, model_type="gemini-2.0-flash-exp"):
        """
        Use Google Gemini with vision capabilities for multimodal analysis
        
        Args:
            frames: List of PIL Image objects
            transcript: Text transcript
            prompt: Analysis prompt
            model_type: Gemini model (must support vision)
            
        Returns:
            Multimodal analysis text
        """
        load_dotenv()
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
            model = genai.GenerativeModel(model_type)
            
            # Prepare content: prompt + frames + transcript
            content = [prompt]
            
            # Add frames (Gemini can handle up to 20 frames efficiently)
            max_frames = min(len(frames), 20)
            for frame in frames[:max_frames]:
                content.append(frame)
            
            # Add transcript
            if transcript:
                content.append(f"\n\nTranscript:\n{transcript}")
            
            response = model.generate_content(content)
            return response.text
            
        except Exception as e:
            response_error = "⚠️ Error with Gemini Vision API."
            return response_error, str(e)
    
    @staticmethod
    def openai_gpt4_vision(frames, transcript, prompt):
        """
        Use OpenAI GPT-4 Vision for multimodal analysis
        
        Args:
            frames: List of PIL Image objects
            transcript: Text transcript
            prompt: Analysis prompt
            
        Returns:
            Multimodal analysis text
        """
        load_dotenv()
        try:
            import base64
            from io import BytesIO
            
            client = OpenAI(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))
            
            # Convert frames to base64 (GPT-4 can handle up to 20 frames)
            base64_frames = []
            max_frames = min(len(frames), 20)
            for frame in frames[:max_frames]:
                buffered = BytesIO()
                frame.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_frames.append(img_str)
            
            # Prepare messages with images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Add frames
            for img_base64 in base64_frames:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            
            # Add transcript
            if transcript:
                messages[0]["content"].append({
                    "type": "text",
                    "text": f"\n\nTranscript:\n{transcript}"
                })
            
            response = client.chat.completions.create(
                model="gpt-4o",  # GPT-4 with vision
                messages=messages,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            response_error = "⚠️ Error with GPT-4 Vision API."
            return response_error, str(e)
    
    @staticmethod
    def groq_vision(frames, transcript, prompt):
        """
        Use Groq Llama 3.2 Vision for multimodal analysis
        
        Args:
            frames: List of PIL Image objects
            transcript: Text transcript
            prompt: Analysis prompt
            
        Returns:
            Multimodal analysis text
        """
        load_dotenv()
        try:
            import base64
            from io import BytesIO
            
            client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
            
            # Convert frames to base64
            base64_frames = []
            max_frames = 5  # Groq has stricter limits
            for frame in frames[:max_frames]:
                buffered = BytesIO()
                frame.save(buffered, format="JPEG", quality=70)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_frames.append(img_str)
            
            # Prepare messages with images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Add frames
            for img_base64 in base64_frames:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            
            # Add transcript
            if transcript:
                messages[0]["content"].append({
                    "type": "text",
                    "text": f"\n\nTranscript:\n{transcript}"
                })
            
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",  # Llama 3.2 Vision
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback: Describe frames with text
            response_error = "⚠️ Groq Vision not available. Using text-based frame description."
            return Model._describe_frames_fallback(frames, transcript, prompt)
    
    @staticmethod
    def bart_with_ocr(frames, transcript, prompt):
        """
        Use BART with OCR for multimodal analysis
        Extracts text from frames using OCR and combines with transcript
        
        Args:
            frames: List of PIL Image objects
            transcript: Text transcript
            prompt: Analysis prompt
            
        Returns:
            Multimodal analysis text
        """
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            import pytesseract
            
            # Extract text from frames using OCR
            frame_texts = []
            for i, frame in enumerate(frames[:5]):  # Limit to 5 frames
                try:
                    text = pytesseract.image_to_string(frame)
                    if text.strip():
                        frame_texts.append(f"Frame {i+1}: {text.strip()}")
                except:
                    pass
            
            # Combine frame text with transcript
            combined_text = ""
            if frame_texts:
                combined_text += "Visual content from frames:\n" + "\n".join(frame_texts) + "\n\n"
            if transcript:
                combined_text += f"Audio transcript:\n{transcript}"
            
            # Use BART to summarize
            model_name = "facebook/bart-large-cnn"
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
            
            # Tokenize and truncate
            inputs = tokenizer(
                combined_text, 
                max_length=1024, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return f"📝 Multimodal Summary (BART + OCR):\n{summary}\n\n💡 Note: BART uses OCR to extract text from frames."
            
        except Exception as e:
            # Fallback to regular BART
            return Model.facebook_bart(transcript, prompt)
    
    @staticmethod
    def _describe_frames_fallback(frames, transcript, prompt):
        """
        Fallback method: Describe frames using basic analysis
        Used when vision APIs are not available
        """
        try:
            import numpy as np
            
            # Basic frame analysis
            frame_descriptions = []
            for i, frame in enumerate(frames[:5]):
                # Convert to numpy array
                img_array = np.array(frame)
                
                # Basic statistics
                avg_brightness = np.mean(img_array)
                
                # Simple description
                if avg_brightness > 200:
                    desc = "bright scene"
                elif avg_brightness > 100:
                    desc = "normal lighting"
                else:
                    desc = "dark scene"
                
                frame_descriptions.append(f"Frame {i+1}: {desc}")
            
            # Combine with transcript
            visual_info = "Visual analysis (limited): " + ", ".join(frame_descriptions)
            combined = f"{visual_info}\n\nAudio content: {transcript}"
            
            # Use Groq text model for summary
            return Model.groq_ai(combined, prompt)
            
        except Exception as e:
            # Ultimate fallback: just use transcript
            return Model.groq_ai(transcript, prompt)
