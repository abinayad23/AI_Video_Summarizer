import streamlit as st
import os
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI

from src.video_info import GetVideo
from src.model import Model
from src.prompt import Prompt
from src.misc import Misc
from src.timestamp_formatter import TimestampFormatter
from src.copy_module_edit import ModuleEditor
from src.video_frames import VideoFrameExtractor
from src.cnn_analyzer import CNNAnalyzer
from src.accuracy_metrics import AccuracyMetrics
from st_copy_to_clipboard import st_copy_to_clipboard


class AIVideoSummarizer:
    def __init__(self):
        load_dotenv()

        self.youtube_url = None
        self.video_id = None
        self.video_title = None

        self.video_transcript = None
        self.video_transcript_time = None
        self.summary = None
        self.time_stamps = None

        self.model_name = None
        self.gemini_model_type = "gemini-2.5-flash"

        self.col1 = None
        self.col2 = None
        self.col3 = None

        self.model_env_checker = []

        self.audio_path = "temp_audio.wav"
        self.video_path = "temp_video.mp4"

        self.summary_length = 250
        self.video_frames = None
        self.use_multimodal = False
        self.num_frames = 20
        self.cnn_analysis = None
        self.use_cnn_analysis = False

    # ---------------------- MODEL ROUTING ----------------------
    def _is_groq_model(self):
        return self.model_name.startswith("Groq")

    def _call_text_model(self, transcript, prompt, extra=""):
        """Route to correct model based on selection"""
        if self.model_name == "Gemini":
            return Model.google_gemini(transcript, prompt, extra=extra, model_type=self.gemini_model_type)
        elif self.model_name == "ChatGPT":
            return Model.openai_chatgpt(transcript, prompt, extra=extra)
        elif self.model_name == "Groq - Llama 3.3 70B":
            return Model.groq_ai(transcript, prompt, extra=extra, model_id="llama-3.3-70b-versatile")
        elif self.model_name == "Groq - Mistral Saba 24B":
            return Model.mistral_groq(transcript, prompt, extra=extra)
        elif self.model_name == "Groq - Gemma 2 9B":
            return Model.gemma_groq(transcript, prompt, extra=extra)
        elif self.model_name == "Groq - DeepSeek R1 70B":
            return Model.deepseek_groq(transcript, prompt, extra=extra)
        elif self.model_name == "Groq - Llama 3.1 8B":
            return Model.llama3_groq(transcript, prompt, extra=extra)
        elif self.model_name == "Groq - Qwen QwQ 32B":
            return Model.qwen_groq(transcript, prompt, extra=extra)
        elif self.model_name == "BART (Local)":
            return Model.facebook_bart(transcript, prompt, extra=extra)
        else:
            return Model.groq_ai(transcript, prompt, extra=extra)

    def _call_vision_model(self, frames, transcript, prompt):
        """Route to correct vision model based on selection"""
        if self.model_name == "Gemini":
            return Model.google_gemini_vision(frames, transcript, prompt, model_type=self.gemini_model_type)
        elif self.model_name == "ChatGPT":
            return Model.openai_gpt4_vision(frames, transcript, prompt)
        elif self._is_groq_model():
            return Model.groq_vision(frames, transcript, prompt)
        elif self.model_name == "BART (Local)":
            return Model.bart_with_ocr(frames, transcript, prompt)
        else:
            return Model.groq_vision(frames, transcript, prompt)

    # ---------------------- SESSION STATE INIT ----------------------
    @staticmethod
    def _init_session_state():
        defaults = {
            "qa_history": [],
            "qa_enabled": False,
            "current_transcript": None,
            "current_frames": None,
            "multimodal_qa": False,
            # Cache keys
            "cached_video_id": None,
            "cached_summary": None,
            "cached_summary_length": None,
            "cached_summary_model": None,
            "cached_summary_multimodal": None,
            "cached_timestamps": None,
            "cached_timestamps_model": None,
            "cached_transcript": None,
            "cached_cnn_result": None,
            "cached_cnn_method": None,
            "cached_cnn_num_frames": None,
            "cached_frames": None,
            "cached_frames_num": None,
            "transcript_source": "youtube_api",
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ---------------------- YOUTUBE INFO ----------------------
    def get_youtube_info(self):
        self.youtube_url = st.text_input("Enter YouTube Video Link")

        if os.getenv("GOOGLE_GEMINI_API_KEY") and os.getenv("GOOGLE_GEMINI_API_KEY") != "your-new-gemini-key-here":
            self.model_env_checker.append("Gemini")
        if os.getenv("OPENAI_CHATGPT_API_KEY") and os.getenv("OPENAI_CHATGPT_API_KEY") != "your-new-openai-key-here":
            self.model_env_checker.append("ChatGPT")
        if os.getenv("GROQ_API_KEY"):
            self.model_env_checker.append("Groq - Llama 3.3 70B")
            self.model_env_checker.append("Groq - Mistral Saba 24B")
            self.model_env_checker.append("Groq - Gemma 2 9B")
            self.model_env_checker.append("Groq - DeepSeek R1 70B")
            self.model_env_checker.append("Groq - Llama 3.1 8B")
            self.model_env_checker.append("Groq - Qwen QwQ 32B")
        self.model_env_checker.append("BART (Local)")

        if not self.model_env_checker:
            st.error("No valid API keys found. Please update your .env file.")
            st.stop()

        with self.col2:
            self.model_name = st.selectbox("Select the model", self.model_env_checker)
            if self.model_name == "Gemini":
                self.gemini_model_type = st.selectbox(
                    "Select Gemini Model",
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    index=0
                )

        if self.youtube_url:
            try:
                self.video_id = GetVideo.Id(self.youtube_url)
            except Exception as e:
                st.error(f"Invalid YouTube URL: {e}")
                st.stop()

            self.video_title = GetVideo.title(self.youtube_url)
            if "⚠️" in self.video_title:
                st.error(self.video_title)
                st.stop()

            st.write(f"### {self.video_title}")
            st.image(f"http://img.youtube.com/vi/{self.video_id}/0.jpg", width="stretch")

            # Clear cache if video changed
            if st.session_state.cached_video_id != self.video_id:
                for key in [
                    "cached_summary", "cached_summary_length", "cached_summary_model",
                    "cached_summary_multimodal", "cached_timestamps", "cached_timestamps_model",
                    "cached_transcript", "cached_cnn_result", "cached_cnn_method",
                    "cached_cnn_num_frames", "cached_frames", "cached_frames_num",
                    "current_transcript", "current_frames", "qa_history",
                    "qa_enabled", "multimodal_qa", "transcript_source"
                ]:
                    st.session_state[key] = None if key not in ["qa_history"] else []
                st.session_state.qa_enabled = False
                st.session_state.multimodal_qa = False
                st.session_state.cached_video_id = self.video_id

    # ---------------------- AUDIO EXTRACTION ----------------------
    def extract_audio_from_youtube(self):
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "temp_audio.%(ext)s",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
            "quiet": True
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.youtube_url])
            self.audio_path = "temp_audio.wav"
            return True
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")
            return False

    # ---------------------- TRANSCRIPT ----------------------
    def get_transcript(self):
        # Return cached transcript if available
        if st.session_state.cached_transcript:
            self.video_transcript = st.session_state.cached_transcript
            return self.video_transcript

        try:
            st.info("Fetching YouTube transcript...")
            self.video_transcript = GetVideo.transcript(self.youtube_url)
            if self.video_transcript:
                st.success("YouTube transcript fetched successfully")
                st.session_state.cached_transcript = self.video_transcript
                st.session_state.transcript_source = "youtube_api"
                return self.video_transcript
        except Exception as e:
            st.warning(f"YouTube transcript not available: {e}")

        st.info("Generating transcript using Groq Whisper API...")
        if self.extract_audio_from_youtube():
            try:
                client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
                with open(self.audio_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-large-v3-turbo", file=audio_file, response_format="text"
                    )
                self.video_transcript = transcription
                if self.video_transcript:
                    st.success("Transcript generated successfully")
                    if os.path.exists(self.audio_path):
                        os.remove(self.audio_path)
                    st.session_state.cached_transcript = self.video_transcript
                    st.session_state.transcript_source = "whisper"
                    return self.video_transcript
            except Exception as e:
                st.error(f"Transcription failed: {e}")
        return None

    # ---------------------- VIDEO FRAMES ----------------------
    def extract_video_frames(self, num_frames=20):
        # Return cached frames if same count
        if st.session_state.cached_frames and st.session_state.cached_frames_num == num_frames:
            self.video_frames = st.session_state.cached_frames
            return self.video_frames

        try:
            st.info("Downloading video for frame extraction...")
            video_path = VideoFrameExtractor.download_video(self.youtube_url, self.video_path)
            st.info(f"Extracting {num_frames} frames from video...")
            frames, frame_indices, duration = VideoFrameExtractor.extract_frames(video_path, num_frames=num_frames, max_frames=30)

            if frames:
                st.success(f"Extracted {len(frames)} frames from {duration:.1f}s video")
                self.video_frames = frames
                st.session_state.cached_frames = frames
                st.session_state.cached_frames_num = num_frames

                with st.expander("Preview Extracted Frames"):
                    grid_size = (2, 5) if len(frames) <= 10 else (4, 5) if len(frames) <= 20 else (5, 6)
                    grid = VideoFrameExtractor.create_frame_grid(frames, grid_size=grid_size)
                    if grid:
                        st.image(grid, caption=f"Extracted {len(frames)} frames", use_container_width=True)

                VideoFrameExtractor.cleanup_temp_files(video_path)
                return frames
            else:
                st.error("No frames extracted")
                return None
        except Exception as e:
            st.error(f"Frame extraction failed: {e}")
            VideoFrameExtractor.cleanup_temp_files(self.video_path)
            return None

    def display_transcript(self):
        transcript = self.get_transcript()
        if transcript:
            st.subheader("Video Transcript")
            st.text_area("Transcript", transcript, height=400)
            st_copy_to_clipboard(transcript, "Copy Transcript")
        else:
            st.error("Could not generate transcript")

    # ---------------------- SUMMARY ----------------------
    def generate_summary(self):
        # Check cache: same video, model, length, and mode
        cache_hit = (
            st.session_state.cached_summary is not None
            and st.session_state.cached_summary_length == self.summary_length
            and st.session_state.cached_summary_model == self.model_name
            and st.session_state.cached_summary_multimodal == self.use_multimodal
        )
        if cache_hit:
            st.info("Showing cached summary (no API call needed).")
            self.summary = st.session_state.cached_summary
            self.video_transcript = st.session_state.cached_transcript
            self._display_summary()
            return

        if not self.video_transcript:
            self.video_transcript = self.get_transcript()

        if self.use_multimodal:
            if not self.video_frames:
                self.video_frames = self.extract_video_frames(num_frames=self.num_frames)
            if self.use_cnn_analysis and self.video_frames:
                st.info("Running CNN object detection...")
                try:
                    self.cnn_analysis = CNNAnalyzer.generate_cnn_description(self.video_frames, method="mobilenet")
                    st.success("CNN analysis completed")
                except Exception as e:
                    st.warning(f"CNN analysis failed: {e}")

        if not self.video_transcript and not self.video_frames:
            st.error("No content available for summary generation")
            return

        st.info("Generating AI summary...")
        try:
            if self.use_multimodal and self.video_frames:
                st.info(f"Using multimodal analysis (visual + audio) with {self.model_name}...")
                self.summary = self._call_vision_model(
                    self.video_frames,
                    self.video_transcript or "",
                    Prompt.prompt1(ID="multimodal_summary", summary_length=self.summary_length)
                )
            else:
                self.summary = self._call_text_model(
                    self.video_transcript,
                    Prompt.prompt1(summary_length=self.summary_length)
                )

            if isinstance(self.summary, tuple):
                st.error(self.summary[0])
                st.error(f"Error details: {self.summary[1]}")
            elif self.summary:
                # Cache the result
                st.session_state.cached_summary = self.summary
                st.session_state.cached_summary_length = self.summary_length
                st.session_state.cached_summary_model = self.model_name
                st.session_state.cached_summary_multimodal = self.use_multimodal
                self._display_summary()
            else:
                st.error("Summary generation failed")
        except Exception as e:
            st.error(f"Error generating summary: {e}")

    def _display_summary(self):
        st.success("Summary generated successfully")
        if self.use_multimodal and self.video_frames:
            st.info(f"This summary includes both visual and audio analysis ({self.model_name})")
        st.subheader("AI Summary")
        st.markdown(self.summary)
        st_copy_to_clipboard(self.summary, "Copy Summary")
        self.show_accuracy_metrics(
            summary=self.summary,
            transcript=self.video_transcript or "",
            transcript_source=st.session_state.get("transcript_source", "youtube_api")
        )

    # ---------------------- TIMESTAMPS ----------------------
    def generate_time_stamps(self):
        # Check cache
        if st.session_state.cached_timestamps and st.session_state.cached_timestamps_model == self.model_name:
            st.info("Showing cached timestamps.")
            self.time_stamps = st.session_state.cached_timestamps
            self._display_timestamps()
            return

        st.info("Generating timestamps...")
        try:
            self.video_transcript_time = GetVideo.transcript_time(self.youtube_url)
            if not self.video_transcript_time:
                st.error("Could not fetch transcript with timestamps")
                return

            youtube_url_full = f"https://youtube.com/watch?v={self.video_id}"
            self.time_stamps = self._call_text_model(
                self.video_transcript_time,
                Prompt.prompt1(ID="timestamp"),
                extra=youtube_url_full
            )

            if isinstance(self.time_stamps, tuple):
                st.error(self.time_stamps[0])
            elif self.time_stamps:
                st.session_state.cached_timestamps = self.time_stamps
                st.session_state.cached_timestamps_model = self.model_name
                self._display_timestamps()
            else:
                st.error("Timestamp generation failed")
        except Exception as e:
            st.error(f"Error generating timestamps: {e}")

    def _display_timestamps(self):
        st.success("Timestamps generated successfully")
        st.subheader("Video Timestamps")
        st.markdown(self.time_stamps)
        cp_text = TimestampFormatter.format(self.time_stamps)
        st_copy_to_clipboard(cp_text, "Copy Timestamps")

    # ---------------------- CNN ANALYSIS ----------------------
    def generate_cnn_analysis(self, method="MobileNetV2 (Fast)"):
        import numpy as np

        method_map = {
            "MobileNetV2 (Fast)": "mobilenet",
            "ResNet50 (Accurate)": "resnet",
            "Feature Extraction": "features",
            "Scene Detection": "scene"
        }
        cnn_method = method_map.get(method, "mobilenet")

        # Check cache
        if (st.session_state.cached_cnn_result is not None
                and st.session_state.cached_cnn_method == cnn_method
                and st.session_state.cached_cnn_num_frames == self.num_frames):
            st.info("Showing cached CNN analysis.")
            self.cnn_analysis = st.session_state.cached_cnn_result
            self._display_cnn_result(self.cnn_analysis, cnn_method, np)
            return

        if not self.video_frames:
            self.video_frames = self.extract_video_frames(num_frames=self.num_frames)
        if not self.video_frames:
            st.error("No frames available for CNN analysis")
            return

        st.info(f"Running CNN analysis using {method}...")
        try:
            if cnn_method == "scene":
                scene_changes = CNNAnalyzer.detect_scene_changes(self.video_frames, threshold=0.7)
                result = {"method": "scene", "scene_changes": scene_changes}
                st.session_state.cached_cnn_result = result
                st.session_state.cached_cnn_method = cnn_method
                st.session_state.cached_cnn_num_frames = self.num_frames
                self.cnn_analysis = result
                self._display_cnn_result(result, cnn_method, np)

            elif cnn_method == "features":
                features = CNNAnalyzer.extract_features_cnn(self.video_frames[0])
                result = {"method": "features", "features": features}
                st.session_state.cached_cnn_result = result
                st.session_state.cached_cnn_method = cnn_method
                st.session_state.cached_cnn_num_frames = self.num_frames
                self.cnn_analysis = result
                self._display_cnn_result(result, cnn_method, np)

            else:
                st.info(f"Classifying objects in {len(self.video_frames)} frames...")
                analysis_results = CNNAnalyzer.analyze_frames_with_cnn(self.video_frames, method=cnn_method)
                description = CNNAnalyzer.generate_cnn_description(self.video_frames, method=cnn_method)
                result = {"method": cnn_method, "results": analysis_results, "description": description}
                st.session_state.cached_cnn_result = result
                st.session_state.cached_cnn_method = cnn_method
                st.session_state.cached_cnn_num_frames = self.num_frames
                self.cnn_analysis = result
                self._display_cnn_result(result, cnn_method, np)

        except Exception as e:
            st.error(f"Error during CNN analysis: {e}")
            st.info("Make sure tensorflow is installed: pip install tensorflow")

    def _display_cnn_result(self, result, cnn_method, np):
        if cnn_method == "scene":
            scene_changes = result["scene_changes"]
            st.success(f"Detected {len(scene_changes)} scene changes")
            st.subheader("Scene Change Analysis")
            st.write(f"Total Scenes: {len(scene_changes)}")
            st.write(f"Scene Change Frames: {scene_changes}")
            if scene_changes:
                cols = st.columns(min(len(scene_changes), 5))
                for idx, frame_idx in enumerate(scene_changes[:5]):
                    with cols[idx]:
                        st.image(self.video_frames[frame_idx], caption=f"Scene {idx+1} (Frame {frame_idx+1})")

        elif cnn_method == "features":
            features = result.get("features")
            if features is not None:
                st.success("Features extracted successfully")
                st.subheader("CNN Feature Extraction")
                st.write(f"Feature Vector Size: {len(features)}")
                st.write(f"Mean: {np.mean(features):.4f} | Std: {np.std(features):.4f} | Min: {np.min(features):.4f} | Max: {np.max(features):.4f}")
                st.line_chart(features[:100])
            else:
                st.error("Feature extraction failed")

        else:
            analysis_results = result.get("results", [])
            description = result.get("description", "")
            if not analysis_results:
                st.error("CNN analysis failed")
                return

            st.success("CNN analysis completed")
            st.subheader("CNN Object Classification Results")

            all_objects = {}
            for frame_result in analysis_results:
                for item in frame_result["classifications"]:
                    label, conf = item[0], item[1]
                    if label == "Error":
                        continue
                    if label not in all_objects:
                        all_objects[label] = []
                    all_objects[label].append(conf)

            sorted_objects = sorted(all_objects.items(), key=lambda x: (len(x[1]), np.mean(x[1])), reverse=True)

            st.markdown("**Top Detected Objects:**")
            for label, confidences in sorted_objects[:10]:
                avg_conf = np.mean(confidences)
                freq = len(confidences)
                st.write(f"- **{label}**: {avg_conf:.1%} confidence (appears in {freq}/{len(self.video_frames)} frames)")

            with st.expander("Detailed Frame-by-Frame Analysis"):
                for frame_result in analysis_results:
                    st.markdown(f"**Frame {frame_result['frame_number']}:**")
                    for item in frame_result["classifications"]:
                        label, conf = item[0], item[1]
                        err = item[2] if len(item) > 2 else None
                        if label == "Error":
                            st.error(f"  {err}")
                        else:
                            st.write(f"  - {label}: {conf:.1%}")

            st.markdown("**CNN Analysis Summary:**")
            st.text(description)
            st_copy_to_clipboard(description, "Copy CNN Analysis")

    # ---------------------- Q&A ----------------------
    def ask_question(self, question):
        # Check Q&A cache
        qa_cache = st.session_state.get("qa_cache", {})
        cache_key = f"{question.strip().lower()}_{self.model_name}"
        if cache_key in qa_cache:
            return qa_cache[cache_key]

        has_frames = (st.session_state.current_frames is not None
                      and len(st.session_state.current_frames) > 0)

        try:
            if has_frames and st.session_state.multimodal_qa:
                qa_prompt = Prompt.prompt1(ID="qa_multimodal")
                full_prompt = qa_prompt + f"\n\nUser Question: {question}\n\nAnswer:"

                if self.model_name == "Gemini":
                    import google.generativeai as genai
                    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
                    model = genai.GenerativeModel(self.gemini_model_type)
                    content = [full_prompt]
                    for frame in st.session_state.current_frames[:20]:
                        content.append(frame)
                    if st.session_state.current_transcript:
                        content.append(f"\n\nTranscript:\n{st.session_state.current_transcript}")
                    answer = model.generate_content(content).text
                elif self.model_name == "BART (Local)":
                    qa_prompt = Prompt.prompt1(ID="qa")
                    full_context = qa_prompt + (st.session_state.current_transcript or "") + f"\n\nUser Question: {question}\n\nAnswer:"
                    answer = Model.facebook_bart(transcript="", prompt=full_context)
                else:
                    answer = self._call_vision_model(
                        st.session_state.current_frames,
                        st.session_state.current_transcript or "",
                        full_prompt
                    )
            else:
                if self.model_name == "BART (Local)":
                    st.warning("BART does not support Q&A well. Use a Groq model instead.")
                    return None
                qa_prompt = Prompt.prompt1(ID="qa")
                full_context = qa_prompt + (st.session_state.current_transcript or "") + f"\n\nUser Question: {question}\n\nAnswer:"
                answer = self._call_text_model(transcript="", prompt=full_context)

            if isinstance(answer, tuple):
                st.error(answer[0])
                return None

            # Cache the answer
            if "qa_cache" not in st.session_state:
                st.session_state.qa_cache = {}
            st.session_state.qa_cache[cache_key] = answer
            return answer

        except Exception as e:
            st.error(f"Error answering question: {e}")
            return None

    # ---------------------- ACCURACY METRICS ----------------------
    def show_accuracy_metrics(self, summary, transcript, transcript_source="youtube_api"):
        with st.expander("Accuracy Metrics", expanded=True):
            st.markdown("### Summary Quality Analysis")

            m1 = AccuracyMetrics.word_count_accuracy(summary, self.summary_length)
            m2 = AccuracyMetrics.structure_score(summary)
            m3 = AccuracyMetrics.content_coverage(summary, transcript)
            m4 = AccuracyMetrics.compression_ratio(summary, transcript)
            m5 = AccuracyMetrics.readability_score(summary)
            m6 = AccuracyMetrics.transcript_accuracy(transcript, transcript_source)
            m7 = AccuracyMetrics.rouge_f1_score(summary, transcript)
            overall = AccuracyMetrics.overall_accuracy([m1, m2, m3, m4, m5])

            score = overall["overall_score"]
            grade = overall["grade"]
            if score >= 80:
                st.success(f"Overall Accuracy: **{score}%** — Grade: **{grade}**")
            elif score >= 60:
                st.warning(f"Overall Accuracy: **{score}%** — Grade: **{grade}**")
            else:
                st.error(f"Overall Accuracy: **{score}%** — Grade: **{grade}**")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count Accuracy", f"{m1['score']}%", f"{m1['actual']} / {m1['target']} words")
                st.caption(f"Grade: {m1['grade']} | Deviation: +/-{m1['deviation']} words")
            with col2:
                st.metric("Structure Score", f"{m2['score']}%", f"{m2['passed']}/{m2['total']} checks passed")
                st.caption(f"Grade: {m2['grade']}")
            with col3:
                st.metric("Content Coverage", f"{m3['score']}%", f"{m3.get('covered', 0)} keywords matched")
                st.caption(f"Grade: {m3['grade']}")

            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Compression Ratio", f"{m4['score']}%", f"{m4.get('ratio_percent', 0)}% of transcript")
                st.caption(f"Grade: {m4['grade']} | {m4.get('summary_words', 0)} / {m4.get('transcript_words', 0)} words")
            with col5:
                st.metric("Readability", f"{m5['score']}%", f"Flesch: {m5.get('flesch_score', 0)}")
                st.caption(f"Grade: {m5['grade']} | Avg sentence: {m5.get('avg_sentence_length', 0)} words")
            with col6:
                st.metric("Transcript Quality", f"{m6['score']}%", f"Source: {m6.get('source', 'N/A')}")
                st.caption(f"Grade: {m6['grade']} | {m6.get('word_count', 0)} words")

            st.markdown("---")
            st.markdown("**ROUGE F1 Score (Keyword Overlap)**")
            col_p, col_r, col_f = st.columns(3)
            with col_p:
                st.metric("Precision", f"{m7['precision']}%", help="Of words in summary, how many came from transcript")
            with col_r:
                st.metric("Recall", f"{m7['recall']}%", help="Of transcript keywords, how many made it into summary")
            with col_f:
                st.metric("F1 Score", f"{m7['score']}%", f"Grade: {m7['grade']}")
            st.caption(f"TP: {m7['tp']} matched | FP: {m7['fp']} extra | FN: {m7['fn']} missed  |  F1 = 2×(P×R)/(P+R)")

            st.markdown("---")
            st.markdown("**Structure Check Details:**")
            for check, passed in m2["checks"].items():
                icon = "Pass" if passed else "Fail"
                st.write(f"[{icon}] {check}")

            st.markdown("---")
            mode_label = "Multimodal" if self.use_multimodal else "Audio-only"
            st.caption(f"Model: **{self.model_name}** | Mode: {mode_label}")

    # ---------------------- MAIN APP ----------------------
    def run(self):
        st.set_page_config(page_title="AI Video Summarizer", page_icon="", layout="wide")
        st.title("AI Video Summarizer")
        st.markdown("Transform YouTube videos into summaries, timestamps, and transcripts using AI")

        self._init_session_state()

        try:
            editor = ModuleEditor("st_copy_to_clipboard")
            editor.modify_frontend_files()
        except:
            pass

        self.col1, self.col2, self.col3 = st.columns([2, 1, 2])

        with self.col1:
            self.get_youtube_info()

        if self.youtube_url and self.video_id:
            with self.col3:
                st.markdown("### Select Output Type")
                mode = st.radio(
                    "What would you like to generate?",
                    ["AI Summary", "AI Timestamps", "Full Transcript", "CNN Analysis"],
                    index=0
                )

                if mode == "AI Summary":
                    st.markdown("### Summary Length")
                    self.summary_length = st.select_slider(
                        "Choose summary length (words)",
                        options=[100, 150, 200, 250, 300, 400, 500],
                        value=250
                    )
                    st.info(f"Summary will be approximately {self.summary_length} words")

                    st.markdown("### Analysis Mode")
                    self.use_multimodal = st.checkbox(
                        "Use Multimodal Analysis (Visual + Audio)",
                        value=False,
                        help="Analyze both video frames and audio. Takes longer but provides better context."
                    )

                    if self.use_multimodal:
                        st.markdown("### Frame Quality")
                        self.num_frames = st.select_slider(
                            "Number of frames to extract",
                            options=[5, 10, 15, 20, 25, 30],
                            value=20
                        )
                        st.info(f"Will extract {self.num_frames} frames for visual analysis")

                        self.use_cnn_analysis = st.checkbox(
                            "Add CNN Object Detection",
                            value=False,
                            help="Use CNN to detect objects and visual elements in frames"
                        )
                        if self.use_cnn_analysis:
                            st.info("Will use ResNet/MobileNet CNN for object detection")

                        model_notes = {
                            "Gemini": "Gemini: Native vision support",
                            "ChatGPT": "ChatGPT: GPT-4 Vision support",
                            "BART (Local)": "BART: OCR text extraction from frames"
                        }
                        note = model_notes.get(self.model_name, "Groq: Llama 3.2 Vision (experimental)")
                        st.info(note)

                elif mode == "CNN Analysis":
                    st.markdown("### CNN Analysis Options")
                    st.info("Direct CNN-based visual analysis using pre-trained models")

                    cnn_method = st.selectbox(
                        "Select CNN Model",
                        ["MobileNetV2 (Fast)", "ResNet50 (Accurate)", "Feature Extraction", "Scene Detection"]
                    )
                    st.session_state.cnn_method = cnn_method

                    self.num_frames = st.select_slider(
                        "Number of frames to analyze",
                        options=[5, 10, 15, 20],
                        value=10
                    )
                    st.info(f"Will analyze {self.num_frames} frames using {cnn_method}")

                generate_button = st.button("Generate", type="primary", use_container_width=True)

            if generate_button:
                loaders = Misc.loaderx()
                n, loader = loaders[0], loaders[1]

                with st.spinner(loader[n]):
                    if mode == "AI Summary":
                        self.generate_summary()
                        if self.video_transcript:
                            st.session_state.qa_enabled = True
                            st.session_state.current_transcript = self.video_transcript
                            if self.use_multimodal and self.video_frames:
                                st.session_state.current_frames = self.video_frames
                                st.session_state.multimodal_qa = True
                            else:
                                st.session_state.current_frames = None
                                st.session_state.multimodal_qa = False

                    elif mode == "AI Timestamps":
                        self.generate_time_stamps()
                        if self.video_transcript_time:
                            if not self.video_transcript:
                                self.video_transcript = self.get_transcript()
                            st.session_state.qa_enabled = True
                            st.session_state.current_transcript = self.video_transcript
                            st.session_state.current_frames = None
                            st.session_state.multimodal_qa = False

                    elif mode == "CNN Analysis":
                        cnn_method = st.session_state.get("cnn_method", "MobileNetV2 (Fast)")
                        self.generate_cnn_analysis(method=cnn_method)
                        if self.video_frames:
                            st.session_state.current_frames = self.video_frames
                            st.session_state.multimodal_qa = True
                            if not self.video_transcript:
                                self.video_transcript = self.get_transcript()
                            st.session_state.current_transcript = self.video_transcript
                            st.session_state.qa_enabled = True

                    else:  # Full Transcript
                        self.display_transcript()
                        if self.video_transcript:
                            st.session_state.qa_enabled = True
                            st.session_state.current_transcript = self.video_transcript
                            st.session_state.current_frames = None
                            st.session_state.multimodal_qa = False

            # Q&A Section
            if st.session_state.qa_enabled and st.session_state.current_transcript:
                st.markdown("---")
                st.markdown("### Ask Questions About This Video")

                if st.session_state.multimodal_qa and st.session_state.current_frames:
                    st.success("Multimodal Q&A Active: Can answer questions about both visual content and audio")
                else:
                    st.info("Audio-only Q&A: Can answer questions about what was said in the video")

                for qa in st.session_state.qa_history:
                    with st.chat_message("user"):
                        st.write(qa["question"])
                    with st.chat_message("assistant"):
                        st.write(qa["answer"])

                user_question = st.chat_input("Ask a question about the video...")

                if user_question:
                    with st.chat_message("user"):
                        st.write(user_question)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            self.video_transcript = st.session_state.current_transcript
                            answer = self.ask_question(user_question)
                            if answer:
                                st.write(answer)
                                st.session_state.qa_history.append({
                                    "question": user_question,
                                    "answer": answer
                                })
                                st.rerun()

                if st.session_state.qa_history:
                    if st.button("Clear Chat History"):
                        st.session_state.qa_history = []
                        if "qa_cache" in st.session_state:
                            st.session_state.qa_cache = {}
                        st.rerun()

        st.markdown("---")
        st.write(Misc.footer(), unsafe_allow_html=True)


if __name__ == "__main__":
    app = AIVideoSummarizer()
    app.run()
