# 🎬 AI Video Summarizer & Transcript Generator

<h4 align="center">Transform YouTube videos into AI-powered summaries, timestamps, and transcripts</h4>

<p align="center">
  <a href="https://github.com/siddharthsky/AI-Video-Summarizer/issues"><img src="https://img.shields.io/github/issues/siddharthsky/google-gemini-yt-video-summarizer-AI-p"></a> 
  <a href="https://github.com/siddharthsky/AI-Video-Summarizer/stargazers"><img src="https://img.shields.io/github/stars/siddharthsky/google-gemini-yt-video-summarizer-AI-p"></a>
  <a href="https://github.com/siddharthsky/AI-Video-Summarizer/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg">
  <img src="https://img.shields.io/badge/Status-Fully%20Functional-success.svg">
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://github.com/siddharthsky/AI-Video-Summarizer/blob/main/research/demo3.gif" alt="Demo" width="600">
</p>

---

## 🚀 Quick Start

Get running in 5 minutes:

```bash
# 1. Install FFmpeg
# Windows: choco install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get free Gemini API key
# Visit: https://makersuite.google.com/app/apikey

# 4. Add your API key to .env file
GOOGLE_GEMINI_API_KEY="your-key-here"

# 5. Test setup
python test_setup.py

# 6. Run the app
streamlit run app.py
```

**Access at:** http://localhost:8501

---

## ✨ Features

### 📊 AI Summary Generation
- Condenses videos into 250-word summaries
- Highlights key points and insights
- Structured format with clear sections
- Copy-to-clipboard functionality

### 🕐 AI Timestamp Generation
- Automatically detects main topics
- Creates clickable YouTube chapter links
- Formatted timestamps (HH:MM:SS)
- Perfect for navigation and sharing

### 📝 Full Transcript Extraction
- **Fast:** Fetches YouTube captions when available
- **Universal:** Falls back to Whisper AI for any video
- Supports videos without captions
- Multi-language support

### 🤖 Dual AI Model Support
- **Google Gemini 2.5** - Free tier (1000 requests/day)
- **OpenAI GPT-3.5** - Pay-per-use
- Easy model switching in UI
- Automatic error handling

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup guide |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Detailed installation instructions |
| [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | Complete technical documentation |
| [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) | Real-world examples with outputs |
| [CHANGES_MADE.md](CHANGES_MADE.md) | Recent updates and fixes |

---

## 🎯 Usage

### Basic Workflow

1. **Enter YouTube URL**
   ```
   https://www.youtube.com/watch?v=VIDEO_ID
   ```

2. **Select AI Model**
   - Choose Gemini (free) or ChatGPT (paid)
   - Select model variant

3. **Choose Output Type**
   - 📊 AI Summary - Quick overview
   - 🕐 AI Timestamps - Chapter markers
   - 📝 Full Transcript - Complete text

4. **Generate & Copy**
   - Click "Generate" button
   - Wait for AI processing
   - Copy results with one click

### Example Outputs

#### AI Summary
```markdown
Setting the Stage:
This video discusses machine learning fundamentals...

Key Points:
• Supervised learning techniques
• Neural network architectures
• Real-world applications

Conclusions:
The video provides a solid foundation for beginners...
```

#### AI Timestamps
```markdown
1. [00:00:15](https://youtube.com/watch?v=ID&t=15) Introduction
2. [00:02:30](https://youtube.com/watch?v=ID&t=150) Main Topic
3. [00:05:45](https://youtube.com/watch?v=ID&t=345) Demonstration
```

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **AI Models:** Google Gemini 2.5, OpenAI GPT-3.5
- **Transcription:** YouTube Transcript API, OpenAI Whisper
- **Audio Processing:** yt-dlp, FFmpeg
- **Language:** Python 3.10+

---

## 📋 Prerequisites

- Python 3.10 or higher
- FFmpeg (for audio extraction)
- Valid API key (Gemini or OpenAI)

### Get API Keys

**Google Gemini (Recommended - FREE)**
- Visit: https://makersuite.google.com/app/apikey
- Free tier: 1,000 requests/day
- No credit card required

**OpenAI ChatGPT (Paid)**
- Visit: https://platform.openai.com/api-keys
- Requires billing setup
- ~$0.002 per request

---

## 🔧 Installation

### Detailed Setup

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete instructions.

### Quick Install

```bash
# Clone repository
git clone https://github.com/siddharthsky/AI-Video-Summarizer.git
cd AI-Video-Summarizer

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# Verify setup
python test_setup.py

# Run application
streamlit run app.py
```

---

## 📊 Performance

### Processing Times

| Video Length | Summary | Timestamps | Transcript |
|--------------|---------|------------|------------|
| 5 min | 8 sec | 10 sec | 2 sec* |
| 15 min | 12 sec | 15 sec | 3 sec* |
| 30 min | 15 sec | 20 sec | 5 sec* |
| 60 min | 18 sec | 25 sec | 10 sec* |

*With YouTube captions. Add 2-5 minutes if using Whisper AI.

### API Costs

| Model | Cost | Free Tier |
|-------|------|-----------|
| Gemini 2.5 Flash | FREE | 1000/day |
| Gemini 2.5 Pro | FREE | 50/day |
| GPT-3.5 Turbo | $0.002 | No |

---

## 🎓 Use Cases

- **Students:** Summarize lecture videos
- **Researchers:** Extract key points from talks
- **Content Creators:** Generate video descriptions
- **Accessibility:** Create transcripts for deaf/hard-of-hearing
- **Language Learning:** Get transcripts for study
- **Time Saving:** Quick overview before watching

---

## 🐛 Troubleshooting

### Common Issues

**"API quota exceeded"**
- Get new API key at https://makersuite.google.com/app/apikey
- Wait for daily quota reset

**"FFmpeg not found"**
```bash
# Windows
choco install ffmpeg

# Mac
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

**"Invalid YouTube URL"**
- Check URL format
- Try: `https://www.youtube.com/watch?v=VIDEO_ID`

For more help, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- [Google Gemini API](https://ai.google.dev/)
- [OpenAI API](https://platform.openai.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/siddharthsky/AI-Video-Summarizer/issues)
- **Documentation:** See docs folder
- **Quick Help:** [QUICKSTART.md](QUICKSTART.md)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/SiddharthSky">SiddharthSky</a>
</p>

<p align="center">
  <sub>Last Updated: March 2026 | Version 2.0</sub>
</p> 
