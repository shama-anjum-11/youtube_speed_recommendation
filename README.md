# youtube_speed_recommendation
## 📌 Introduction

This is a small full-stack project that analyzes YouTube videos and recommends an optimal playback speed based on audio characteristics, transcription analysis, and heuristic speed–completion logic.

The backend is built using FastAPI and exposes multiple REST endpoints for video metadata extraction, audio analysis, and playback speed recommendation. Core audio processing, transcription, and ML-based logic are implemented in a separate module for better modularity and clarity.

A Vite + React frontend consumes these APIs and provides an interactive UI where users can input YouTube links and receive recommended playback speeds along with analysis results.

---

## 🧩 System Architecture Overview

- **Backend (FastAPI)**  
  Handles video metadata retrieval, audio processing, transcription, heuristic analysis, and speed recommendation.

- **Core Analysis Layer**  
  Performs audio segmentation, speech-rate analysis, NLP-based word density calculations, and speed vs completion-time evaluation.

- **Frontend (Vite + React)**  
  Provides a user interface to interact with backend APIs and visualize results.

---

## 🛠️ Tech Stack & Key Components

### 🔹 Backend
- **FastAPI + Uvicorn**  
  Entry point: `app` in `main.py`  
  Dependencies listed in `requirements.txt`

- **API Endpoints**
  - `/health`
  - `/api/video-info`
  - `/api/recommend-speed`
  - `/api/analyze`

- **Pydantic Models** (defined in `main.py`)
  - `VideoAnalysisRequest`
  - `SpeedRecommendationResponse`
  - `FullAnalysisResponse`

- **Testing**
  - API test client: `test_api.py`

---

### 🔹 Audio Processing, Transcription & ML
- **Faster-Whisper** for speech-to-text
- **PyTorch**, **torchaudio**
- **librosa**, **soundfile**

Core logic implemented in `recom.py`:
- `transcribe_with_faster_whisper()`
- Helper functions:
  - `_segments_to_dicts`
  - `_count_words`

**External dependency (required):**
- **FFmpeg** (must be installed separately; not included via pip)

---

### 🔹 Video & Timing Utilities
- **yt-dlp** for YouTube video download and metadata extraction

Helper functions implemented in `dif_speeds.py`:
- `get_video_info`
- `calculate_speed_times`

---

### 🔹 Frontend
- **Vite + React + TypeScript**
- **Tailwind CSS**
- **shadcn/ui**

Key frontend files:
- Configuration:
  - `package.json`
  - `vite.config.ts`
  - `tailwind.config.ts`
- Main UI & API integration:
  - `Index.tsx`
  - `api.ts` (defines `API_BASE`)

Frontend directory:
https://github.com/shama-anjum-11/youtube-speed-guide
