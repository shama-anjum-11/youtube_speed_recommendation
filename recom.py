# ================== 0. INSTALL DEPENDENCIES (Colab) ==================
# Run this cell once per runtime.


# ================== 1. IMPORTS & GLOBALS =============================

import os
import warnings
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import timedelta

import librosa
import soundfile as sf
from pytubefix import YouTube
from pytubefix.cli import on_progress
from faster_whisper import WhisperModel
import yt_dlp

import re
import statistics

# Allowed playback speeds
ALLOWED_SPEEDS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

WORD_RE = re.compile(r"\w+")
DISPLAY_SPEEDS = [1.0, 1.25, 1.5, 1.75, 2.0]

# ================== 2. YOUTUBE DOWNLOAD ==============================

def download_youtube_audio(url: str, out_dir: str = "downloads") -> str:
    """
    Download YouTube audio using pytubefix without forcing an mp3 rename.
    The returned file keeps the YouTube stream's native extension (usually webm/mp4),
    which avoids corrupt headers from renaming. [web:32][web:37]
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    yt = YouTube(url, on_progress_callback=on_progress)
    print("Title:", yt.title)

    stream = (
        yt.streams.filter(only_audio=True)
        .order_by("abr")
        .desc()
        .first()
    )
    out_file = stream.download(output_path=str(out_path))
    return out_file


def _format_time(seconds: float) -> str:
    """
    Format seconds into a human-friendly string.
    """
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _get_video_info(url: str) -> Dict[str, Any]:
    """
    Fetch title and duration without downloading media.
    """
    ydl_opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title"),
        "duration": info.get("duration", 0.0),
        "duration_formatted": _format_time(info.get("duration", 0.0)),
    }


def _calculate_speed_times(duration_seconds: float) -> List[Dict[str, Any]]:
    """
    Compute completion times at preset speeds for display.
    """
    results = []
    for speed in DISPLAY_SPEEDS:
        time_at_speed = duration_seconds / speed if speed else duration_seconds
        results.append(
            {
                "speed": f"{speed}x",
                "time": _format_time(time_at_speed),
                "seconds": time_at_speed,
            }
        )
    return results

# ================== 3. AUDIO → WAV + TRANSCRIBE ======================

def _get_ffmpeg_bin() -> str:
    """
    Resolve ffmpeg binary path. Allows override via FFMPEG_PATH env.
    """
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    found = shutil.which("ffmpeg")
    if found:
        return found

    raise RuntimeError(
        "ffmpeg is required but was not found on PATH. "
        "Install ffmpeg (https://ffmpeg.org/download.html) or set FFMPEG_PATH "
        "to the ffmpeg binary and try again."
    )


def mp3_to_wav(audio_path: str, target_sr: int = 16000) -> str:
    """
    Convert downloaded audio to mono 16 kHz WAV for Whisper via ffmpeg to avoid
    decoder issues seen with direct mp3 renames. [web:55]
    """
    ffmpeg_bin = _get_ffmpeg_bin()
    wav_path = str(Path(audio_path).with_suffix(".wav"))
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        wav_path,
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to convert audio: {exc.stderr.decode(errors='ignore')}"
        ) from exc
    return wav_path

# Small / tiny model for speed. [web:23][web:25]
MODEL_SIZE = "tiny"
whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def transcribe_with_faster_whisper(wav_path: str):
    """
    Transcribe WAV with Faster‑Whisper and return segments, full text, and duration. [web:23][web:55]
    """
    segments_gen, info = whisper_model.transcribe(
        wav_path,
        beam_size=1,
        condition_on_previous_text=False,
        language="en",
    )
    segments = list(segments_gen)
    full_text = " ".join(s.text.strip() for s in segments)
    audio_duration = float(info.duration)
    return segments, full_text, audio_duration

# ================== 4. FEATURE EXTRACTION ============================

def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))

def _segments_to_dicts(segments) -> List[Dict[str, Any]]:
    data = []
    for s in segments:
        txt = s.text or ""
        data.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": txt,
                "n_words": _count_words(txt),
            }
        )
    data.sort(key=lambda x: x["start"])
    return data

def _timing_features(
    segs: List[Dict[str, Any]],
    audio_duration: float,
    pause_threshold: float = 0.25,
    window_s: float = 10.0,
) -> Dict[str, Any]:
    """
    Compute WPM, articulation rate, pauses, and speech‑rate variability. [web:66][web:71]
    """
    T_total = float(audio_duration)
    if T_total <= 0:
        return {
            "total_words": 0,
            "overall_wpm": 0.0,
            "articulation_wpm": 0.0,
            "avg_pause_s": 0.0,
            "pauses_per_min": 0.0,
            "n_pauses": 0,
            "srv_std": 0.0,
            "srv_level": "Low",
        }

    total_words = sum(s["n_words"] for s in segs)
    overall_wpm = total_words / (T_total / 60.0)  # typical speech ~130–160 WPM. [web:57][web:59]

    pauses = []
    for i in range(len(segs) - 1):
        gap = segs[i + 1]["start"] - segs[i]["end"]
        if gap > pause_threshold:
            pauses.append(gap)

    total_pause_time = sum(pauses)
    T_speech = max(T_total - total_pause_time, 1e-6)
    articulation_wpm = total_words / (T_speech / 60.0)

    avg_pause_s = statistics.mean(pauses) if pauses else 0.0
    n_pauses = len(pauses)
    pauses_per_min = n_pauses / (T_total / 60.0)

    # Windowed WPM for speech‑rate variability
    wpm_values = []
    t = 0.0
    while t < T_total:
        window_end = min(t + window_s, T_total)
        win_len = window_end - t
        if win_len <= 0:
            break

        words_in_window = 0
        for s in segs:
            if s["end"] <= t:
                continue
            if s["start"] >= window_end:
                break
            words_in_window += s["n_words"]

        wpm_win = words_in_window / (win_len / 60.0)
        wpm_values.append(wpm_win)
        t += window_s

    if len(wpm_values) > 1:
        srv_std = statistics.pstdev(wpm_values)
    else:
        srv_std = 0.0

    if srv_std < 10:
        srv_level = "Low"
    elif srv_std < 25:
        srv_level = "Medium"
    else:
        srv_level = "High"

    return {
        "total_words": total_words,
        "overall_wpm": overall_wpm,
        "articulation_wpm": articulation_wpm,
        "avg_pause_s": avg_pause_s,
        "pauses_per_min": pauses_per_min,
        "n_pauses": n_pauses,
        "srv_std": srv_std,
        "srv_level": srv_level,
    }

def _topic_complexity(text: str) -> Dict[str, Any]:
    """
    Simple lexical complexity heuristic using word length, type–token ratio, and long words. [web:66][web:69]
    """
    words = WORD_RE.findall(text.lower())
    n_words = len(words)
    if n_words == 0:
        return {
            "topic_complexity_level": "Low",
            "avg_word_len": 0.0,
            "type_token_ratio": 0.0,
            "long_word_ratio": 0.0,
        }

    avg_word_len = sum(len(w) for w in words) / n_words
    type_token_ratio = len(set(words)) / n_words
    long_word_ratio = sum(1 for w in words if len(w) >= 8) / n_words

    score = 0
    if avg_word_len >= 5:
        score += 1
    if type_token_ratio >= 0.4:
        score += 1
    if long_word_ratio >= 0.15:
        score += 1

    if score <= 1:
        level = "Low"
    elif score == 2:
        level = "Medium"
    else:
        level = "High"

    return {
        "topic_complexity_level": level,
        "avg_word_len": avg_word_len,
        "type_token_ratio": type_token_ratio,
        "long_word_ratio": long_word_ratio,
    }

def extract_speed_features(
    segments,
    audio_duration: float,
    full_text: str,
    asr_confidence: float | None = None,
) -> Dict[str, Any]:
    """
    Bundle all features needed for heuristic speed recommendation.
    """
    segs = _segments_to_dicts(segments)
    timing = _timing_features(segs, audio_duration)
    lexical = _topic_complexity(full_text)

    return {
        "overall_wpm": timing["overall_wpm"],
        "articulation_wpm": timing["articulation_wpm"],
        "avg_pause_s": timing["avg_pause_s"],
        "pauses_per_min": timing["pauses_per_min"],
        "n_pauses": timing["n_pauses"],
        "srv_std": timing["srv_std"],
        "srv_level": timing["srv_level"],
        "total_words": timing["total_words"],
        "topic_complexity_level": lexical["topic_complexity_level"],
        "avg_word_len": lexical["avg_word_len"],
        "type_token_ratio": lexical["type_token_ratio"],
        "long_word_ratio": lexical["long_word_ratio"],
        # acoustic_quality_level omitted now; ASR confidence not used
    }

# ================== 5. HEURISTIC SPEED RECOMMENDER ===================

def recommend_speed_heuristic(features: dict) -> tuple[float, str]:
    """
    Heuristic mapping from features to playback speed.
    Speeds are constrained to 1.0x–2.0x. [web:57][web:59]
    """
    wpm = features["overall_wpm"]
    complexity = features["topic_complexity_level"]
    srv = features["srv_level"]
    avg_pause = features["avg_pause_s"]

    # 1) Base speed from speaking rate (no slower than 1.0x)
    if wpm < 130:
        base = 1.75   # quite slow → can speed up more
    elif wpm < 160:
        base = 1.5    # typical speaking rate
    elif wpm < 190:
        base = 1.25   # already a bit fast
    else:
        base = 1.0    # very fast → keep at normal speed

    reason_parts = [f"Detected speaking rate ≈ {wpm:.0f} WPM."]

    # 2) Adjust for topic complexity. [web:61][web:62]
    if complexity == "High":
        base -= 0.25
        reason_parts.append(
            "Lexical features suggest high topic complexity, so slightly slower playback aids comprehension."
        )
    elif complexity == "Medium":
        reason_parts.append(
            "Content has medium complexity; normal learning speeds are appropriate."
        )
    else:  # Low
        base += 0.25
        reason_parts.append(
            "Content looks relatively simple, so faster playback is usually comfortable."
        )

    # 3) Adjust for speech rate variability & pauses. [web:66][web:69]
    if srv == "High" or avg_pause > 0.8:
        base -= 0.25
        reason_parts.append(
            "Speech rate and pauses vary a lot, so keeping the speed closer to 1.0x improves clarity."
        )

    # 4) Clamp to nearest allowed speed within 1.0–2.0x
    best_speed = min(ALLOWED_SPEEDS, key=lambda s: abs(s - base))
    best_speed = max(min(best_speed, 2.0), 1.0)

    reason = " ".join(reason_parts)
    return best_speed, reason

# ================== 6. END-TO-END DRIVER ============================

def process_youtube_and_recommend_speed(youtube_url: str):
    # 1. Download
    audio_path = download_youtube_audio(youtube_url)

    # 2. Convert to wav
    wav_path = mp3_to_wav(audio_path, target_sr=16000)

    # 3. Transcribe
    segments, full_text, audio_duration = transcribe_with_faster_whisper(wav_path)

    # 4. Features
    features = extract_speed_features(
        segments=segments,
        audio_duration=audio_duration,
        full_text=full_text,
        asr_confidence=None,
    )
    # Removed print statements for features: print("\nComputed features:") and print(features)

    # 5. Heuristic recommendation
    speed, reason = recommend_speed_heuristic(features)

    print(f"\nRecommended playback speed: {speed}x")
    print("Reason:", reason)
    # Return features as well so callers can expose metrics
    return speed, reason, features

# ================== 7. MAIN (INTERACTIVE) ===========================

if __name__ == "__main__":
    url = input("Enter YouTube URL: ").strip()
    process_youtube_and_recommend_speed(url)