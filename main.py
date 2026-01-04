"""
FastAPI backend for YouTube Speed Recommender
Integrates dif_speeds.py and recom.py with a REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from dif_speeds import get_video_info, calculate_speed_times
from recom import process_youtube_and_recommend_speed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Speed Recommender API",
    description="API for analyzing YouTube videos and recommending optimal playback speeds",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080", "*"],  # Allow frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Request/Response Models =============

class VideoAnalysisRequest(BaseModel):
    url: str
    content_type: Optional[str] = "lecture"


class SpeedTimeResult(BaseModel):
    speed: str
    time: str
    seconds: float


class VideoInfoResponse(BaseModel):
    title: str
    duration: int
    duration_formatted: str
    speed_times: list[SpeedTimeResult]


class SpeedRecommendationResponse(BaseModel):
    recommended_speed: float
    reason: str
    features: Optional[dict] = None


class FullAnalysisResponse(BaseModel):
    video_info: VideoInfoResponse
    recommendation: SpeedRecommendationResponse


# ============= API Endpoints =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "YouTube Speed Recommender API is running"}


@app.post("/api/video-info", response_model=VideoInfoResponse)
async def get_video_details(request: VideoAnalysisRequest):
    """
    Get video information including title, duration, and speed completion times.
    This uses dif_speeds.py functionality.
    """
    try:
        logger.info(f"Fetching video info for URL: {request.url}")
        video_info = get_video_info(request.url)
        speed_times = calculate_speed_times(video_info['duration'])
        
        return VideoInfoResponse(
            title=video_info['title'],
            duration=video_info['duration'],
            duration_formatted=video_info['duration_formatted'],
            speed_times=[SpeedTimeResult(**st) for st in speed_times]
        )
    except Exception as e:
        logger.error(f"Error fetching video info: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/recommend-speed", response_model=SpeedRecommendationResponse)
async def recommend_speed(request: VideoAnalysisRequest):
    """
    Get speed recommendation based on video analysis.
    This uses recom.py functionality (transcription, feature extraction, etc.).
    Note: This endpoint may take some time as it downloads and analyzes the audio.
    """
    try:
        logger.info(f"Processing video for speed recommendation: {request.url}")
        
        speed, reason, features = process_youtube_and_recommend_speed(request.url)

        # Map backend feature keys to frontend-expected keys
        metrics = None
        if isinstance(features, dict):
            # recom.extract_speed_features uses snake_case keys
            # Round numeric metrics to 2 decimal places for frontend display
            try:
                metrics = {
                    "overallWPM": round(float(features.get("overall_wpm", 0.0)), 2),
                    "articulationWPM": round(float(features.get("articulation_wpm", 0.0)), 2),
                    "avgPauseLength": round(float(features.get("avg_pause_s", 0.0)), 2),
                    "pausesPerMinute": round(float(features.get("pauses_per_min", 0.0)), 2),
                    "speechRateVariability": features.get("srv_level", "Unknown"),
                }
            except Exception:
                metrics = {
                    "overallWPM": 0.0,
                    "articulationWPM": 0.0,
                    "avgPauseLength": 0.0,
                    "pausesPerMinute": 0.0,
                    "speechRateVariability": features.get("srv_level", "Unknown"),
                }

        return SpeedRecommendationResponse(
            recommended_speed=speed,
            reason=reason,
            features={"metrics": metrics} if metrics is not None else None
        )
    except Exception as e:
        logger.error(f"Error getting speed recommendation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze", response_model=FullAnalysisResponse)
async def full_analysis(request: VideoAnalysisRequest):
    """
    Complete analysis combining video info and speed recommendation.
    """
    try:
        logger.info(f"Running full analysis for URL: {request.url}")
        
        # Get video info
        video_info = get_video_info(request.url)
        speed_times = calculate_speed_times(video_info['duration'])
        
        video_response = VideoInfoResponse(
            title=video_info['title'],
            duration=video_info['duration'],
            duration_formatted=video_info['duration_formatted'],
            speed_times=[SpeedTimeResult(**st) for st in speed_times]
        )
        
        # Get recommendation
        speed, reason, features = process_youtube_and_recommend_speed(request.url)

        metrics = None
        if isinstance(features, dict):
            try:
                metrics = {
                    "overallWPM": round(float(features.get("overall_wpm", 0.0)), 2),
                    "articulationWPM": round(float(features.get("articulation_wpm", 0.0)), 2),
                    "avgPauseLength": round(float(features.get("avg_pause_s", 0.0)), 2),
                    "pausesPerMinute": round(float(features.get("pauses_per_min", 0.0)), 2),
                    "speechRateVariability": features.get("srv_level", "Unknown"),
                }
            except Exception:
                metrics = {
                    "overallWPM": 0.0,
                    "articulationWPM": 0.0,
                    "avgPauseLength": 0.0,
                    "pausesPerMinute": 0.0,
                    "speechRateVariability": features.get("srv_level", "Unknown"),
                }

        recommendation = SpeedRecommendationResponse(
            recommended_speed=speed,
            reason=reason,
            features={"metrics": metrics} if metrics is not None else None
        )
        
        return FullAnalysisResponse(
            video_info=video_response,
            recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Error during full analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ============= Root endpoint =============

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "YouTube Speed Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "video_info": "/api/video-info (POST)",
            "recommend_speed": "/api/recommend-speed (POST)",
            "full_analysis": "/api/analyze (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)