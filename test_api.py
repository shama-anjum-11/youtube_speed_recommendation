from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def run_health():
    print("== Health ==")
    r = client.get("/health")
    print(r.status_code)
    print(r.json())

def run_video_info():
    print("== Video Info ==")
    payload = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    r = client.post("/api/video-info", json=payload)
    print(r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print("Failed to parse JSON response:", e)
        print(r.text)

def run_recommendation():
    print("== Recommend Speed ==")
    payload = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    try:
        r = client.post("/api/recommend-speed", json=payload, timeout=120)
        print(r.status_code)
        try:
            print(r.json())
        except Exception as e:
            print("Failed to parse JSON response:", e)
            print(r.text)
    except Exception as e:
        print("Request failed:", repr(e))

if __name__ == '__main__':
    run_health()
    run_video_info()
    run_recommendation()
