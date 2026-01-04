import re
import yt_dlp
from datetime import timedelta

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def format_time(seconds):
    """Format seconds into readable time format"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def calculate_speed_times(duration_seconds):
    """Calculate completion times at different speeds"""
    speeds = [1, 1.25, 1.5, 1.75, 2]
    results = []
    
    for speed in speeds:
        time_at_speed = duration_seconds / speed
        results.append({
            'speed': f"{speed}x",
            'time': format_time(time_at_speed),
            'seconds': time_at_speed
        })
    
    return results

def get_video_info(url):
    """Get YouTube video information"""
    try:
        ydl_opts = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
        return {
            'title': info.get('title'),
            'duration': info.get('duration', 0),
            'duration_formatted': format_time(info.get('duration', 0))
        }
    except Exception as e:
        raise Exception(f"Error fetching video: {str(e)}")

def main():
    print("=" * 60)
    print("YouTube Video Speed Calculator")
    print("=" * 60)
    print()
    
    url = input("Enter YouTube URL: ").strip()
    
    if not url:
        print("Error: No URL provided")
        return
    
    try:
        print("\nFetching video information...")
        video_info = get_video_info(url)
        
        print("\n" + "=" * 60)
        print(f"Video: {video_info['title']}")
        print(f"Original Duration: {video_info['duration_formatted']}")
        print("=" * 60)
        print()
        
        speed_times = calculate_speed_times(video_info['duration'])
        
        print("Completion Times at Different Speeds:")
        print("-" * 60)
        print(f"{'Speed':<10} {'Time to Complete':<20}")
        print("-" * 60)
        
        for result in speed_times:
            print(f"{result['speed']:<10} {result['time']:<20}")
        
        print("-" * 60)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have a valid YouTube URL and internet connection.")

if __name__ == "__main__":
    main()