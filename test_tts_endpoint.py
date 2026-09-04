import urllib.request
import json

url = "http://127.0.0.1:5000/api/text_to_speech"

test_cases = [
    ("en", "Hello, how are you?"),
    ("ta", "வணக்கம், எப்படி இருக்கிறீர்கள்?"),
    ("te", "నమస్కారం, మీరు ఎలా ఉన్నారు?")
]

print("=== VERIFYING LIVE /api/text_to_speech HTTP SERVER ENDPOINT ===")
for lang, text in test_cases:
    payload = json.dumps({"text": text, "language": lang}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req) as resp:
        content_type = resp.headers.get("Content-Type")
        audio_data = resp.read()
        status = resp.getcode()
        
        print(f"[{lang.upper()}] HTTP Status: {status}")
        print(f"  Content-Type: {content_type}")
        print(f"  Audio Bytes : {len(audio_data)} bytes\n")
        
        assert status == 200, f"Expected HTTP 200, got {status}"
        assert "audio/mpeg" in content_type, f"Expected audio/mpeg, got {content_type}"
        assert len(audio_data) > 1000, f"Expected audio > 1000 bytes, got {len(audio_data)}"

print("ALL 3 LANGUAGES (EN, TA, TE) GENERATED AND STREAMED REAL MP3 AUDIO SUCCESSFULLY WITH HTTP 200 OK!")
