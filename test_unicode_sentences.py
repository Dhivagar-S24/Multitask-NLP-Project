import app
import tempfile
import numpy as np
import soundfile as sf

client = app.app.test_client()

test_cases = [
    ("en", "ta", "Hello, how are you?"),
    ("ta", "en", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),
    ("en", "te", "Hello, how are you?"),
    ("te", "en", "నమస్కారం, మీరు ఎలా ఉన్నారు?"),
    ("ta", "te", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),
    ("te", "ta", "నమస్కారం, మీరు ఎలా ఉన్నారు?")
]

print("=== REQUIREMENT 18: UNICODE SENTENCE TRANSLATION TEST ===")
for src, tgt, text in test_cases:
    res = client.post("/api/translate", json={
        "text": text,
        "source": src,
        "target": tgt
    })
    data = res.get_json()
    print(f"[{src.upper()} -> {tgt.upper()}] HTTP {res.status_code}")
    print(f"  Input : {text}")
    print(f"  Result: {data.get('translated_text')}\n")
