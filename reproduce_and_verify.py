import os
import sys
import app

# Reconfigure stdout for UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

client = app.app.test_client()

audio_file = "tests/audio/test_english.wav"
print(f"=== TESTING REPRODUCE AND VERIFY WITH AUDIO '{audio_file}' ===")

directions = [
    ("en", "ta"),
    ("en", "te"),
    ("ta", "en"),
    ("te", "en")
]

for src, tgt in directions:
    with open(audio_file, "rb") as f:
        res = client.post("/api/speech_translate", data={
            "audio": (f, "test_english.wav"),
            "source_lang": src,
            "target_lang": tgt
        })

    print(f"[{src.upper()} -> {tgt.upper()}] HTTP Status: {res.status_code}")
    data = res.get_json() or {}
    print(f"  Success      : {data.get('success')}")
    print(f"  Transcript len: {len(data.get('transcript', ''))}")
    print(f"  Translation len: {len(data.get('translation', ''))}\n")
