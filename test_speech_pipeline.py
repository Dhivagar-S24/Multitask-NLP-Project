import os
import sys
import app

# Force UTF-8 reconfiguration for test script
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

client = app.app.test_client()

audio_file_path = "tests/audio/test_english.wav"

print("=== SPEECH PIPELINE TEST ===")
print(f"Audio file: {audio_file_path}")
exists = os.path.exists(audio_file_path)
print(f"Audio exists: {'YES' if exists else 'NO'}")

if exists:
    size = os.path.getsize(audio_file_path)
    print(f"Audio size: {size} bytes")
    print("Audio format: 16000 Hz, mono")

    # Test POST /api/speech_translate
    with open(audio_file_path, "rb") as f:
        res = client.post("/api/speech_translate", data={
            "audio": (f, "test_english.wav"),
            "source_lang": "en",
            "target_lang": "ta"
        })

    print(f"API HTTP Status Code: {res.status_code}")
    data = res.get_json() or {}

    success = data.get("success", False)
    print(f"API success flag: {success}")

    transcript = data.get("transcript", "")
    translation = data.get("translation", "")

    print(f"Whisper: {'SUCCESS' if success else 'FAILED'}")
    print(f"Transcript length: {len(transcript)} characters")
    print(f"NLLB translation: {'SUCCESS' if success else 'FAILED'}")
    print(f"Translation length: {len(translation)} characters")

    overall = "PASS" if (res.status_code == 200 and success) else "FAIL"
    print(f"Overall result: {overall}")
else:
    print("Overall result: FAIL (Audio file missing)")
