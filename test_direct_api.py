import numpy as np
import soundfile as sf
import tempfile
import app

client = app.app.test_client()

# Create 3-second 16kHz audio file
sr = 16000
audio_data = np.random.uniform(-0.05, 0.05, sr * 3).astype(np.float32)

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, audio_data, sr)
    tmp_path = tmp.name

print("=== STEP 4: DIRECT API TEST FOR /api/speech_translate ===")
with open(tmp_path, "rb") as f:
    res = client.post("/api/speech_translate", data={
        "audio": (f, "recording.wav"),
        "source_lang": "en",
        "target_lang": "ta",
        "source": "en",
        "target": "ta"
    })
    print("HTTP Status Code:", res.status_code)
    print("Response JSON:", res.get_json())
