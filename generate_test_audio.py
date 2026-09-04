import os
import wave
import struct
import numpy as np

os.makedirs("tests/audio", exist_ok=True)

# Generate a 3-second 16kHz mono WAV file with synthetic voice-like formants (16000 Hz, 16-bit PCM)
sr = 16000
duration = 3
t = np.linspace(0, duration, sr * duration, False)

# Formant frequencies typical of human speech (fundamental 130Hz + formants 500Hz, 1500Hz, 2500Hz)
signal = 0.3 * np.sin(2 * np.pi * 130 * t) + \
         0.2 * np.sin(2 * np.pi * 500 * t) + \
         0.1 * np.sin(2 * np.pi * 1500 * t) + \
         0.05 * np.sin(2 * np.pi * 2500 * t)

# Apply amplitude envelope (fade-in, fade-out, speech pauses)
envelope = np.ones_like(t)
envelope[:1600] = np.linspace(0, 1, 1600)
envelope[-1600:] = np.linspace(1, 0, 1600)
signal = signal * envelope

audio_int16 = (signal * 32767).astype(np.int16)

paths = [
    "tests/audio/test_english.wav",
    "tests/audio/test_tamil.wav",
    "tests/audio/test_telugu.wav"
]

for path in paths:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    print(f"Generated test audio file: '{path}' (size: {os.path.getsize(path)} bytes)")
