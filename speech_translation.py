import os
import sys
import io
import tempfile
import subprocess
import torch
import imageio_ffmpeg
import soundfile as sf
from transformers import pipeline, logging as tf_logging

# Suppress non-critical transformers warnings
tf_logging.set_verbosity_error()

# Configure UTF-8 encoding for standard output and error streams safely on Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(msg):
    """
    Safely writes ASCII diagnostic messages to standard output to prevent Windows charmap exceptions.
    """
    try:
        ascii_msg = str(msg).encode("ascii", errors="replace").decode("ascii")
        sys.stdout.write(ascii_msg + "\n")
        sys.stdout.flush()
    except Exception:
        pass


# Ensure FFmpeg binary from imageio_ffmpeg is accessible
try:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
except Exception as e:
    ffmpeg_exe = "ffmpeg"
    safe_print(f"[Stage 3] Warning setting FFmpeg PATH: {e}")

# Device selection: GPU if available, otherwise CPU
device = 0 if torch.cuda.is_available() else -1
safe_print(f"[Stage 0] Loading fast Whisper-Base model on {'GPU (CUDA)' if device == 0 else 'CPU'}...")

# Load Whisper-Base model ONCE on startup for maximum speed
MODEL_NAME = "openai/whisper-base"
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=device
)

safe_print("[Stage 0] Whisper-Base Speech Recognition model loaded successfully!\n")

WHISPER_LANGUAGES = {
    "en": "english",
    "ta": "tamil",
    "te": "telugu"
}


def convert_to_16k_wav(input_path):
    """
    Converts uploaded audio file (WebM, OGG, MP4, WAV, etc.) to 16kHz mono 16-bit PCM WAV
    using imageio_ffmpeg's bundled ffmpeg executable.
    """
    safe_print(f"[Stage 3] FFmpeg conversion started for '{input_path}'...")
    out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out_wav
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="ignore")
            safe_print(f"[Stage 3] FFmpeg conversion error: {err_msg[:100]}")
            raise RuntimeError(f"FFmpeg audio conversion error: {err_msg[:200]}")

        safe_print("[Stage 4] FFmpeg conversion completed successfully")
        return out_wav
    except Exception as e:
        safe_print(f"[Stage 3] Exception in convert_to_16k_wav: {e}")
        raise


def transcribe_speech(audio_path, source_lang):
    """
    Transcribes audio file to text using Whisper-Base for given source language ('en', 'ta', 'te').
    Decodes audio into float32 numpy array to bypass external system FFmpeg dependency in Hugging Face.
    Returns standard Python Unicode string.
    """
    if source_lang not in WHISPER_LANGUAGES:
        raise ValueError(f"Unsupported source language '{source_lang}'. Must be one of {list(WHISPER_LANGUAGES.keys())}")

    lang_name = WHISPER_LANGUAGES[source_lang]
    converted_wav = None

    try:
        # Audio Conversion
        converted_wav = convert_to_16k_wav(audio_path)

        # WAV Loading
        safe_print("[Stage 5] WAV loading started with soundfile...")
        try:
            audio_array, samplerate = sf.read(converted_wav)
            safe_print(f"[Stage 5] WAV loaded successfully (samples: {len(audio_array)}, samplerate: {samplerate})")
        except Exception as e:
            safe_print(f"[Stage 5] Exception loading WAV file: {e}")
            raise

        if len(audio_array) == 0:
            safe_print("[Stage 5] Audio array is empty (0 samples).")
            return ""

        # Whisper Transcription
        safe_print(f"[Stage 6] Whisper-Base started for language '{lang_name}'...")
        try:
            result = asr_pipeline(
                {"array": audio_array, "sampling_rate": samplerate},
                generate_kwargs={
                    "language": lang_name,
                    "task": "transcribe"
                }
            )
            safe_print("[Stage 7] Whisper-Base completed successfully")
        except Exception as e:
            safe_print(f"[Stage 6] Exception inside Whisper pipeline: {e}")
            raise

        # Transcript Obtained
        transcript = result.get("text", "").strip()
        safe_print(f"[Stage 8] Transcript obtained successfully (length: {len(transcript)} characters)")
        return transcript

    finally:
        if converted_wav and os.path.exists(converted_wav):
            try:
                os.remove(converted_wav)
            except Exception:
                pass
