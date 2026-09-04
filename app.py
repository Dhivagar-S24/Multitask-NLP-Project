import os
import sys
import io
import traceback
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from gtts import gTTS

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

os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Import existing NLP modules
import next_word
import perplexity
import translation
import sentiment
import speech_translation

app = Flask(__name__)

# Preserve UTF-8 Unicode characters in Flask JSON responses without ASCII escaping
app.config["JSON_AS_ASCII"] = False
if hasattr(app, "json") and hasattr(app.json, "ensure_ascii"):
    app.json.ensure_ascii = False


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/translate", methods=["POST"])
def api_translate():
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        source = data.get("source", "").strip().lower()
        target = data.get("target", "").strip().lower()

        if not text:
            return jsonify({"success": False, "status": "error", "error": "Please enter text to translate."}), 400

        if source not in translation.LANGUAGES or target not in translation.LANGUAGES:
            return jsonify({"success": False, "status": "error", "error": "Invalid source or target language."}), 400

        if source == target:
            return jsonify({"success": False, "status": "error", "error": "Source and target languages must be different."}), 400

        result = translation.translate(text, source, target)
        return jsonify({
            "success": True,
            "status": "success",
            "source": source,
            "target": target,
            "original_text": text,
            "translated_text": result
        })
    except Exception as e:
        tb_str = traceback.format_exc()
        safe_print(f"[TextTranslate] Exception:\n{tb_str}")
        return jsonify({"success": False, "status": "error", "error": str(e)}), 500


@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"success": False, "status": "error", "error": "Please enter text for sentiment analysis."}), 400

        res = sentiment.analyze_sentiment(text)
        return jsonify({
            "success": True,
            "status": "success",
            "text": text,
            "sentiment": res["sentiment"],
            "confidence": res["confidence"]
        })
    except Exception as e:
        tb_str = traceback.format_exc()
        safe_print(f"[Sentiment] Exception:\n{tb_str}")
        return jsonify({"success": False, "status": "error", "error": str(e)}), 500


@app.route("/api/next_word", methods=["POST"])
def api_next_word():
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        top_k = int(data.get("top_k", 5))

        if not text:
            return jsonify({"success": False, "status": "error", "error": "Please enter a sentence or prompt."}), 400

        predictions = next_word.predict_next_words(text, top_k=top_k)
        return jsonify({
            "success": True,
            "status": "success",
            "prompt": text,
            "predictions": predictions
        })
    except Exception as e:
        tb_str = traceback.format_exc()
        safe_print(f"[NextWord] Exception:\n{tb_str}")
        return jsonify({"success": False, "status": "error", "error": str(e)}), 500


@app.route("/api/perplexity", methods=["POST"])
def api_perplexity():
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()

        res = perplexity.evaluate_perplexity(text if text else None)
        return jsonify({
            "success": True,
            "status": "success",
            "evaluated_text": text if text else perplexity.test_text.strip(),
            "loss": res["loss"],
            "perplexity": res["perplexity"]
        })
    except Exception as e:
        tb_str = traceback.format_exc()
        safe_print(f"[Perplexity] Exception:\n{tb_str}")
        return jsonify({"success": False, "status": "error", "error": str(e)}), 500


@app.route("/api/speech_translate", methods=["POST"])
def api_speech_translate():
    temp_path = None
    try:
        safe_print("[Stage 1] Audio received at /api/speech_translate")
        source = (request.form.get("source_lang") or request.form.get("source") or "").strip().lower()
        target = (request.form.get("target_lang") or request.form.get("target") or "").strip().lower()

        safe_print(f"[Stage 1] Source: {source}, Target: {target}")

        if source not in translation.LANGUAGES or target not in translation.LANGUAGES:
            safe_print("[Stage 1] Error: Invalid language choice")
            return jsonify({"success": False, "error": "Invalid source or target language selected."}), 400

        if source == target:
            safe_print("[Stage 1] Error: Source and target match")
            return jsonify({"success": False, "error": "Source and target languages must be different."}), 400

        if "audio" not in request.files:
            safe_print("[Stage 1] Error: No audio file attached")
            return jsonify({"success": False, "error": "No audio file provided in request."}), 400

        audio_file = request.files["audio"]
        if not audio_file or audio_file.filename == "":
            safe_print("[Stage 1] Error: Empty audio file attached")
            return jsonify({"success": False, "error": "Empty audio file uploaded."}), 400

        ext = os.path.splitext(audio_file.filename)[1]
        if not ext or len(ext) > 5:
            ext = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name

        audio_size = os.path.getsize(temp_path)
        safe_print(f"[Stage 2] Audio saved: '{temp_path}', Size: {audio_size} bytes")

        if audio_size < 500:
            safe_print("[Stage 2] Audio size too small")
            return jsonify({
                "success": False,
                "error": "No audio was recorded. Please speak clearly into the microphone.",
                "transcript": "",
                "translation": "",
                "source_lang": source,
                "target_lang": target
            }), 400

        # 1. Whisper Speech Recognition
        try:
            transcript = speech_translation.transcribe_speech(temp_path, source)
        except Exception as e:
            safe_print(f"[Stage 6-8] Exception during transcription: {e}")
            tb_str = traceback.format_exc()
            safe_print(f"[Stage 6-8] Traceback:\n{tb_str}")
            return jsonify({"success": False, "error": f"Speech Recognition Error: {str(e)}"}), 500

        if not transcript or not transcript.strip():
            safe_print("[Stage 8] Empty transcript returned from Whisper")
            return jsonify({
                "success": True,
                "transcript": "",
                "translation": "",
                "source_lang": source,
                "target_lang": target
            })

        # 2. NLLB Translation
        safe_print("[Stage 9] NLLB translation started...")
        try:
            translated_text = translation.translate(transcript, source, target)
            safe_print(f"[Stage 10] NLLB translation completed successfully (length: {len(translated_text)} characters)")
        except Exception as e:
            safe_print(f"[Stage 9-10] Exception during NLLB translation: {e}")
            tb_str = traceback.format_exc()
            safe_print(f"[Stage 9-10] Traceback:\n{tb_str}")
            return jsonify({"success": False, "error": f"Translation Error: {str(e)}"}), 500

        safe_print("[Stage 11] JSON response object created successfully")
        safe_print("[Stage 12] Returning JSON response to browser")
        return jsonify({
            "success": True,
            "transcript": transcript,
            "translation": translated_text,
            "source_lang": source,
            "target_lang": target
        })

    except Exception as e:
        tb_str = traceback.format_exc()
        safe_print(f"[Stage 1-12] Global exception in speech_translate:\n{tb_str}")
        return jsonify({"success": False, "error": f"Speech Translation Exception: {str(e)}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route("/api/text_to_speech", methods=["POST"])
def api_text_to_speech():
    """
    Generates clear MP3 speech audio for English ('en'), Tamil ('ta'), or Telugu ('te') using gTTS.
    Returns audio/mpeg MP3 stream.
    """
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        lang = data.get("language", "").strip().lower()

        if not text:
            safe_print("[TTS] Error: Empty text provided")
            return jsonify({"success": False, "error": "Please provide text to convert to speech."}), 400

        if lang not in ["en", "ta", "te"]:
            safe_print(f"[TTS] Error: Unsupported language code '{lang}'")
            return jsonify({"success": False, "error": f"Unsupported language '{lang}'. Must be en, ta, or te."}), 400

        safe_print(f"[TTS] Generating speech audio for language '{lang}' (text length: {len(text)} chars)...")

        # Generate MP3 audio using gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        safe_print("[TTS] Speech audio generated successfully (mimetype: audio/mpeg)")
        return send_file(
            mp3_fp,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name=f"speech_{lang}.mp3"
        )
    except Exception as e:
        tb_str = traceback.format_exc()
        safe_print(f"[TTS] Exception during TTS generation:\n{tb_str}")
        return jsonify({"success": False, "error": f"Unable to generate speech audio: {str(e)}"}), 500


if __name__ == "__main__":
    safe_print("\nStarting Multitask NLP System Web Server...")
    safe_print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
