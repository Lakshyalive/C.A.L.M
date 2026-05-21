"""
voice_input.py — Improved with live level meter, timeout, and robust VAD
"""

import sys
import time
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ── Load Whisper ──────────────────────────────────────────
print("[STT] Loading Whisper small model...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")
print("[STT] Whisper ready")

# ── Settings — TUNE THESE if needed ──────────────────────
MIN_SPEECH_SECS    = 3.0
SAMPLE_RATE        = 16000
CHANNELS           = 1
SPEECH_THRESHOLD   = 0.02
SILENCE_SECONDS    = 5.0    # Was 2.2 — gives more time after you stop
PRE_SPEECH_TIMEOUT = 10.0
MAX_DURATION       = 120.0
SMOOTH_WINDOW      = 15      # NEW — rolling average over last 8 chunks (400ms)


def record_until_silence():
    """
    Record audio with live visual feedback.
    Stops after SILENCE_SECONDS of quiet following speech.
    Returns numpy audio array or None.
    """
    audio_buffer    = []
    state           = {"phase": "waiting"}  # waiting → speaking → silence → done
    silence_start   = [None]
    speech_start    = [time.time()]
    lock            = threading.Lock()

    chunk_size = int(SAMPLE_RATE * 0.05)   # 50ms chunks for responsive meter

    def audio_callback(indata, frames, timestamp, status):
        chunk = indata.flatten().copy()
        rms   = float(np.sqrt(np.mean(chunk ** 2)))

        with lock:
            now = time.time()

            # Draw live level meter
            bars = min(int(rms * 600), 35)
            if state["phase"] == "waiting":
                indicator = "🔇 Waiting"
            elif state["phase"] == "speaking":
                indicator = "🔴 Recording"
            else:
                indicator = "⏸  Silence"

            print(
                f"\r  {indicator}  {'█' * bars:<35}  {rms:.3f}  ",
                end="", flush=True
            )

            # State machine
            if state["phase"] == "waiting":
                if rms > SPEECH_THRESHOLD:
                    state["phase"] = "speaking"
                    silence_start[0] = None
                elif now - speech_start[0] > PRE_SPEECH_TIMEOUT:
                    state["phase"] = "timeout"
                    return  # Give up

            elif state["phase"] == "speaking":
                audio_buffer.append(chunk)
                if rms < SPEECH_THRESHOLD:
                    if silence_start[0] is None:
                        silence_start[0] = now
                    elif now - silence_start[0] >= SILENCE_SECONDS:
                        speech_duration = sum(len(c) for c in audio_buffer) / SAMPLE_RATE
                        if speech_duration >= MIN_SPEECH_SECS:
                            state["phase"] = "done"
                else:
                    silence_start[0] = None  # Reset silence timer

            elif state["phase"] == "done":
                return  # Stop collecting

    print("\n🎙  Listening... (speak naturally, pause when done)\n")

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = "float32",
        blocksize  = chunk_size,
        callback   = audio_callback
    ):
        deadline = time.time() + MAX_DURATION
        while time.time() < deadline:
            time.sleep(0.05)
            with lock:
                if state["phase"] in ("done", "timeout"):
                    break

    print()  # New line after meter

    if state["phase"] == "timeout":
        print("  [No speech detected — try speaking louder or check mic]\n")
        return None

    if not audio_buffer:
        return None

    return np.concatenate(audio_buffer)


def transcribe(audio_array):
    """Convert audio to text via Whisper."""
    if audio_array is None or len(audio_array) < SAMPLE_RATE * 0.3:
        return None

    print("  ✍  Transcribing...", end="\r")

    segments, _ = whisper.transcribe(
        audio_array,
        language      = "en",
        beam_size     = 5,
        vad_filter    = True,
        vad_parameters = dict(min_silence_duration_ms=300)
    )

    text = " ".join(seg.text for seg in segments).strip()
    print("                      ", end="\r")  # Clear line

    return text if text else None


def listen():
    """Main entry point — record then transcribe."""
    audio = record_until_silence()
    if audio is None:
        return None
    return transcribe(audio)