"""
voice_input.py — Microphone recording + Whisper transcription
"""

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
SILENCE_SECONDS    = 5.0    # gives enough pause time after you stop speaking
PRE_SPEECH_TIMEOUT = 10.0
MAX_DURATION       = 120.0


def _open_input_stream(callback, chunk_size, max_retries=5, retry_delay=0.6):
    """
    Open sd.InputStream with automatic retry for macOS CoreAudio AUHAL errors.

    Background: when sd.play() (output at 24000 Hz) finishes and the audio
    device's CoreAudio unit hasn't fully torn down yet, opening a new InputStream
    at 16000 Hz triggers kAudioUnitErr_InvalidPropertyValue (-10851) / PaErrorCode
    -9986.  Waiting a short moment and retrying reliably resolves it.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            stream = sd.InputStream(
                samplerate = SAMPLE_RATE,
                channels   = CHANNELS,
                dtype      = "float32",
                blocksize  = chunk_size,
                callback   = callback,
            )
            stream.start()
            return stream
        except sd.PortAudioError as e:
            last_err = e
            if attempt < max_retries - 1:
                # Stop any lingering global output stream before retrying
                try:
                    sd.stop()
                except Exception:
                    pass
                time.sleep(retry_delay)
            # else fall through and raise
    raise last_err


def record_until_silence(stop_event=None):
    """
    Record audio with live visual feedback.
    Stops after SILENCE_SECONDS of quiet following speech.
    Returns numpy audio array or None.

    stop_event — optional threading.Event. When set from outside (e.g. user
    clicks the mic button), recording finishes immediately:
      • If already in 'speaking' phase → transcribe what's buffered.
      • If still in 'waiting' phase    → treat as timeout (no speech).
    """
    audio_buffer  = []
    state         = {"phase": "waiting"}   # waiting → speaking → done/timeout
    silence_start = [None]
    speech_start  = [time.time()]
    lock          = threading.Lock()

    chunk_size = int(SAMPLE_RATE * 0.05)   # 50 ms chunks for responsive meter

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
                    return

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
                    silence_start[0] = None

            elif state["phase"] == "done":
                return  # Stop collecting

    print("\n🎙  Listening... (speak naturally, pause when done)\n")

    try:
        stream = _open_input_stream(audio_callback, chunk_size)
    except Exception as e:
        raise RuntimeError(f"Error opening InputStream: {e}") from e

    try:
        deadline = time.time() + MAX_DURATION
        while time.time() < deadline:
            time.sleep(0.05)
            with lock:
                if state["phase"] in ("done", "timeout"):
                    break
            # External stop signal (user clicked the mic button)
            if stop_event is not None and stop_event.is_set():
                with lock:
                    if state["phase"] == "speaking":
                        state["phase"] = "done"    # Finalise: transcribe what's buffered
                    else:
                        state["phase"] = "timeout" # Nothing useful recorded yet
                break
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

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
        language       = "en",
        beam_size      = 5,
        vad_filter     = True,
        vad_parameters = dict(min_silence_duration_ms=300)
    )

    text = " ".join(seg.text for seg in segments).strip()
    print("                      ", end="\r")  # Clear line

    return text if text else None


def listen(stop_event=None):
    """Main entry point — record then transcribe.

    stop_event — optional threading.Event for early termination.
    When set, recording stops immediately and whatever audio has been
    captured is transcribed (secondary 'done' button behaviour).
    """
    audio = record_until_silence(stop_event=stop_event)
    if audio is None:
        return None
    return transcribe(audio)