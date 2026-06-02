"""
therapy_voice.py — Main application
RAG Therapy AI with Kokoro Voice
Run: python3 therapy_voice.py
"""

import random
import time
import numpy as np
import ollama
import threading
import queue

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from kokoro import KPipeline
from scripts.voice_input import listen
from scripts.prompt_builder import (
    build_system_prompt,
    format_retrieved_docs,
    check_for_crisis
)
from scripts.web_server import (
    start_ui_server,
    send_to_ui,
    set_welcome_messages,
    interrupt_event,
    stop_listening_event,
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION — Edit these if needed
# ═══════════════════════════════════════════════════════════

LLM_MODEL   = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
DB_DIR      = "./therapy_db"
VOICE       = "af_heart"     # af_heart=warm | bf_emma=professional | af_bella=soft
SPEED       = 0.92           # Slightly slower = calming
TOP_K       = 4              # Number of book passages to retrieve per query

# ═══════════════════════════════════════════════════════════
# VARIED OPENING GREETINGS
# One is chosen at random each session so it never feels scripted.
# ═══════════════════════════════════════════════════════════

OPENINGS = [
    (
        "Hello. I'm really glad you're here. "
        "This is your space — no judgment, no rush. "
        "What's on your mind today?"
    ),
    (
        "Hi. However you're feeling right now, that's exactly where we'll start. "
        "I'm here to listen. "
        "What would you like to talk about?"
    ),
    (
        "Hello. Sometimes just showing up is the hardest part, "
        "and you've already done that. "
        "What's been weighing on you lately?"
    ),
    (
        "Hi there. I'm Samantha, and I'm here to listen — really listen. "
        "There's no right or wrong thing to say. "
        "What's on your heart today?"
    ),
    (
        "Welcome. I'm glad you took this step. "
        "Say whatever feels true — this is your space. "
        "How are you really doing?"
    ),
    (
        "Hello. Take a breath — you're safe here. "
        "We'll go at whatever pace feels right. "
        "Where would you like to begin?"
    ),
    (
        "Hi. It takes courage to reach out, and I don't take that lightly. "
        "I'm here with you. "
        "What's been going on?"
    ),
]


def synthesize_sentence(tts, sentence):
    """Convert one sentence to audio array (public, used by the pipeline)."""
    sentence = sentence.strip()
    if not sentence or len(sentence) < 2:
        return None
    chunks = []
    try:
        for _, _, audio in tts(sentence, voice=VOICE, speed=SPEED):
            if audio is not None:
                chunks.append(audio)
        if chunks:
            return np.concatenate(chunks)
    except Exception as e:
        print(f"\n  [TTS error: {e}]")
    return None



# ═══════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════

def initialize():
    """Load all models and databases. Runs once at startup."""

    print("\n" + "="*55)
    print("  THERAPY AI — Initializing")
    print("="*55)

    print("\n[1/5] Loading knowledge base...")
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory  = DB_DIR,
        embedding_function = embeddings,
        collection_name    = "therapy_knowledge"
    )
    collection_size = vectorstore._collection.count()
    print(f"      Loaded {collection_size} passages from 5 books")

    print("[2/5] Loading voice model (Kokoro)...")
    tts = KPipeline(lang_code='en-us')
    print("      Voice model ready")

    print("[3/5] Verifying LLM connection...")
    ollama.chat(
        model    = LLM_MODEL,
        messages = [{"role": "user", "content": "Say OK"}]
    )
    print(f"      LLM ready: {LLM_MODEL}")

    print("[4/5] Starting UI Server...")
    start_ui_server()

    print("\n  All systems ready.\n")
    return vectorstore, tts

# ═══════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════

def retrieve(vectorstore, query, k=TOP_K):
    """Semantic search over the therapy knowledge base."""
    return vectorstore.similarity_search_with_score(query, k=k)

# ═══════════════════════════════════════════════════════════
# STREAMING PIPELINE
# ═══════════════════════════════════════════════════════════

SENTENCE_ENDS = {'.', '!', '?', '…'}




def speak(tts, text):
    """
    Single-shot TTS for short non-RAG responses (opening, farewell, crisis).
    Polls interrupt_event every 30 ms so the user can interrupt at any time.
    After completion, calls sd.stop() + waits 0.25 s so macOS CoreAudio fully
    releases the 24000 Hz output device before listen() opens the 16000 Hz
    InputStream (prevents PaErrorCode -9986 / AUHAL -10851).
    """
    import sounddevice as sd
    audio = synthesize_sentence(tts, text)
    if audio is not None:
        audio = np.array(audio, dtype=np.float32)
        peak  = np.max(np.abs(audio))
        if peak > 0.01:
            audio = (audio / peak) * 0.88

        # Tail silence so the last word isn't clipped
        tail  = np.zeros(int(24000 * 0.3), dtype=np.float32)
        audio = np.concatenate([audio, tail])

        sd.play(audio, samplerate=24000)

        # Wait for playback to finish, bail on interrupt
        duration = len(audio) / 24000.0
        t0 = time.time()
        while time.time() - t0 < duration:
            if interrupt_event.is_set():
                sd.stop()
                time.sleep(0.25)   # Let CoreAudio release the device
                return
            time.sleep(0.03)
        sd.wait()
        sd.stop()                  # Explicitly release output device
        time.sleep(0.25)           # Let macOS CoreAudio fully tear down


def stream_response_with_voice(vectorstore, user_input, history, tts):
    """
    3-thread pipeline:
      Thread 1 — streams LLM tokens, detects sentence boundaries, queues text
      Thread 2 — synthesizes queued sentences into audio arrays
      Thread 3 — plays audio arrays immediately via sounddevice (no subprocess)

    Supports mid-stream interruption via interrupt_event.
    """

    # Clear any leftover interrupt from a previous cycle
    interrupt_event.clear()

    # Start timing for Time-to-Speech latency
    pipeline_start = time.time()

    # ── Retrieve context ──────────────────────────────────
    retrieved     = retrieve(vectorstore, user_input)
    context       = format_retrieved_docs(retrieved)
    system_prompt = build_system_prompt(context)

    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": system_prompt}] + history

    # ── Sources ───────────────────────────────────────────
    sources = [
        f"{doc.metadata['framework']} — {doc.metadata['author']}"
        for doc, score in retrieved if score < 1.5
    ]

    # ── Queues connecting the three threads ───────────────
    synth_queue = queue.Queue()   # Thread 1 → Thread 2: sentence strings
    audio_queue = queue.Queue()   # Thread 2 → Thread 3: numpy audio arrays

    # ── Thread 2: Synthesis ───────────────────────────────
    def synthesis_worker():
        while True:
            sentence = synth_queue.get()
            if sentence is None or interrupt_event.is_set():
                audio_queue.put(None)
                break
            audio = synthesize_sentence(tts, sentence)
            if audio is not None and not interrupt_event.is_set():
                audio_queue.put(audio)

    # ── Thread 3: Playback ────────────────────────────────
    def playback_worker():
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate = 24000,
            channels   = 1,
            dtype      = "float32",
            blocksize  = 4096,
        )
        stream.start()

        first_chunk_played = False
        while True:
            audio_data = audio_queue.get()
            if audio_data is None or interrupt_event.is_set():
                break

            if not first_chunk_played:
                first_chunk_played = True
                tts_latency = time.time() - pipeline_start
                print(f"\n  [Time-to-Speech Latency: {tts_latency:.2f}s]")

            audio_data = np.array(audio_data, dtype=np.float32)
            peak = np.max(np.abs(audio_data))
            if peak > 0.01:
                audio_data = (audio_data / peak) * 0.88

            stream.write(audio_data.reshape(-1, 1))

        # Short silence drain so audio doesn't click on close
        stream.write(np.zeros((int(24000 * 0.15), 1), dtype=np.float32))
        stream.stop()
        stream.close()
        # Let macOS CoreAudio fully release the 24000 Hz device before
        # the next listen() cycle opens a 16000 Hz InputStream.
        import sounddevice as _sd
        try:
            _sd.stop()
        except Exception:
            pass
        time.sleep(0.25)

    # Start background threads
    synth_thread = threading.Thread(target=synthesis_worker, daemon=True)
    play_thread  = threading.Thread(target=playback_worker,  daemon=True)
    synth_thread.start()
    play_thread.start()

    # ── Thread 1: Stream tokens (main thread) ─────────────
    current_sentence = ""
    full_response    = []

    print("\nSamantha: ", end="", flush=True)
    send_to_ui({"type": "state", "value": "speaking"})

    try:
        llm_stream = ollama.chat(
            model    = LLM_MODEL,
            messages = messages,
            stream   = True,
            options  = {
                "temperature":    0.72,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
                "num_predict":    250,
            }
        )

        for chunk in llm_stream:
            if interrupt_event.is_set():
                break

            token = chunk['message']['content']
            print(token, end="", flush=True)
            full_response.append(token)
            current_sentence += token

            send_to_ui({"type": "stream", "role": "samantha", "token": token})

            # Sentence boundary detected — queue for TTS immediately
            if current_sentence.rstrip() and current_sentence.rstrip()[-1] in SENTENCE_ENDS:
                synth_queue.put(current_sentence.strip())
                current_sentence = ""

        # Queue any remaining text after stream ends (or after interrupt)
        if current_sentence.strip() and not interrupt_event.is_set():
            synth_queue.put(current_sentence.strip())

        print("\n")

    finally:
        # ── Shutdown pipeline ─────────────────────────────────
        synth_queue.put(None)   # Tell synthesis thread to stop
        synth_thread.join()     # Wait for synthesis to finish
        play_thread.join()      # Wait for audio to finish playing

    # ── Save to history ───────────────────────────────────
    full_text = "".join(full_response)
    history.append({"role": "assistant", "content": full_text})

    return full_text, list(set(sources)), context

# ═══════════════════════════════════════════════════════════
# CRISIS RESPONSE
# ═══════════════════════════════════════════════════════════

CRISIS_RESPONSE = (
    "I hear you, and what you just shared is something I'm taking "
    "very seriously. You don't have to face this alone right now. "
    "Please reach out to a crisis line immediately — in India you "
    "can call iCall at 9152987821, or text HOME to 741741 from "
    "anywhere. They have real humans available right now. "
    "Can you do that for me?"
)

# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════

def print_welcome():
    print("="*55)
    print("  THERAPY AI")
    print("  Knowledge: Rogers | Burns | Frankl | Brown | van der Kolk")
    print("="*55)
    print("  'clear'  — start a fresh conversation")
    print("  'quit'   — exit")
    print("  Crisis Resources: iCall India — 9152987821")
    print("="*55 + "\n")


def main():
    vectorstore, tts = initialize()
    history      = []

    print_welcome()

    # Pick a random opening so it never sounds scripted
    opening = random.choice(OPENINGS)
    print(f"Samantha: {opening}\n")

    # Send the opening immediately — messages are buffered until the browser
    # connects, so there's no race condition regardless of connection timing.
    send_to_ui({"type": "state", "value": "speaking"})
    send_to_ui({"type": "transcript", "role": "samantha", "text": opening})
    speak(tts, opening)
    send_to_ui({"type": "state", "value": "idle"})

    # Store the opening so page refreshes and late connections still see it.
    set_welcome_messages([
        {"type": "transcript", "role": "samantha", "text": opening},
        {"type": "state", "value": "idle"},
    ])

    while True:
        # Clear all events from the previous cycle
        interrupt_event.clear()
        stop_listening_event.clear()

        try:
            send_to_ui({"type": "state", "value": "listening"})

            user_input = listen(stop_event=stop_listening_event)

            if not user_input:
                print("  [Didn't catch that — try speaking again]\n")
                send_to_ui({"type": "state", "value": "idle"})
                continue

            print(f"\nYou: {user_input}")
            send_to_ui({"type": "transcript", "role": "user", "text": user_input})
            lower = user_input.lower().strip()

            if any(word in lower for word in ["quit", "exit", "goodbye", "bye"]):
                farewell = (
                    "Take good care of yourself. "
                    "Reaching out is always a sign of strength. Goodbye."
                )
                print(f"\nSamantha: {farewell}\n")
                send_to_ui({"type": "state", "value": "speaking"})
                send_to_ui({"type": "transcript", "role": "samantha", "text": farewell})
                speak(tts, farewell)
                break

            if "clear" in lower or "start over" in lower:
                history.clear()
                print("\n[Conversation cleared]\n")
                ack = "Of course. Let's start fresh. What's on your mind?"
                print(f"Samantha: {ack}\n")
                send_to_ui({"type": "state", "value": "speaking"})
                send_to_ui({"type": "transcript", "role": "samantha", "text": ack})
                speak(tts, ack)
                send_to_ui({"type": "state", "value": "idle"})
                continue

            if check_for_crisis(user_input):
                print(f"\nSamantha: {CRISIS_RESPONSE}\n")
                send_to_ui({"type": "state", "value": "speaking"})
                send_to_ui({"type": "transcript", "role": "samantha", "text": CRISIS_RESPONSE})
                speak(tts, CRISIS_RESPONSE)
                send_to_ui({"type": "state", "value": "idle"})
                continue

            print("  Searching knowledge base...\n")
            start = time.time()
            send_to_ui({"type": "state", "value": "thinking"})

            reply, sources, context = stream_response_with_voice(
                vectorstore, user_input, history, tts
            )

            # Samantha's reply is already streamed token-by-token via 'stream' messages.
            send_to_ui({"type": "state", "value": "idle"})

            elapsed = time.time() - start

            if sources:
                print(f"  [Sources: {' | '.join(set(sources))}]")
            print(f"  [{elapsed:.1f}s]\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye. Take care of yourself.")
            break

        except Exception as e:
            print(f"\n[Error: {e}]")
            print("Try speaking again.\n")
            send_to_ui({"type": "state", "value": "idle"})

if __name__ == "__main__":
    main()