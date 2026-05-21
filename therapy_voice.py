"""
therapy_voice.py — Main application
RAG Therapy AI with Kokoro Voice
Run: python3 therapy_voice.py
"""

import sys
import time
import subprocess
import numpy as np
import soundfile as sf
import ollama
import threading
import queue

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from kokoro import KPipeline
from scripts.voice_input import listen

# Add scripts to path
sys.path.insert(0, "./scripts")
from prompt_builder import (
    build_system_prompt,
    format_retrieved_docs,
    check_for_crisis
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
# INITIALIZATION
# ═══════════════════════════════════════════════════════════

def initialize():
    """Load all models and databases. Runs once at startup."""

    print("\n" + "="*55)
    print("  THERAPY AI — Initializing")
    print("="*55)

    print("\n[1/3] Loading knowledge base...")
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory  = DB_DIR,
        embedding_function = embeddings,
        collection_name    = "therapy_knowledge"
    )
    collection_size = vectorstore._collection.count()
    print(f"      Loaded {collection_size} passages from 5 books")

    print("[2/3] Loading voice model (Kokoro)...")
    tts = KPipeline(lang_code='en-us')
    print("      Voice model ready")

    print("[3/3] Verifying LLM connection...")
    test = ollama.chat(
        model    = LLM_MODEL,
        messages = [{"role": "user", "content": "Say OK"}]
    )
    print(f"      LLM ready: {LLM_MODEL}")

    print("\n  All systems ready.\n")
    return vectorstore, tts

# ═══════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════

def retrieve(vectorstore, query, k=TOP_K):
    """Semantic search over the therapy knowledge base."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

# ═══════════════════════════════════════════════════════════
# STREAMING PIPELINE
# ═══════════════════════════════════════════════════════════


SENTENCE_ENDS = {'.', '!', '?', '…'}

def synthesize_sentence(tts, sentence):
    """Convert one sentence to audio array."""
    sentence = sentence.strip()
    if not sentence or len(sentence) < 3:
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

def speak(tts, text):
    """Single-shot TTS for short non-RAG responses."""
    import sounddevice as sd
    audio = synthesize_sentence(tts, text)
    if audio is not None:
        audio = np.array(audio, dtype=np.float32)
        peak  = np.max(np.abs(audio))
        if peak > 0.01:
            audio = (audio / peak) * 0.88

        # Add tail silence so last word doesn't get cut
        tail    = np.zeros(int(24000 * 0.3), dtype=np.float32)
        audio   = np.concatenate([audio, tail])

        sd.play(audio, samplerate=24000)
        sd.wait()

def stream_response_with_voice(vectorstore, user_input, history, tts):
    """
    3-thread pipeline:
    Thread 1 — streams LLM tokens, detects sentence boundaries, queues text
    Thread 2 — synthesizes queued sentences into audio arrays
    Thread 3 — plays audio arrays immediately via sounddevice (no subprocess)
    No thread ever blocks another.
    """

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
            if sentence is None:          # Sentinel — streaming finished
                audio_queue.put(None)     # Pass sentinel to playback
                break
            audio = synthesize_sentence(tts, sentence)
            if audio is not None:
                audio_queue.put(audio)

    # ── Thread 3: Playback ────────────────────────────────
    def playback_worker():
        import sounddevice as sd

        # Open ONE persistent stream for the entire response
        # Writing to it is continuous — no gaps between sentences
        stream = sd.OutputStream(
            samplerate = 24000,
            channels   = 1,
            dtype      = "float32",
            blocksize  = 4096,     # Larger block = smoother, less CPU overhead
        )
        stream.start()

        while True:
            audio_data = audio_queue.get()
            if audio_data is None:
                break

            # Normalize — prevents crackling
            audio_data = np.array(audio_data, dtype=np.float32)
            peak = np.max(np.abs(audio_data))
            if peak > 0.01:
                audio_data = (audio_data / peak) * 0.88

            # Write directly into running stream — instant, no process startup
            stream.write(audio_data.reshape(-1, 1))

        # Drain remaining buffer before closing
        silence = np.zeros((int(24000 * 0.35), 1), dtype=np.float32)
        stream.write(silence)
        stream.stop()
        stream.close()

    # Start background threads
    synth_thread = threading.Thread(target=synthesis_worker, daemon=True)
    play_thread  = threading.Thread(target=playback_worker,  daemon=True)
    synth_thread.start()
    play_thread.start()

    # ── Thread 1: Stream tokens (main thread) ─────────────
    current_sentence = ""
    full_response    = []

    print("\nSamantha: ", end="", flush=True)

    stream = ollama.chat(
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

    for chunk in stream:
        token = chunk['message']['content']
        print(token, end="", flush=True)
        full_response.append(token)
        current_sentence += token

        # Sentence boundary detected — queue text instantly, never block
        if current_sentence.rstrip() and current_sentence.rstrip()[-1] in SENTENCE_ENDS:
            synth_queue.put(current_sentence.strip())
            current_sentence = ""

    # Queue any remaining text after stream ends
    if current_sentence.strip():
        synth_queue.put(current_sentence.strip())

    print("\n")

    # ── Shutdown pipeline ─────────────────────────────────
    synth_queue.put(None)     # Tell synthesis thread to stop
    synth_thread.join()       # Wait for all synthesis to finish
    play_thread.join()        # Wait for all audio to finish playing

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
    print("  Commands:")
    print("  'sources' — show which book passages were retrieved")
    print("  'clear'   — start a fresh conversation")
    print("  'quit'    — exit")
    print("  Crisis Resources: iCall India — 9152987821")
    print("="*55 + "\n")


def main():
    vectorstore, tts = initialize()
    history          = []
    last_sources     = []
    last_context     = ""

    print_welcome()

    opening = (
        "Hello. I'm really glad you're here. "
        "This is a safe space — there's no judgment, only listening. "
        "What's been on your mind lately?"
    )
    print(f"Samantha: {opening}\n")
    speak(tts, opening)

    while True:
        try:
            user_input = listen()

            if not user_input:
                print("  [Didn't catch that — try speaking again]\n")
                continue

            print(f"\nYou: {user_input}")
            lower = user_input.lower().strip()

            if any(word in lower for word in ["quit", "exit", "goodbye", "bye"]):
                farewell = (
                    "Take good care of yourself. "
                    "Reaching out is always a sign of strength. Goodbye."
                )
                print(f"\nSamantha: {farewell}\n")
                speak(tts, farewell)
                break

            if "clear" in lower or "start over" in lower:
                history.clear()
                print("\n[Conversation cleared]\n")
                ack = "Of course. Let's start fresh. What's on your mind?"
                print(f"Samantha: {ack}\n")
                speak(tts, ack)
                continue

            if "show sources" in lower or "what sources" in lower:
                if last_sources:
                    print(f"\n[Retrieved from: {' | '.join(last_sources)}]\n")
                else:
                    print("\n[No sources yet]\n")
                continue

            if check_for_crisis(user_input):
                print(f"\nSamantha: {CRISIS_RESPONSE}\n")
                speak(tts, CRISIS_RESPONSE)
                continue

            print("  Searching knowledge base...\n")
            start = time.time()

            reply, sources, context = stream_response_with_voice(
                vectorstore, user_input, history, tts
            )

            elapsed      = time.time() - start
            last_sources = sources
            last_context = context

            if sources:
                print(f"  [Sources: {' | '.join(set(sources))}]")
            print(f"  [{elapsed:.1f}s]\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye. Take care of yourself.")
            break

        except Exception as e:
            print(f"\n[Error: {e}]")
            print("Try speaking again.\n")



if __name__ == "__main__":
    main()
