# C.A.L.M.
### Context-Aware Language Memory

> A fully local, offline-first AI therapy companion grounded in five foundational works of modern psychology — with real-time voice interaction, semantic retrieval, and zero cloud dependency.

---

## What Is C.A.L.M.?

C.A.L.M. is a Retrieval-Augmented Generation (RAG) system that transforms five clinical psychology books into a living knowledge base. When you speak, it retrieves the most relevant passages from Rogers, Burns, Frankl, Brown, or van der Kolk — then uses a local LLM to respond with genuine therapeutic grounding, not generic chatbot output.

Everything runs on your machine. No API keys. No data leaves your device.

---

## Demo

> *Coming soon — demo video link here*

```
You:      "I feel like no matter how hard I try, nothing ever changes."

[Sources: RESILIENCE — Brené Brown | CBT — David Burns]

Samantha: "That sense of running in place, despite giving everything you have —
           that's one of the most exhausting feelings there is, and I want you
           to know it makes complete sense that you'd feel that way.
           What does 'trying' look like for you right now?"
```

---

## Architecture

```
Your Voice (Microphone)
        │
        ▼
┌───────────────────┐
│  faster-whisper   │  Speech → Text  (local, int8 quantized)
│  "small" model    │
└────────┬──────────┘
         │  transcribed text
         ▼
┌───────────────────────────────────────────────┐
│              RAG RETRIEVAL ENGINE             │
│                                               │
│  nomic-embed-text  →  ChromaDB Vector Store   │
│  (Ollama, local)       (5 books, ~1000 chunks)│
│                                               │
│   HUMANISTIC   CBT   EXISTENTIAL              │
│   RESILIENCE   TRAUMA                         │
│         ↓ Top-K relevant passages             │
└────────────────────┬──────────────────────────┘
                     │  retrieved context
                     ▼
┌───────────────────────────────────────────────┐
│           INFERENCE ENGINE                    │
│                                               │
│  Dynamic system prompt (context injected)     │
│  Llama 3.2 3B  ←  Ollama  ←  Apple Metal     │
│  Streaming token output                       │
└────────────────────┬──────────────────────────┘
                     │  streaming tokens
         ┌───────────┴──────────────┐
         ▼                          ▼
  ┌─────────────┐          ┌──────────────────┐
  │  Terminal   │          │   Web UI         │
  │  (live      │          │   FastAPI +      │
  │   stream)   │          │   WebSocket      │
  └─────────────┘          └──────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│           3-THREAD VOICE PIPELINE             │
│                                               │
│  Thread 1: Detect sentence boundaries         │
│  Thread 2: Kokoro-82M TTS synthesis           │
│  Thread 3: sounddevice OutputStream           │
│            (gapless, persistent stream)       │
└───────────────────────────────────────────────┘
         │
         ▼
  You hear Samantha speak
```

---

## Knowledge Base

Five frameworks. One cohesive companion.

| Framework | Book | Author | What It Teaches Samantha |
|-----------|------|--------|--------------------------|
| `HUMANISTIC` | On Becoming a Person | Carl Rogers | Unconditional positive regard, empathy, reflective listening |
| `CBT` | Feeling Good: The New Mood Therapy | David D. Burns | Cognitive distortions, thought records, structured reframing |
| `EXISTENTIAL` | Man's Search for Meaning | Viktor Frankl | Meaning-making, purpose, resilience through suffering |
| `RESILIENCE` | The Gifts of Imperfection | Brené Brown | Shame resilience, vulnerability, self-worth, belonging |
| `TRAUMA` | The Body Keeps the Score | Bessel van der Kolk | Trauma-informed care, somatic awareness, grounding |

Retrieval is framework-aware. A question about shame surfaces Brown. A question about intrusive memories surfaces van der Kolk. The system routes to what actually fits.

---

## Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Llama 3.2 3B via Ollama | Local inference, Apple Metal acceleration |
| **Embeddings** | nomic-embed-text via Ollama | Semantic chunk encoding |
| **Vector Database** | ChromaDB | Local persistent similarity search |
| **RAG Framework** | LangChain | Document loading, chunking, retrieval |
| **PDF Processing** | PyMuPDF | Book ingestion and text extraction |
| **Speech-to-Text** | faster-whisper (small, int8) | Local microphone transcription |
| **Text-to-Speech** | Kokoro-82M | Natural voice synthesis, fully offline |
| **Audio I/O** | sounddevice | Mic capture + gapless OutputStream playback |
| **Web UI** | FastAPI + WebSockets | Real-time browser interface with orb states |
| **Platform** | macOS (Apple Silicon) | M1/M2/M3 optimised, 8GB+ RAM |

---

## Repository Structure

```
C.A.L.M/
│
├── therapy_voice.py          # Main application entry point
│
├── scripts/
│   ├── build_database.py     # One-time PDF → ChromaDB ingestion
│   ├── prompt_builder.py     # Dynamic system prompt construction
│   ├── voice_input.py        # Whisper STT + silence-detection VAD
│   ├── web_server.py         # FastAPI + WebSocket UI bridge
│   ├── test_mic.py           # Microphone level diagnostic
│   └── test_retrieval.py     # Vector DB retrieval diagnostic
│
├── static/
│   └── index.html            # Web UI (animated orb + chat interface)
│
├── books/                    # Place your 5 PDFs here (not committed)
│   ├── rogers.pdf
│   ├── burns_cbt.pdf
│   ├── frankl.pdf
│   ├── brown.pdf
│   └── van_der_kolk.pdf
│
├── therapy_db/               # ChromaDB vector store (auto-generated)
├── data/                     # Optional: conversation logs
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- macOS with Apple Silicon (M1 / M2 / M3)
- Python 3.11+
- [Ollama](https://ollama.com) installed
- 5 PDF books placed in the `books/` folder (see Knowledge Base table above)
- ~5 GB free disk space (models + database)

---

### 1 — Clone the Repository

```bash
git clone https://github.com/Lakshyalive/C.A.L.M.git
cd C.A.L.M
```

---

### 2 — Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4 — Pull Local Models via Ollama

```bash
# Start Ollama server (keep this terminal tab open)
ollama serve

# In a new tab — download the LLM (~2 GB)
ollama pull llama3.2:3b

# Download the embedding model (~270 MB)
ollama pull nomic-embed-text
```

---

### 5 — Verify Microphone (Optional but Recommended)

```bash
python3 scripts/test_mic.py
```

Speak during the 5-second test and confirm bars appear. Note the RMS value when speaking normally — if it reads below `0.020`, lower `SPEECH_THRESHOLD` in `scripts/voice_input.py`.

---

### 6 — Build the Knowledge Base

Run once. Reads all five PDFs, chunks them, embeds each chunk, and saves to ChromaDB locally.

```bash
python3 scripts/build_database.py
```

Expected output:
```
Total chunks across all books: ~1000-1200
  HUMANISTIC:  180 chunks
  CBT:         240 chunks
  EXISTENTIAL: 160 chunks
  RESILIENCE:  190 chunks
  TRAUMA:      210 chunks

Database built in 340 seconds
```

This step takes 5–10 minutes. Run it only once. The database persists to `./therapy_db/`.

---

### 7 — Verify Retrieval (Optional)

```bash
python3 scripts/test_retrieval.py
```

Confirms the correct frameworks surface for different query types (shame → Brené Brown, trauma → van der Kolk).

---

### 8 — Run C.A.L.M.

```bash
python3 therapy_voice.py
```

The application will:
1. Load the knowledge base and all models
2. Open the web UI in your browser at `http://127.0.0.1:8000`
3. Speak the opening greeting
4. Begin listening for your voice

---

## How It Works

Every conversation follows this pipeline:

```
1. Listen        →  sounddevice captures microphone
2. Detect        →  VAD (rolling RMS) detects speech end after 5s silence
3. Transcribe    →  faster-whisper converts audio to text (local)
4. Embed query   →  nomic-embed-text encodes the user message
5. Retrieve      →  ChromaDB returns top-4 relevant passages
6. Build prompt  →  Retrieved context injected into Samantha's system prompt
7. Generate      →  Llama 3.2 3B streams response tokens
8. Synthesize    →  Kokoro-82M converts each sentence to audio (Thread 2)
9. Play          →  sounddevice OutputStream plays continuously (Thread 3)
10. Display      →  WebSocket pushes tokens + sources to browser UI
```

The 3-thread voice pipeline (generate → synthesize → play) runs in parallel, so audio starts within ~1.5 seconds of response generation beginning.

---

## Voice Pipeline Detail

The audio playback uses a persistent `sounddevice.OutputStream` — one stream stays open for the entire response and sentences are written directly into it as they synthesize. This eliminates the ~80–100ms gap that subprocess-based players introduce between sentences.

```
Token stream  →  sentence boundary detected
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        synth_queue           (streaming continues)
              │
              ▼
        Kokoro synthesis
              │
              ▼
        audio_queue
              │
              ▼
        sd.OutputStream.write()   ← no restart, no gap
```

---

## Configuration

All tunable settings are at the top of each file.

### `therapy_voice.py`

```python
LLM_MODEL   = "llama3.2:3b"    # Ollama model name
EMBED_MODEL = "nomic-embed-text"
VOICE       = "af_heart"        # af_heart | bf_emma | af_bella
SPEED       = 0.92              # 1.0 = normal, lower = calmer
TOP_K       = 4                 # Passages retrieved per query
```

### `scripts/voice_input.py`

```python
SPEECH_THRESHOLD  = 0.02   # RMS level that counts as speech
                            # Raise if background noise triggers it
                            # Lower if mic is quiet

SILENCE_SECONDS   = 5.0    # Seconds of quiet before recording stops
                            # Raise if speech keeps getting cut off

MIN_SPEECH_SECS   = 3.0    # Minimum recorded duration before stopping
SMOOTH_WINDOW     = 15     # Rolling average window (750ms at 50ms chunks)
```

---

## Voice Commands

While running, you can say:

| Say | Effect |
|-----|--------|
| *"quit"* / *"goodbye"* | Graceful exit with farewell |
| *"clear"* / *"start over"* | Reset conversation history |
| *"show sources"* | Print retrieved book passages to terminal |

---

## Web UI

When the app starts, it automatically opens `http://127.0.0.1:8000` in your browser.

Key UI features and layout:
- **Interface & Typography**: Uses a dark theme utilizing `Inter` (sans-serif) for UI components and `Lora` (serif) for dialogues.
- **Interactive Button (Orb)**:
  - **Click to Interrupt**: Clicking the circle at any time during speaking or thinking immediately halts background synthesis and audio playback via `sounddevice`, returning the system to the listening state.
  - **Manual Done Listening**: Clicking the circle during the active listening phase overrides the silence timeout, halting recording and submitting the captured audio for transcription.
- **Visual State Colors**:
  - 🔵 **Blue (Listening)**: Concentric ring animation representing microphone capture.
  - 🟣 **Purple (Thinking)**: Rotating gradient ring representing document search and LLM response generation.
  - 🟢 **Green (Speaking)**: Breathing wave representing voice synthesis and active playback.
  - ⚫ **Dark (Idle)**: Slow pulse representing a silent/inactive pipeline.
- **Chat Log**: Renders standard dialogue bubbles with live word-by-word streaming of text tokens and badges showing book metadata sources (e.g. `CBT`, `HUMANISTIC`).
- **Telemetry Indicators**: Displays a connection status pill and an elapsed session time counter.

---

## Guardrails

Samantha is designed as a supportive companion, not a clinical tool.

- **No diagnosis** — never identifies or labels a mental health condition
- **No medication advice** — never recommends or discusses dosage
- **Crisis detection** — keyword matching on messages triggers an immediate crisis response with helpline numbers
- **Anti-interrogation rule** — Samantha validates before asking, never chains follow-up questions
- **Professional referral** — always encourages professional care for persistent or serious concerns

### Crisis Resources

| Region | Resource | Contact |
|--------|----------|---------|
| India | iCall | 9152987821 |
| India | Vandrevala Foundation | 1860-2662-345 |
| International | findahelpline.com | — |
| International | Crisis Text Line | Text HOME to 741741 |

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Mic never starts listening | macOS permissions | System Settings → Privacy → Microphone → enable Terminal |
| Recording cuts off mid-sentence | `SILENCE_SECONDS` too low | Raise to `6.0` in `voice_input.py` |
| Background noise triggers recording | `SPEECH_THRESHOLD` too low | Raise to `0.030` |
| Whisper mishears your name | Accent/proper noun | Add name to `initial_prompt` in `transcribe()` |
| Audio crackles | Values outside ±1.0 | Normalization already applied — check sounddevice version |
| Headphones not detected | Plugged in after stream opened | Plug in before starting a new response |
| `ollama: connection refused` | Server not running | Run `ollama serve` in a separate terminal tab |
| ChromaDB import error | Wrong package | `pip install langchain-chroma --upgrade` |
| `No module named scripts` | Wrong working directory | Run from `~/C.A.L.M/`, not a subdirectory |
| Web UI shows "reconnecting" | FastAPI not started | Ensure `web_server.py` imported in `therapy_voice.py` |

---

## Requirements

The application runs on standard Python dependencies pinned to stable working versions from the virtual environment:

```
ollama==0.6.2
langchain-ollama==1.1.0
chromadb==1.5.9
langchain-chroma==1.1.0
langchain==1.3.1
langchain-community==0.4.1
langchain-text-splitters==1.1.2
PyMuPDF==1.27.2.3
faster-whisper==1.2.1
kokoro==0.9.4
sounddevice==0.5.5
soundfile==0.13.1
numpy==2.4.5
fastapi==0.136.1
uvicorn==0.47.0
python-dotenv==1.2.2
rich==15.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Resume Description

> Architected **C.A.L.M.** (Context-Aware Language Memory), a fully local RAG therapy companion grounded in a 5-book psychology corpus (Rogers, Burns, Frankl, Brown, van der Kolk); implemented semantic chunking with `nomic-embed-text` embeddings into a ChromaDB vector store with framework-tagged metadata for framework-aware retrieval; built a 3-thread streaming pipeline (LLM token stream → Kokoro-82M TTS synthesis → `sounddevice` OutputStream) for gapless voice output; integrated faster-Whisper local STT with rolling-RMS VAD; deployed a FastAPI + WebSocket server for a real-time browser UI with an animated state orb — all running offline on Apple Silicon at zero cost.

---

## Roadmap

- [ ] Conversation memory across sessions (persistent history)
- [ ] Improved retrieval scoring with reranking
- [ ] Exportable session transcripts
- [ ] Evaluation pipeline — LLM judge measuring framework alignment
- [ ] Configurable persona (beyond Samantha)
- [ ] Support for additional book corpora
- [ ] Push-to-talk mode as alternative to VAD
- [ ] Mobile-optimised web UI

---

## Disclaimer

**C.A.L.M. is not a replacement for professional mental health care.**

It is a portfolio project demonstrating local AI system design, retrieval pipelines, and voice interfaces. If you are experiencing a mental health crisis, please contact a qualified professional or the crisis resources listed above.

---

## Author

**Lakshya Soni**
[GitHub](https://github.com/Lakshyalive) · [LinkedIn](#) · [Hugging Face](#)

Assistant persona: **Samantha**
Project: **C.A.L.M. — Context-Aware Language Memory**