"""
test_blocking_latency.py
Measures the baseline (blocking) Time-to-Speech latency.
Calculates how long it takes to perform:
  1. RAG Retrieval
  2. Complete LLM response generation (non-streamed)
  3. Complete TTS audio synthesis (entire response)
This serves as the baseline comparison against the optimized 3-thread pipeline.
"""

import sys
import time
import numpy as np
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration matching main application
LLM_MODEL   = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
DB_DIR      = "./therapy_db"
VOICE       = "af_heart"
SPEED       = 0.92

def test_blocking():
    print("\n" + "="*60)
    print("  MEASURING BASELINE BLOCKING LATENCY")
    print("="*60)

    print("\n[1/4] Loading models and vector store...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="therapy_knowledge"
    )
    
    # Pre-warm connection to Ollama and Kokoro pipeline
    from kokoro import KPipeline
    tts = KPipeline(lang_code='en-us')
    
    # Warm up Ollama LLM
    ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "Hello"}])
    print("      Models ready.")

    query = "I feel like no matter how hard I try, nothing ever changes."
    print(f"\n[2/4] Simulating query: '{query}'")

    start_time = time.time()

    # Step 1: Retrieval
    retrieved = vectorstore.similarity_search_with_score(query, k=4)
    retrieval_time = time.time() - start_time
    print(f"      ➔ Retrieval finished: {retrieval_time:.2f}s")

    # Step 2: Blocking LLM Call
    context_str = "\n\n".join([doc.page_content for doc, _ in retrieved])
    system_prompt = (
        "You are a supportive, psychology-grounded voice companion. "
        "Keep your response extremely short (2-3 sentences max). "
        f"Ground your advice using this context:\n{context_str}"
    )
    
    llm_start = time.time()
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    reply_text = response['message']['content']
    llm_time = time.time() - llm_start
    print(f"      ➔ Full LLM response generated: {llm_time:.2f}s")
    print(f"      Response: \"{reply_text}\"")

    # Step 3: Blocking TTS Synthesis
    tts_start = time.time()
    chunks = []
    for _, _, audio in tts(reply_text, voice=VOICE, speed=SPEED):
        if audio is not None:
            chunks.append(audio)
    if chunks:
        _ = np.concatenate(chunks)
    tts_time = time.time() - tts_start
    print(f"      ➔ Full audio synthesis finished: {tts_time:.2f}s")

    total_latency = time.time() - start_time
    print("\n" + "="*60)
    print(f"  BASELINE BLOCKING TIME-TO-SPEECH: {total_latency:.2f} seconds")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_blocking()
