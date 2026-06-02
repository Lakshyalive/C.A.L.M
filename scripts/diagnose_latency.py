"""
diagnose_latency.py
Diagnoses exact timing for each stage of the conversational pipeline:
  1. ChromaDB retrieval (with query embedding)
  2. LLM Time-to-First-Token (TTFT)
  3. LLM Time-to-Sentence-Boundary
  4. TTS first-sentence synthesis time
"""

import sys
import time
import numpy as np
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

LLM_MODEL   = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
DB_DIR      = "./therapy_db"
VOICE       = "af_heart"
SPEED       = 0.92

def diagnose():
    print("\n" + "="*60)
    print("  DIAGNOSING SYSTEM LATENCY PIPELINE")
    print("="*60)

    # 1. Warm-up
    print("\n[1/5] Loading models and vector store...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="therapy_knowledge"
    )
    from kokoro import KPipeline
    tts = KPipeline(lang_code='en-us')
    
    # Pre-warm LLM
    ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "Say OK"}])
    print("      Warmup complete.")

    query = "I feel like no matter how hard I try, nothing ever changes."
    print(f"\nTesting Query: '{query}'")

    # 2. Measure Retrieval In Isolation
    print("\n[2/5] Testing ChromaDB Retrieval...")
    t0 = time.time()
    retrieved = vectorstore.similarity_search_with_score(query, k=4)
    retrieval_time = time.time() - t0
    print(f"      ➔ Retrieval Time: {retrieval_time:.3f}s (nomic-embed-text query + search)")

    # Format Context
    context_str = "\n\n".join([doc.page_content for doc, _ in retrieved])
    system_prompt = (
        "You are Samantha, a supportive voice companion. Ground your advice in this context. "
        "Keep your response extremely short (2 sentences max).\n"
        f"Context:\n{context_str}"
    )

    # 3. Measure LLM Latency (First Token & First Sentence)
    print("\n[3/5] Testing LLM Generation Latency...")
    t0 = time.time()
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        stream=True
    )
    
    first_token_time = None
    first_sentence_time = None
    sentence = ""
    sentence_ends = {'.', '!', '?', '…'}
    
    llm_start = time.time()
    for chunk in stream:
        token = chunk['message']['content']
        if first_token_time is None:
            first_token_time = time.time() - llm_start
        sentence += token
        if sentence.rstrip() and sentence.rstrip()[-1] in sentence_ends:
            if first_sentence_time is None:
                first_sentence_time = time.time() - llm_start
                break

    print(f"      ➔ Time-to-First-Token (TTFT): {first_token_time:.3f}s")
    print(f"      ➔ Time-to-First-Sentence: {first_sentence_time:.3f}s")
    print(f"      First Sentence: \"{sentence.strip()}\"")

    # 4. Measure TTS Synthesis Latency
    print("\n[4/5] Testing Voice Synthesis Latency...")
    t0 = time.time()
    chunks = []
    for _, _, audio in tts(sentence.strip(), voice=VOICE, speed=SPEED):
        if audio is not None:
            chunks.append(audio)
    synthesis_time = time.time() - t0
    print(f"      ➔ TTS Synthesis Time: {synthesis_time:.3f}s (for first sentence)")

    # 5. Summary
    total_time = retrieval_time + first_sentence_time + synthesis_time
    print("\n" + "="*60)
    print("  LATENCY BUDGET SUMMARY")
    print("="*60)
    print(f"  Chroma Retrieval:   {retrieval_time:5.2f}s  ({(retrieval_time/total_time)*100:4.1f}%)")
    print(f"  LLM First Sentence: {first_sentence_time:5.2f}s  ({(first_sentence_time/total_time)*100:4.1f}%)")
    print(f"  TTS Synthesis:      {synthesis_time:5.2f}s  ({(synthesis_time/total_time)*100:4.1f}%)")
    print("-"*60)
    print(f"  TOTAL TTS LATENCY:  {total_time:5.2f} seconds")
    print("="*60 + "\n")

if __name__ == "__main__":
    diagnose()
