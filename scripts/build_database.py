"""
build_database.py
Reads 5 therapy books, chunks them, embeds them, saves to ChromaDB.
Run this once. Takes about 5-10 minutes.
"""

import os
import sys
import time
import pymupdf  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Book Registry ─────────────────────────────────────────────────────────────
# Each book has a filename, author, framework label, and short description.
# This metadata is stored with every chunk for transparent retrieval.

BOOKS = [
    {
        "file":        "rogers.pdf",
        "author":      "Carl Rogers",
        "title":       "On Becoming a Person",
        "framework":   "HUMANISTIC",
        "approach":    "Client-centered empathy, unconditional positive regard, "
                       "reflective listening, non-directive support"
    },
    {
        "file":        "burns_cbt.pdf",
        "author":      "David D. Burns",
        "title":       "Feeling Good: The New Mood Therapy",
        "framework":   "CBT",
        "approach":    "Cognitive distortions, thought records, behavioral "
                       "activation, structured reframing"
    },
    {
        "file":        "frankl.pdf",
        "author":      "Viktor Frankl",
        "title":       "Man's Search for Meaning",
        "framework":   "EXISTENTIAL",
        "approach":    "Logotherapy, meaning-making, suffering as growth, "
                       "existential purpose"
    },
    {
        "file":        "brown.pdf",
        "author":      "Brene Brown",
        "title":       "The Gifts of Imperfection",
        "framework":   "RESILIENCE",
        "approach":    "Shame resilience, vulnerability, self-worth, "
                       "belonging, wholehearted living"
    },
    {
        "file":        "van_der_kolk.pdf",
        "author":      "Bessel van der Kolk",
        "title":       "The Body Keeps the Score",
        "framework":   "TRAUMA",
        "approach":    "Trauma-informed care, somatic awareness, nervous "
                       "system regulation, grounding techniques"
    },
]

BOOKS_DIR   = "./books"
DB_DIR      = "./therapy_db"
EMBED_MODEL = "nomic-embed-text"

# ── Text Splitter ─────────────────────────────────────────────────────────────
# chunk_size=600: Large enough for meaningful context, small enough for precision
# chunk_overlap=80: Prevents cutting ideas mid-sentence at boundaries

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 600,
    chunk_overlap = 80,
    separators    = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)

# ── Main Processing ───────────────────────────────────────────────────────────

def load_and_chunk_book(book_info):
    """Load one PDF and return tagged LangChain Document chunks."""
    filepath = os.path.join(BOOKS_DIR, book_info["file"])

    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found. Skipping.")
        return []

    print(f"  Loading: {book_info['title']} by {book_info['author']}...")

    # Open PDF with PyMuPDF
    pdf = pymupdf.open(filepath)
    full_text = ""
    for page in pdf:
        full_text += page.get_text()
    pdf.close()

    # Remove junk: multiple blank lines, page numbers, headers
    import re
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'^\d+\s*$', '', full_text, flags=re.MULTILINE)

    # Split into chunks
    raw_chunks = splitter.split_text(full_text)

    # Wrap each chunk as a LangChain Document with metadata
    documents = []
    for i, chunk in enumerate(raw_chunks):
        # Skip chunks that are too short to be useful
        if len(chunk.strip()) < 100:
            continue

        documents.append(Document(
            page_content = chunk.strip(),
            metadata     = {
                "author":    book_info["author"],
                "title":     book_info["title"],
                "framework": book_info["framework"],
                "approach":  book_info["approach"],
                "chunk_id":  i,
                "source":    book_info["file"]
            }
        ))

    print(f"  Done: {len(documents)} chunks created")
    return documents


def build_database():
    print("\n" + "="*60)
    print("  THERAPY RAG — Building Knowledge Database")
    print("="*60)
    print(f"\nEmbedding model: {EMBED_MODEL}")
    print(f"Output directory: {DB_DIR}\n")

    # Process all books
    all_documents = []
    for book in BOOKS:
        chunks = load_and_chunk_book(book)
        all_documents.extend(chunks)

    if not all_documents:
        print("ERROR: No documents loaded. Check your books/ folder.")
        sys.exit(1)

    print(f"\nTotal chunks across all books: {len(all_documents)}")
    print("\nBreakdown by framework:")

    framework_counts = {}
    for doc in all_documents:
        fw = doc.metadata["framework"]
        framework_counts[fw] = framework_counts.get(fw, 0) + 1
    for fw, count in framework_counts.items():
        print(f"  {fw}: {count} chunks")

    # Build vector store
    print(f"\nEmbedding {len(all_documents)} chunks with {EMBED_MODEL}...")
    print("This takes 5-10 minutes. Do not close the terminal.\n")

    start = time.time()

    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents         = all_documents,
        embedding         = embeddings,
        persist_directory = DB_DIR,
        collection_name   = "therapy_knowledge"
    )

    elapsed = time.time() - start
    print(f"\nDatabase built in {elapsed:.0f} seconds")
    print(f"Saved to: {DB_DIR}")
    print("\nAll done. You can now run therapy_voice.py")


if __name__ == "__main__":
    build_database()
