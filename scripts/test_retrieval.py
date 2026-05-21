"""
test_retrieval.py
Quick test to verify your vector database retrieval works correctly.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings  = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory  = "./therapy_db",
    embedding_function = embeddings,
    collection_name    = "therapy_knowledge"
)

test_queries = [
    "I feel so worthless and like nobody loves me",
    "I can't stop thinking about a traumatic event",
    "What is the meaning of life when everything feels pointless",
    "I feel ashamed of who I am",
    "My thoughts are spiraling and I can't control them"
]

print("\nRETRIEVAL TEST\n" + "="*50)

for query in test_queries:
    print(f"\nQUERY: {query}")
    results = vectorstore.similarity_search_with_score(query, k=2)
    for doc, score in results:
        print(f"  [{doc.metadata['framework']}] {doc.metadata['author']}")
        print(f"  Score: {score:.3f}")
        print(f"  Preview: {doc.page_content[:120]}...")
