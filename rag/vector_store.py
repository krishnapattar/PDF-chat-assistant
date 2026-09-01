import chromadb
from rag.embedder import get_embedding

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")

def add_chunks(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

def search(query, n_results=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]

def reset_collection():
    global collection
    client.delete_collection("pdf_chunks")
    collection = client.get_or_create_collection(name="pdf_chunks")

if __name__ == "__main__":
    from rag.loader import load_pdf_text
    from rag.chunker import chunk_text

    text = load_pdf_text("data/sample.pdf")
    chunks = chunk_text(text)
    add_chunks(chunks)
    print(f"Added {len(chunks)} chunks to the vector store.")

    results = search("What is this document about?")
    print("Top matching chunks:")
    for r in results:
        print("-", r[:100])