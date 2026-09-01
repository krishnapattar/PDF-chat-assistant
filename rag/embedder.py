from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text).tolist()

if __name__ == "__main__":
    vector = get_embedding("The refund policy allows returns within 30 days.")
    print(f"Vector length: {len(vector)}")
    print(f"First 5 numbers: {vector[:5]}")