import os
from dotenv import load_dotenv
from openai import OpenAI
from rag.vector_store import search

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

def answer_question(question):
    chunks = search(question, n_results=3)
    context = "\n\n".join(chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
"""

    response = client.chat.completions.create(
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    question = "What is pre-training in the context of LLMs?"
    print(answer_question(question))