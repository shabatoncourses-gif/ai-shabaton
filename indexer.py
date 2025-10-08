import os, openai, chromadb

openai.api_key = "YOUR_OPENAI_API_KEY"
CHROMA_PATH = "data/index"
os.makedirs(CHROMA_PATH, exist_ok=True)
client = chromadb.Client(chromadb.config.Settings(persist_directory=CHROMA_PATH))
collection = client.get_or_create_collection("shabaton")

def embed_text(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response['data'][0]['embedding']

for fname in os.listdir("data/pages"):
    with open(f"data/pages/{fname}", "r", encoding="utf-8") as f:
        text = f.read()
    print("📘 מאנדקס:", fname)
    emb = embed_text(text[:8000])
    collection.add(
        documents=[text],
        embeddings=[emb],
        ids=[fname]
    )

client.persist()
print("✅ אינדוקס הושלם.")
