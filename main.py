from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai, chromadb

openai.api_key = "YOUR_OPENAI_API_KEY"
client = chromadb.Client(chromadb.config.Settings(persist_directory="data/index"))
collection = client.get_collection("shabaton")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query(req: Request):
    data = await req.json()
    user_q = data.get("query", "")
    emb = openai.Embedding.create(input=user_q, model="text-embedding-3-small")["data"][0]["embedding"]
    results = collection.query(query_embeddings=[emb], n_results=3)
    docs = "\n".join(results["documents"][0])
    prompt = f"""
    אתה סוכן שירות רשמי של אתר שבתון.
    ענה רק בעברית, בצורה מקצועית, עניינית ומבוססת על המידע הבא מהאתר:
    {docs}

    שאלה: {user_q}
    תשובה:
    """
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    answer = completion.choices[0].message.content.strip()
    return {"answer": answer}
