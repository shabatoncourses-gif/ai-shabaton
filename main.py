# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import chromadb


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
TOP_K = int(os.getenv('TOP_K', '4'))
CHROMA_DIR = os.getenv('CHROMA_DB_DIR', './data/index')
BACKEND_ORIGIN = os.getenv('BACKEND_ORIGIN', '*')


if not OPENAI_API_KEY:
raise RuntimeError('OPENAI_API_KEY missing')


openai.api_key = OPENAI_API_KEY
client = chromadb.Client(chromadb.config.Settings(persist_directory=CHROMA_DIR))
collection = client.get_collection('shabaton_faq')


app = FastAPI()
app.add_middleware(
CORSMiddleware,
allow_origins=[BACKEND_ORIGIN] if BACKEND_ORIGIN != '*' else ['*'],
allow_methods=['*'],
allow_headers=['*']
)


class QueryIn(BaseModel):
query: str




def build_prompt(question, contexts):
header = (
'אתה סוכן שירות רשמי עבור אתר Shabaton.online. ענה בעברית בלבד, בסגנון רשמי ומקצועי. '
'הסתמך אך ורק על המידע המסופק בקטעים שלמטה. אם לא קיים מידע מתאים — אמור בבירור שאין מידע זמין והפנה לאמצעי יצירת קשר באתר.\n\n'
)
ctx_texts = []
for i, c in enumerate(contexts):
s = f"[מקור {i+1}] {c.get('source','')}\n{c.get('content','')}\n"
ctx_texts.append(s)
prompt = header + '\n\n'.join(ctx_texts) + f"\n\nשאלה: {question}\n\nתשובה (עברית, מקצועית, קצרה, ציין את מספרי המקורות במידת הצורך):"
return prompt


@app.post('/query')
async def query(q: QueryIn):
qtext = q.query.strip()
if not qtext:
raise HTTPException(400, 'empty query')
# embeddings via collection.query (Chroma handles embedding fn internally)
res = collection.query(qtext, n_results=TOP_K, include=['documents','metadatas','ids'])
docs = []
for doc, meta in zip(res.get('documents',[]), res.get('metadatas',[])):
docs.append({'content': doc, 'source': meta.get('source','')})
prompt = build_prompt(qtext, docs)
try:
completion = openai.ChatCompletion.create(
model=LLM_MODEL,
messages=[{'role':'system','content':'You are a helpful assistant. Answer in Hebrew only.'},
{'role':'user','content': prompt}],
temperature=0.2,
max_tokens=700
)
answer = completion.choices[0].message.content.strip()
except Exception as e:
raise HTTPException(500, f"LLM error: {e}")
sources = list({d['source'] for d in docs if d.get('source')})
return {'status':'ok'}
