from flask import Flask, request, render_template
import requests
import pandas as pd
from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
import os
from flask import jsonify

# 🔐 Clés (sécurise-les avec Replit Secrets si possible)

AIRTABLE_API_KEY = os.environ['AIRTABLE_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
BASE_ID = os.environ['BASE_ID']
TABLE_NAME = "Festivals"

# Configuration
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.0-flash-lite")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Airtable → DataFrame
def fetch_data():
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }

    all_records = []
    offset = None

    while True:
        params = {"pageSize": 100}
        if offset:
            params["offset"] = offset
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        batch_records = data.get("records", [])
        if not batch_records:
            break
        all_records.extend(batch_records)
        offset = data.get("offset")
        if not offset:
            break

    records_data = []
    for record in all_records:
        fields = record.get("fields", {})
        if fields:
            record_data = {
                "id": record.get("id"),
                "createdTime": record.get("createdTime"),
                **fields
            }
            records_data.append(record_data)

    df = pd.DataFrame(records_data)
    return df.dropna(axis=1, how='all')

# Vectorisation & Indexation ChromaDB
df = fetch_data()
client = Client(Settings(persist_directory="./chroma_db"))
try:
    client.delete_collection("Festivals")
except:
    pass

collection = client.create_collection(name="Festivals", metadata={"hnsw:space": "ip"})
embeddings, metadatas, ids = [], [], []

for idx, row in df.iterrows():
    if row.dropna().empty:
        continue
    row_text = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
    embedding = model.encode(row_text)
    embedding = embedding / np.linalg.norm(embedding)
    embeddings.append(embedding.tolist())
    metadatas.append({k: str(v) for k, v in row.items() if pd.notna(v)})
    ids.append(f"fest_{idx}")

collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

# 🧠 Fonction IA
def rag_response(user_query: str):
    query_embedding = model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["metadatas", "distances"])

    context_parts = []
    for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        def get_val(key, default="Non spécifié"):
            val = metadata.get(key)
            return default if val in [None, "", "nan"] else val

        context_parts.append(f"""
        🎪 {get_val('Festival Name')} ({get_val('City')}, {get_val('Country')}) - {get_val('Dates')}
        🎶 Genre: {get_val('Genre')} | 💶 Prix: {get_val('Ticket Price (EUR)')}€
        🏨 Hébergement: {get_val('Accommodation option')} | Ambiance: {get_val('Atmosphere')}
        """)

    context = "\n".join(context_parts)

    prompt = f"""
Tu es un expert en festivals. L'utilisateur demande :
"{user_query}"

Voici les suggestions :
{context}

Génère une réponse concise avec emojis, ton enthousiaste, sans répétitions.
"""

    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Erreur lors de la génération: {e}"

# Flask App
app = Flask(__name__)

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query manquante"}), 400
    response_text = rag_response(user_query)
    return jsonify({"response": response_text})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000)
    
