import pandas as pd
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "gemma:2b"

def load_dataset():
    return pd.read_excel("dataset.xlsx")

def create_chunks(df):
    chunks = []
    for _, row in df.iterrows():
        chunk = f"Name: {row['NAME']}. Age: {row['AGE']}. Qualification: {row['QUALIFICATION']}. Skills: {row['SKILLS']}. Experience: {row['EXPERIENCE']}"
        chunks.append(chunk)
    return chunks

VECTOR_DB = []

def add_chunk_to_database(chunk):
    response = ollama.embed(EMBEDDING_MODEL, chunk)
    if 'embedding' in response:
        embedding = response['embedding']
    elif 'embeddings' in response:
        embedding = response['embeddings'][0]
    elif 'data' in response and 'embedding' in response['data'][0]:
        embedding = response['data'][0]['embedding']
    else:
        raise ValueError(f"Unexpected embedding response format: {response}")
    VECTOR_DB.append((chunk, embedding))

df = load_dataset()
chunks = create_chunks(df)
for chunk in chunks:
    add_chunk_to_database(chunk)


def get_top_k_chunks(query, k=3):
    response = ollama.embed(EMBEDDING_MODEL, query)
    if 'embedding' in response:
        query_embedding = response['embedding']
    elif 'embeddings' in response:
        query_embedding = response['embeddings'][0]
    elif 'data' in response and 'embedding' in response['data'][0]:
        query_embedding = response['data'][0]['embedding']
    else:
        raise ValueError(f"Unexpected embedding response format: {response}")

    all_embeddings = [emb for _, emb in VECTOR_DB]
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]

    top_k_indices = similarities.argsort()[-k:][::-1]
    top_chunks = [VECTOR_DB[i][0] for i in top_k_indices]
    return top_chunks


