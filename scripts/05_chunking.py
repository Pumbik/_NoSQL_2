import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
BATCH_SIZE = 100

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# 1. Вибрати 30 статей із найдовшими анотаціями
print("Шукаємо 30 статей з найдовшими анотаціями...")
df['word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
top_30_df = df.nlargest(30, 'word_count').copy()
print(f"Знайдено. Найдовша анотація має {top_30_df['word_count'].max()} слів.")

# 2. Функції для розбиття на чанки
def get_fixed_chunks(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def get_semantic_chunks(text, max_words=50):
    # Розбиваємо по крапці, знаку питання або оклику, за якими йде пробіл
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    curr_chunk = []
    curr_len = 0
    for s in sentences:
        words = s.split()
        if curr_len + len(words) > max_words and curr_chunk:
            chunks.append(" ".join(curr_chunk))
            curr_chunk = words
            curr_len = len(words)
        else:
            curr_chunk.extend(words)
            curr_len += len(words)
    if curr_chunk:
        chunks.append(" ".join(curr_chunk))
    return chunks

# 3. Створення індексів
def create_index_if_not_exists(index_name):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Створення індексу '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Індекс '{index_name}' вже існує.")

create_index_if_not_exists(INDEX_FIXED)
create_index_if_not_exists(INDEX_SEMANTIC)

# 4. Підготовка даних (генерація чанків)
fixed_records = []
semantic_records = []

print("Нарізаємо тексти на чанки...")
for _, row in top_30_df.iterrows():
    text = row['abstract']
    arxiv_id = str(row['id'])
    title = str(row['title'])
    year = int(row['year'])
    category = str(row['category'])
    
    # Fixed chunks
    f_chunks = get_fixed_chunks(text)
    for i, chunk_text in enumerate(f_chunks):
        fixed_records.append({
            "id": f"{arxiv_id}_f_{i}",
            "text": chunk_text,
            "meta": {"arxiv_id": arxiv_id, "title": title, "chunk_text": chunk_text, "chunk_idx": i, "year": year, "category": category}
        })
        
    # Semantic chunks
    s_chunks = get_semantic_chunks(text)
    for i, chunk_text in enumerate(s_chunks):
        semantic_records.append({
            "id": f"{arxiv_id}_s_{i}",
            "text": chunk_text,
            "meta": {"arxiv_id": arxiv_id, "title": title, "chunk_text": chunk_text, "chunk_idx": i, "year": year, "category": category}
        })

print(f"Згенеровано {len(fixed_records)} фіксованих та {len(semantic_records)} семантичних чанків.")

# Функція для кодування та завантаження
def embed_and_upload(records, index_name):
    print(f"\n Обробка та завантаження в '{index_name}'...")
    idx = pc.Index(index_name)
    
    for i in tqdm(range(0, len(records), BATCH_SIZE)):
        batch = records[i:i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        
        # Кодування батчу
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        
        vectors_to_upsert = []
        for j, item in enumerate(batch):
            vectors_to_upsert.append({
                "id": item["id"],
                "values": embeddings[j],
                "metadata": item["meta"]
            })
        idx.upsert(vectors=vectors_to_upsert)

embed_and_upload(fixed_records, INDEX_FIXED)
embed_and_upload(semantic_records, INDEX_SEMANTIC)

# 6. Функція пошуку по чанках
def search_chunks(query, index_name):
    idx = pc.Index(index_name)
    vec = model.encode(query, normalize_embeddings=True).tolist()
    res = idx.query(vector=vec, top_k=5, include_metadata=True)
    
    print(f"\n--- Результати для '{index_name}' ---")
    for i, match in enumerate(res['matches']):
        score = match['score']
        meta = match['metadata']
        print(f"{i+1}. [Score: {score:.4f}] {meta['title']}")
        print(f"   Чанк #{int(meta['chunk_idx'])}: {meta['chunk_text']}\n")

print("\n" + "="*50)
print("ТЕСТУВАННЯ ПОШУКУ ПО ЧАНКАХ")
print("="*50)
test_query = "quantum field theory and black hole thermodynamics"
print(f"Запит: '{test_query}'")

search_chunks(test_query, INDEX_FIXED)
search_chunks(test_query, INDEX_SEMANTIC)