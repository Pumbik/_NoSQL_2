import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
print("Підключення до Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Створення індексу '{INDEX_NAME}' (це може зайняти хвилину)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",  # Наша метрика з попереднього кроку
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Стандартний регіон для безкоштовного тарифу
        )
    )
else:
    print(f"Індекс '{INDEX_NAME}' вже існує.")

# Підключаємось до конкретного індексу
index = pc.Index(INDEX_NAME)

# Завантажуємо дані
print("Читаємо датасет та ембединги...")
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

assert len(df) == len(embeddings), "Помилка: кількість записів у parquet та npy не збігається!"

# Підготовка та завантаження батчами
print(f"Починаємо завантаження {len(df)} векторів у Pinecone...")
vectors_to_upsert = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Завантаження батчів"):
    # 1. Обрізаємо текстові поля згідно з вимогами
    abstract = str(row.get("abstract", ""))[:500]
    authors = str(row.get("authors", ""))[:200]
    
    # 2. Формуємо метадані
    metadata = {
        "arxiv_id": str(row["id"]),
        "title": str(row["title"]),
        "abstract": abstract,
        "authors": authors,
        "year": int(row["year"]),
        "category": str(row["category"])
    }
    
    # 3. Унікальний ID
    vector_id = f"paper_{i}"
    
    # 4. Вектор (Pinecone вимагає стандартний список Python, а не numpy array)
    vector_values = embeddings[i].tolist()
    
    # Додаємо об'єкт у поточний батч
    vectors_to_upsert.append({
        "id": vector_id,
        "values": vector_values,
        "metadata": metadata
    })
    
    # Якщо назбирали BATCH_SIZE елементів — відправляємо їх
    if len(vectors_to_upsert) >= BATCH_SIZE:
        index.upsert(vectors=vectors_to_upsert)
        vectors_to_upsert = []  # Очищаємо батч

# Відправляємо залишки (якщо остання група менша за 200)
if vectors_to_upsert:
    index.upsert(vectors=vectors_to_upsert)

stats = index.describe_index_stats()
print("\n Успішно завершено!")
print(f"Загальна кількість векторів у індексі '{INDEX_NAME}': {stats.total_vector_count}")