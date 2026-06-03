import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# 1. Підключення до Pinecone та завантаження моделі
print("Підключення до Pinecone та завантаження моделі...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# 2. Функція кодування запиту
def encode_query(query_text):
    # нормалізуємо
    return model.encode(query_text, normalize_embeddings=True)

def print_results(header, results, df_reference=None):
    print(f"\n{'='*50}\n{header}\n{'='*50}")
    if not results:
        print("Нічого не знайдено.")
        return
        
    for i, match in enumerate(results['matches']):
        score = match['score']
        meta = match['metadata']
        
        # Підтягуємо повний абстракт з локального датасету (бо в Pinecone він обрізаний)
        arxiv_id = meta['arxiv_id']
        if df_reference is not None:
            full_abstract = df_reference.loc[df_reference['id'] == arxiv_id, 'abstract'].values[0]
        else:
            full_abstract = meta.get('abstract', '')
            
        snippet = full_abstract[:150] + "..." if len(full_abstract) > 150 else full_abstract
        
        print(f"{i+1}. [Score: {score:.4f}] {meta['title']}")
        print(f"   Категорія: {meta['category']} | Рік: {meta['year']}")
        print(f"   Абстракт: {snippet}\n")

# 3. Чистий семантичний пошук
query_1 = "teaching machines to recognize objects in pictures"
print(f"Виконуємо чистий пошук для запиту: '{query_1}'")
vec_1 = encode_query(query_1).tolist()
res_1 = index.query(vector=vec_1, top_k=TOP_K, include_metadata=True)
print_results("РЕЗУЛЬТАТИ: ЧИСТИЙ ПОШУК", res_1, df)

# 4. Пошук з фільтрацією
query_2 = "reinforcement learning"
print(f"Виконуємо пошук з фільтрами для запиту: '{query_2}'")
vec_2 = encode_query(query_2).tolist()

# Приклад A: Останні 5 років (>= 2021) та категорія cs.LG
filter_a = {
    "year": {"$gte": 2021},
    "category": {"$eq": "cs.LG"}
}
res_2a = index.query(vector=vec_2, top_k=TOP_K, include_metadata=True, filter=filter_a)
print_results("РЕЗУЛЬТАТИ: ФІЛЬТР A (>= 2021, cs.LG)", res_2a, df)

# Приклад B: Старі статті (< 2015), будь-яка категорія
filter_b = {
    "year": {"$lt": 2015}
}
res_2b = index.query(vector=vec_2, top_k=TOP_K, include_metadata=True, filter=filter_b)
print_results("РЕЗУЛЬТАТИ: ФІЛЬТР B (< 2015)", res_2b, df)

# 5. Порівняння локальних метрик
print("\n" + "="*50)
print("ПОРІВНЯННЯ ЛОКАЛЬНИХ МЕТРИК СХОЖОСТІ")
print("="*50)
print(" Завантаження локальних ембедингів...")
local_embeddings = np.load("embeddings/embeddings.npy")
q_vec_np = encode_query(query_1)

# Обчислення Dot Product
dot_scores = np.dot(local_embeddings, q_vec_np)

# Обчислення Cosine Similarity
norms_local = np.linalg.norm(local_embeddings, axis=1)
norm_q = np.linalg.norm(q_vec_np)
cos_scores = dot_scores / (norms_local * norm_q)

# Обчислення L2 Distance
l2_distances = np.linalg.norm(local_embeddings - q_vec_np, axis=1)

# Отримання топ-5 індексів для кожної метрики
top_dot_idx = np.argsort(dot_scores)[::-1][:TOP_K]
top_cos_idx = np.argsort(cos_scores)[::-1][:TOP_K]
top_l2_idx = np.argsort(l2_distances)[:TOP_K] # Для L2 менша відстань = краще

print("\nТоп-5 індексів документів (Dot Product):   ", top_dot_idx)
print("Топ-5 індексів документів (Cosine Sim):    ", top_cos_idx)
print("Топ-5 індексів документів (L2 Distance):   ", top_l2_idx)

if np.array_equal(top_dot_idx, top_cos_idx) and np.array_equal(top_cos_idx, top_l2_idx):
    print("\nВисновок коду: Усі три метрики повернули абсолютно ідентичні результати (однакові індекси).")