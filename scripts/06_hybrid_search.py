import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати
RRF_K = 60   # стандартна константа для RRF

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# 1. Побудова локального індексу BM25
print("Побудова локального BM25 індексу (title + abstract)...")
corpus_tokens = []
for _, row in df.iterrows():
    # Проста токенізація: lowercase та розбиття по пробілах
    text = str(row['title']) + " " + str(row['abstract'])
    corpus_tokens.append(text.lower().split())

bm25 = BM25Okapi(corpus_tokens)

# Функція 1: BM25 Пошук
def search_bm25(query, top_k=TOP_K):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_n_idx = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_n_idx):
        if scores[idx] > 0: # Відсікаємо нульові збіги
            results.append({
                'id': str(df.iloc[idx]['id']),
                'title': str(df.iloc[idx]['title']),
                'score': scores[idx],
                'rank': rank + 1
            })
    return results

# Функція 2: Векторний пошук (Pinecone)
def search_vector(query, top_k=TOP_K):
    vec = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    
    results = []
    for rank, match in enumerate(res['matches']):
        results.append({
            'id': match['metadata']['arxiv_id'],
            'title': match['metadata']['title'],
            'score': match['score'],
            'rank': rank + 1
        })
    return results

# Функція 3: Гібридний пошук з RRF
def search_hybrid(query, top_k=5):
    # Отримуємо ширші пули кандидатів
    bm25_results = search_bm25(query, top_k=TOP_K)
    vector_results = search_vector(query, top_k=TOP_K)
    
    rrf_scores = {}
    
    # Додаємо бали від BM25
    for item in bm25_results:
        doc_id = item['id']
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'title': item['title'], 'rrf_score': 0.0}
        rrf_scores[doc_id]['rrf_score'] += 1.0 / (RRF_K + item['rank'])
        
    # Додаємо бали від векторного пошуку
    for item in vector_results:
        doc_id = item['id']
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'title': item['title'], 'rrf_score': 0.0}
        rrf_scores[doc_id]['rrf_score'] += 1.0 / (RRF_K + item['rank'])
        
    # Сортуємо за RRF score
    sorted_hybrid = sorted(rrf_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
    
    return [{'id': k, 'title': v['title'], 'score': v['rrf_score']} for k, v in sorted_hybrid[:top_k]]

# Допоміжна функція для виводу
def print_top5(method_name, results):
    print(f"\n--- {method_name} ---")
    if not results:
        print("Нічого не знайдено.")
    for i, res in enumerate(results[:5]):
        print(f"{i+1}. [Score: {res['score']:.4f}] {res['title']}")

# Демонстрація запитів
queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

print("\n" + "="*60)
print("ПОЧАТОК ГІБРИДНОГО ПОШУКУ")
print("="*60)

for q in queries:
    print(f"\n\n🔍 ЗАПИТ: '{q}'")
    print("-" * 60)
    
    bm25_res = search_bm25(q)
    vec_res = search_vector(q)
    hybrid_res = search_hybrid(q)
    
    print_top5("BM25 (Лексичний)", bm25_res)
    print_top5("Pinecone (Векторний)", vec_res)
    print_top5(f"Hybrid (RRF, k={RRF_K})", hybrid_res)