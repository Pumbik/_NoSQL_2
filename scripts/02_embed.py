# scripts/02_embed.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"

# 1. Створюємо директорію для ембедингів, якщо її немає
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Починаємо процес генерації ембедингів...")

# 2. Завантаження датасету
print(f"Завантажуємо дані з {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)

# 3. Підготовка текстів для кодування
# Формат: title + " [SEP] " + abstract
print("Форматуємо тексти згідно вимог моделі SPECTER2...")
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

# 4. Завантаження моделі
print(f"Завантажуємо модель {MODEL_NAME} (це може зайняти час при першому запуску)...")
model = SentenceTransformer(MODEL_NAME)

# 5. Генерація ембедингів
print(f"Кодуємо {len(texts)} текстів у вектори...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# 6. Вивід статистики
print("\n Статистика результатів:")
print(f"Загальна кількість оброблених текстів: {len(texts)}")
print(f"Розмірність ембедингів: {embeddings.shape[1]} (очікується 768)")
print(f"Норма першого ембедингу: {np.linalg.norm(embeddings[0]):.4f} (повинна бути ~1.0)")

# 7. Збереження результатів у форматі NumPy
print(f"Зберігаємо ембединги у {OUTPUT_FILE}...")
np.save(OUTPUT_FILE, embeddings)

print("🎉 Готово! Ембединги успішно згенеровані та збережені.")