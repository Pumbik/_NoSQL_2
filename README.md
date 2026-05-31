# Semantic Search for Scientific Papers (arXiv)

## Вимоги до проєкту
* **.env і папки `data/`, `embeddings/` додані в `.gitignore`:** 

---

## Частина 1: Підготовка даних і вибір інструментів

### 1.1 Завантаження та очищення датасету
Для підготовки даних було створено скрипт `scripts/01_prepare_data.py`. Скрипт витягує рік публікації (замість дати останнього оновлення), формує читабельний список авторів та бере лише першу (основну) категорію статті. Збереження відбувається у оптимізований формат Parquet.

**Вивід скрипту `01_prepare_data.py`:**
```text
Читаємо датасет: 10001it [00:00, 31102.77it/s]

Завантажено статей: 10000

Розподіл за категоріями (топ-10):
category
astro-ph              1838
hep-th                 680
hep-ph                 671
quant-ph               564
gr-qc                  350
cond-mat.mes-hall      307
cond-mat.str-el        292
cond-mat.mtrl-sci      291
cond-mat.stat-mech     271
math.AG                209
Name: count, dtype: int64

Розподіл за роками:
year
2007    10000
Name: count, dtype: int64

Приклад запису:
{'id': '0704.0001', 'title': 'Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies', 'abstract': 'A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified in which the calculation is most reliable. Good agreement is\ndemonstrated with data from the Fermilab Tevatron, and predictions are made for\nmore detailed tests with CDF and DO data. Predictions are shown for\ndistributions of diphoton pairs produced at the energy of the Large Hadron\nCollider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\nboson are contrasted with those produced from QCD processes at the LHC, showing\nthat enhanced sensitivity to the signal can be obtained with judicious\nselection of events.', 'authors': 'BalázsC., BergerE. L., NadolskyP. M., YuanC. -P.', 'year': 2007, 'category': 'hep-ph'}

Збережено в data/arxiv_subset.parquet



1.2 Відповіді на теоретичні запитання

1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

Pinecone, Qdrant та Chroma — це три популярні векторні бази даних, але вони мають суттєво різні моделі розгортання та філософію. Pinecone — це виключно закрита комерційна система, яка надається як повністю керована хмарна послуга (SaaS). Ви не можете завантажити її код або розгорнути на власних серверах, але натомість отримуєте "out-of-the-box" високу доступність, автоматичне масштабування та оптимізацію продуктивності без потреби в адмініструванні інфраструктури. Це ідеальний вибір для Enterprise-рішень та команд, які хочуть швидко інтегрувати векторний пошук, не витрачаючи час на DevOps. Chroma, навпаки, є повністю open-source (ліцензія Apache 2.0) та орієнтована на розробників AI-застосунків (зокрема з використанням LangChain/LlamaIndex). Вона працює локально (in-memory або з локальним збереженням), надзвичайно проста у налаштуванні для створення прототипів, але менш підходить для надвисоких навантажень або мільярдів векторів порівняно зі спеціалізованими кластерами. Qdrant — це open-source база, написана на Rust (ліцензія Apache 2.0), що гарантує високу продуктивність та безпеку роботи з пам'яттю. Qdrant пропонує як локальне/self-hosted розгортання, так і власну хмарну SaaS-версію. Її головна перевага полягає в потужній підтримці фільтрації (payload-based filtering) на рівні HNSW-графа.

Сценарії: Pinecone я б обрав для production-рішення у компанії, де немає виділеної команди інфраструктури, і важливий швидкий time-to-market. Chroma — для локальної розробки, прототипування RAG-додатків та хакатонів. Qdrant — для високонавантажених систем, де потрібен повний контроль над даними (self-hosted розгортання на власних серверах) або де є складні вимоги до фільтрації за метаданими.

2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

Універсальні моделі, такі як all-MiniLM-L6-v2, чудово справляються із завданнями семантичної близькості на загальновживаному словнику (наприклад, новини, відгуки, запитання-відповіді). Однак наукові тексти мають специфічну лексику, структуру (заголовок + абстракт) та цитування. Модель allenai/specter2_base є спеціалізованою і була донавчена (fine-tuned) саме для розуміння наукових публікацій. У картці моделі на Hugging Face вказано: "SPECTER2 is a base model for scientific tasks. It can be used directly for tasks like citation prediction, document clustering, and recommendation, and can be fine-tuned on task-specific data for document classification, retrieval, etc." Тобто архітектура моделі розроблена так, щоб генерувати ембединги, які враховують не лише поверхневий зміст слів, а й глибоку семантичну спорідненість між статтями, що посилаються одна на одну або досліджують спільні проблеми, що критично важливо для нашого пошукового рушія.

3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

У картці моделі allenai/specter2_base на Hugging Face зазначається, що ембединги цієї моделі не є нормалізованими (L2-normalized) за замовчуванням. Однак для завдань пошуку (Information Retrieval) та визначення схожості документів дослідники зазвичай використовують косинусну схожість (Cosine Similarity). Згідно з документацією SentenceTransformers для моделей типу SPECTER: "By default, input text longer than 512 word pieces is truncated. The model output is NOT normalized." Відповідно, при розрахунку близькості між векторами, або при створенні індексу у векторній базі даних, необхідно вказати саме метрику cosine (а не dot_product чи euclidean). Це критично важливо при створенні індексу в Pinecone: якщо обрати неправильну метрику (наприклад, евклідову відстань), база буде ранжувати документи некоректно, оскільки ненормалізовані вектори різної довжини (магнітуди) можуть мати велику геометричну відстань, хоча їхній кут напрямку (семантична схожість) буде дуже близьким. Метрика cosine нівелює різницю в довжині векторів і фокусується виключно на їхньому напрямку.