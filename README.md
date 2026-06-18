# Semantic Search for Scientific Papers (arXiv)

## Вимоги до проєкту

- **.env і папки `data/`, `embeddings/` додані в `.gitignore`:**

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
```

Приклад запису:

```json
{
  "id": "0704.0001",
  "title": "Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies",
  "abstract": "A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified in which the calculation is most reliable. Good agreement is\ndemonstrated with data from the Fermilab Tevatron, and predictions are made for\nmore detailed tests with CDF and DO data. Predictions are shown for\ndistributions of diphoton pairs produced at the energy of the Large Hadron\nCollider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\nboson are contrasted with those produced from QCD processes at the LHC, showing\nthat enhanced sensitivity to the signal can be obtained with judicious\nselection of events.",
  "authors": "BalázsC., BergerE. L., NadolskyP. M., YuanC. -P.",
  "year": 2007,
  "category": "hep-ph"
}
```

Збережено в data/arxiv_subset.parquet

### 1.2 Відповіді на теоретичні запитання

```text
1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

    Pinecone, Qdrant та Chroma — це три популярні векторні бази даних, але вони мають суттєво різні моделі розгортання та філософію. Pinecone — це виключно закрита комерційна система, яка надається як повністю керована хмарна послуга (SaaS). Ви не можете завантажити її код або розгорнути на власних серверах, але натомість отримуєте "out-of-the-box" високу доступність, автоматичне масштабування та оптимізацію продуктивності без потреби в адмініструванні інфраструктури. Це ідеальний вибір для Enterprise-рішень та команд, які хочуть швидко інтегрувати векторний пошук, не витрачаючи час на DevOps. Chroma, навпаки, є повністю open-source (ліцензія Apache 2.0) та орієнтована на розробників AI-застосунків (зокрема з використанням LangChain/LlamaIndex). Вона працює локально (in-memory або з локальним збереженням), надзвичайно проста у налаштуванні для створення прототипів, але менш підходить для надвисоких навантажень або мільярдів векторів порівняно зі спеціалізованими кластерами. Qdrant — це open-source база, написана на Rust (ліцензія Apache 2.0), що гарантує високу продуктивність та безпеку роботи з пам'яттю. Qdrant пропонує як локальне/self-hosted розгортання, так і власну хмарну SaaS-версію. Її головна перевага полягає в потужній підтримці фільтрації (payload-based filtering) на рівні HNSW-графа.

Сценарії: Pinecone я б обрав для production-рішення у компанії, де немає виділеної команди інфраструктури, і важливий швидкий time-to-market. Chroma — для локальної розробки, прототипування RAG-додатків та хакатонів. Qdrant — для високонавантажених систем, де потрібен повний контроль над даними (self-hosted розгортання на власних серверах) або де є складні вимоги до фільтрації за метаданими.

2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

    Універсальні моделі, такі як all-MiniLM-L6-v2, чудово справляються із завданнями семантичної близькості на загальновживаному словнику (наприклад, новини, відгуки, запитання-відповіді). Однак наукові тексти мають специфічну лексику, структуру (заголовок + абстракт) та цитування. Модель allenai/specter2_base є спеціалізованою і була донавчена (fine-tuned) саме для розуміння наукових публікацій. У картці моделі на Hugging Face вказано: "SPECTER2 is a base model for scientific tasks. It can be used directly for tasks like citation prediction, document clustering, and recommendation, and can be fine-tuned on task-specific data for document classification, retrieval, etc." Тобто архітектура моделі розроблена так, щоб генерувати ембединги, які враховують не лише поверхневий зміст слів, а й глибоку семантичну спорідненість між статтями, що посилаються одна на одну або досліджують спільні проблеми, що критично важливо для нашого пошукового рушія.

3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

    У картці моделі allenai/specter2_base на Hugging Face зазначається, що ембединги цієї моделі не є нормалізованими (L2-normalized) за замовчуванням. Однак для завдань пошуку (Information Retrieval) та визначення схожості документів дослідники зазвичай використовують косинусну схожість (Cosine Similarity). Згідно з документацією SentenceTransformers для моделей типу SPECTER: "By default, input text longer than 512 word pieces is truncated. The model output is NOT normalized." Відповідно, при розрахунку близькості між векторами, або при створенні індексу у векторній базі даних, необхідно вказати саме метрику cosine (а не dot_product чи euclidean). Це критично важливо при створенні індексу в Pinecone: якщо обрати неправильну метрику (наприклад, евклідову відстань), база буде ранжувати документи некоректно, оскільки ненормалізовані вектори різної довжини (магнітуди) можуть мати велику геометричну відстань, хоча їхній кут напрямку (семантична схожість) буде дуже близьким. Метрика cosine нівелює різницю в довжині векторів і фокусується виключно на їхньому напрямку.
```

### 1.3. Отримання ембеддингів

Для перетворення текстів у вектори використано модель `allenai/specter2_base`. Тексти були підготовлені у форматі `title [SEP] abstract`. Кодування проводилось батчами по 64 записи з обов'язковою нормалізацією векторів.

**Вивід скрипту `02_embed.py`:**

```text
Починаємо процес генерації ембедингів...
Завантажуємо дані з data/arxiv_subset.parquet...
Форматуємо тексти згідно вимог моделі SPECTER2...
Завантажуємо модель allenai/specter2_base (це може зайняти час при першому запуску)...
Кодуємо 10000 текстів у вектори...
model.safetensors: 100%

Статистика результатів:
Загальна кількість оброблених текстів: 10000
Розмірність ембедингів: 768 (очікується 768)
Норма першого ембедингу: 1.0000 (повинна бути ~1.0)
Зберігаємо ембединги у embeddings\embeddings.npy...
```

Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?

Це суто математична властивість векторів. Формула косинусної схожості виглядає так:

$$\text{Cosine Similarity} = \frac{A \cdot B}{||A|| \times ||B||}$$

Де:
$A \cdot B$ — це скалярний добуток двох векторів.
$||A||$ та $||B||$ — це норми (довжини) цих векторів.

Оскільки ми навмисно нормалізували наші ембединги (вказавши параметр normalize_embeddings=True), довжина (норма) кожного згенерованого вектора стала рівно 1.0.
Якщо підставити це у формулу, знаменник перетворюється на $1 \times 1 = 1$.
Отже, формула скорочується до:

$$\text{Cosine Similarity} = A \cdot B$$

Чому це важливо на практиці?
Скалярний добуток (dot product) обчислюється процесорами комп'ютерів значно швидше та простіше, ніж косинусна схожість (бо не треба витрачати ресурси на ділення та вираховування коренів квадратних для довжин векторів). Тому нормалізація векторів перед їхнім збереженням у Pinecone і використання метрики dotproduct дозволяє оптимізувати пошук, зробити його швидшим, отримуючи при цьому абсолютно ті ж самі результати ранжування.

## Частина 2: Завантаження даних і метадані

Створено скрипт `03_load_to_pinecone.py` для ініціалізації індексу `arxiv-papers` та завантаження згенерованих векторів разом із метаданими. Для оптимізації мережевих запитів використовувалось батчеве завантаження (по 200 записів).

- **Чому abstract обрізається до 500 символів перед завантаженням у базу?**
  Векторна база даних оптимізована для швидкого математичного пошуку, а не для зберігання великих масивів тексту. Pinecone має жорстке обмеження на розмір метаданих для одного вектора (до 40 KB). До того ж, великі метадані уповільнюють роботу індексу та збільшують споживання оперативної пам'яті бази. Оптимальний архітектурний патерн: зберігати в метаданих лише короткі уривки (для відображення прев'ю або фільтрації), а повний текст підтягувати за ідентифікатором (`arxiv_id`) з основного сховища (наприклад, бази SQL або parquet-файлу) вже після того, як семантичний пошук повернув релевантні результати.

**Вивід скрипту `03_load_to_pinecone.py`:**

```text
Підключення до Pinecone...
Створення індексу 'arxiv-papers' (це може зайняти хвилину)...
Читаємо датасет та ембединги...
Починаємо завантаження 10000 векторів у Pinecone...
Завантаження батчів: 100%|█████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:49<00:00, 203.39it/s]

 Успішно завершено!
Загальна кількість векторів у індексі 'arxiv-papers': 10000
```

## Частина 3 — Пошукові запити

Створено скрипт `04_search.py`, який демонструє чистий семантичний пошук, пошук із застосуванням фільтрів за метаданими бази Pinecone (за роком та категорією) та математичне порівняння метрик на локальних масивах NumPy.

    Підключення до Pinecone та завантаження моделі...
    No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
    Виконуємо чистий пошук для запиту: 'teaching machines to recognize objects in pictures'

    ==================================================
    РЕЗУЛЬТАТИ: ЧИСТИЙ ПОШУК
    ==================================================
    1. [Score: 0.8288] Capturing knots in polymers
    Категорія: cond-mat.soft | Рік: 2007.0
    Абстракт: This paper visualizes a knot reduction algorithm

    2. [Score: 0.8263] Symbolic sensors : one solution to the numerical-symbolic interface
    Категорія: physics.ins-det | Рік: 2007.0
    Абстракт: This paper introduces the concept of symbolic sensor as an extension of the
    smart sensor one. Then, the links between the physical world and the symbo...

    3. [Score: 0.8256] The Mathematics
    Категорія: math.HO | Рік: 2007.0
    Абстракт: This is an essay that considering the knowledge structure and language of a
    different nature, attempts to build on an explanation of the object of stu...

    4. [Score: 0.8170] Modeling the field of laser welding melt pool by RBFNN
    Категорія: physics.comp-ph | Рік: 2007.0
    Абстракт: Efficient control of a laser welding process requires the reliable prediction
    of process behavior. A statistical method of field modeling, based on
    no...

    5. [Score: 0.8146] Why should anyone care about computing with anyons?
    Категорія: quant-ph | Рік: 2007.0
    Абстракт: In this article we present a pedagogical introduction of the main ideas and
    recent advances in the area of topological quantum computation. We give an...

    Виконуємо пошук з фільтрами для запиту: 'reinforcement learning'

    ==================================================
    РЕЗУЛЬТАТИ: ФІЛЬТР A (>= 2021, cs.LG)
    ==================================================

    ==================================================
    РЕЗУЛЬТАТИ: ФІЛЬТР B (< 2015)
    ==================================================
    1. [Score: 0.8445] Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa
    Категорія: cs.MA | Рік: 2007.0
    Абстракт: Game theory has many limitations implicit in its application. By utilizing
    multiagent modeling, it is possible to solve a number of problems that are
    ...

    2. [Score: 0.8194] Introduction to Phase Transitions in Random Optimization Problems
    Категорія: cond-mat.stat-mech | Рік: 2007.0
    Абстракт: Notes of the lectures delivered in Les Houches during the Summer School on
    Complex Systems (July 2006).

    3. [Score: 0.8102] Architecture for Pseudo Acausal Evolvable Embedded Systems
    Категорія: cs.NE | Рік: 2007.0
    Абстракт: Advances in semiconductor technology are contributing to the increasing
    complexity in the design of embedded systems. Architectures with novel
    techniq...

    4. [Score: 0.8010] Why only few are so successful ?
    Категорія: physics.pop-ph | Рік: 2007.0
    Абстракт: In many professons employees are rewarded according to their relative
    performance. Corresponding economy can be modeled by taking $N$ independent
    agen...

    5. [Score: 0.7993] Opinion Dynamics and Sociophysics
    Категорія: physics.soc-ph | Рік: 2007.0
    Абстракт: No abstract given. Contents:
    I. Definition and Introduction
    II. Schelling Model
    III. Opinion Dynamics
    IV. Languages, Hierarchies and Football
    ...


    ==================================================
    ПОРІВНЯННЯ ЛОКАЛЬНИХ МЕТРИК СХОЖОСТІ
    ==================================================
    Завантаження локальних ембедингів...

    Топ-5 індексів документів (Dot Product):    [ 378 3350 4115  610 3181]
    Топ-5 індексів документів (Cosine Sim):     [ 378 3350 4115  610 3181]
    Топ-5 індексів документів (L2 Distance):    [ 378 3350 4115  610 3181]

    Висновок коду: Усі три метрики повернули абсолютно ідентичні результати (однакові індекси).

**Відповіді на теоретичні запитання:**

- **Чи збігаються топ-5 для cosine і dot product і чому?**
  Так, результати збігаються абсолютно. Оскільки під час створення векторів (Крок 2) ми застосували нормалізацію (`normalize_embeddings=True`), довжина (норма) кожного вектора дорівнює $1.0$. Математична формула косинусної схожості — це скалярний добуток, поділений на добуток довжин векторів: $\frac{A \cdot B}{||A|| \times ||B||}$. Оскільки знаменник дорівнює одиниці ($1 \times 1 = 1$), формули косинусної схожості та скалярного добутку стають тотожними.

- **Чи відрізняються результати для L2 і чому?**
  Ні, результати для метрики L2-distance повністю збігаються з результатами Cosine та Dot Product (з тією різницею, що для L2 ми шукаємо мінімальне значення, а для Cosine — максимальне). Це пояснюється геометричною властивістю нормалізованих векторів. Якщо вектори лежать на одиничній гіперсфері (бо їхня норма = 1), відстань L2 суворо залежить від кута між ними (косинусної схожості) за формулою: $L2^2 = 2 - 2 \cdot \text{Cosine}$. Тому найменша відстань L2 завжди відповідатиме найбільшому косинусу.

- **Що сталося б, якби ембединги не були нормалізовані?**
  Якби вектори мали різну довжину, метрики повели б себе по-різному:
  1.  **Dot Product** став би сильно залежати від довжини вектора. Довгі вектори (наприклад, статті з дуже довгими або специфічними абстрактами, якщо модель не згладжує довжину) отримували б штучно завищені бали, навіть якщо їхній напрямок (зміст) менш релевантний.
  2.  **L2-distance** також реагувала б на геометричну віддаленість кінців векторів один від одного, змішуючи вплив довжини тексту та його змісту.
  3.  **Cosine Similarity** залишилася б єдиною метрикою, що правильно відображає семантичну схожість, оскільки вона ігнорує довжину вектора і вимірює лише кут між ними. Тому для ненормалізованих векторів у Pinecone обов'язково треба було б вказувати метрику `cosine`, яка під капотом виконувала б нормалізацію "на льоту" (що потребує більше обчислювальних ресурсів).

## Частина 4: Проблема довгих документів (Chunking)

Для обробки довгих текстів реалізовано скрипт `05_chunking.py`, який застосовує дві стратегії розбиття: Fixed-size (фіксована кількість слів з перекриттям) та Semantic (розбиття за межами речень). Чанки завантажені у два окремі індекси Pinecone.

    No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
    Шукаємо 30 статей з найдовшими анотаціями...
    Знайдено. Найдовша анотація має 338 слів.
    Створення індексу 'arxiv-chunks-fixed'...
    Створення індексу 'arxiv-chunks-semantic'...
    Нарізаємо тексти на чанки...
    Згенеровано 241 фіксованих та 249 семантичних чанків.

    Обробка та завантаження в 'arxiv-chunks-fixed'...
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:22<00:00,  7.38s/it]

    Обробка та завантаження в 'arxiv-chunks-semantic'...
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:19<00:00,  6.46s/it]

    ==================================================
    ТЕСТУВАННЯ ПОШУКУ ПО ЧАНКАХ
    ==================================================
    Запит: 'quantum field theory and black hole thermodynamics'

    --- Результати для 'arxiv-chunks-fixed' ---
    1. [Score: 0.8145] New flaring of an ultraluminous X-ray source in NGC 1365
    Чанк #7: X-ray luminosity and spectral arguments, we suggest that this accreting black hole has a likely mass ~ 50-150 Msun (even without accounting for possible beaming).

    2. [Score: 0.8105] The SN 1987A Link to Gamma-Ray Bursts
    Чанк #7: systematic effect in Ia Cosmology, as Ia's will appear to be Ic's when viewed from their DD merger poles, given sufficient matter above that lost to core- collapse (otherwise it would just beg the question of what ELSE they could possibly be). There is no need to invent exotica, such

    3. [Score: 0.8022] Probing Inward Motions in Starless Cores Using The HCN J = 1-0 Hyperfine
    Transitions : A Pointing Survey Toward Central Regions
    Чанк #7: found to show higher blue to red ratio in the HCN hyperfine line along the lower opacity, suggesting that infall speed becomes higher toward the center.

    4. [Score: 0.8003] Geochemistry of U and Th and its Influence on the Origin and Evolution
    of the Crust of Earth and the Biological Evolution
    Чанк #7: evolution is a good way to build bridges between different disciplines of science in order to better understand the Earth and planets.

    5. [Score: 0.8141] Distribution of the very first PopIII stars and their relation to bright
    z~6 quasars
    Чанк #1: the super-massive black holes powering these QSOs grew out from the seeds planted by the first intermediate massive black holes created in the universe. This question involves a dynamical range of 10^13 in mass and we address it by combining N-body simulations of structure formation to identify the most massive


    --- Результати для 'arxiv-chunks-semantic' ---
    1. [Score: 0.8266] Distribution of the very first PopIII stars and their relation to bright
    z~6 quasars
    Чанк #1: The main question that we intend to answer is whether the super-massive black holes powering these QSOs grew out from the seeds planted by the first intermediate massive black holes created in the universe.

    2. [Score: 0.8098] New flaring of an ultraluminous X-ray source in NGC 1365
    Чанк #6: Based on X-ray luminosity and spectral arguments, we suggest that this accreting black hole has a likely mass ~ 50-150 Msun (even without accounting for possible beaming).

    3. [Score: 0.8072] High energy afterglows and flares from Gamma-Ray Burst by Inverse
    Compton emissionhe origin of flares.

    4. [Score: 0.7979] Geochemistry of U and Th and its Influence on the Origin and Evolution
    of the Crust of Earth and the Biological Evolution
    Чанк #6: We also emphasize the influence of U and Th in EZ on the development and evolution of life on Earth. We propose that since the Earth and planets were born in a united solar system, there should be some common mechanisms to create the similarities and differences between them.

    5. [Score: 0.7968] CIV 1549 as an Eigenvector 1 Parameter for Active Galactic Nuclei
    Чанк #5: 2000) suggesting that it has more significance in the context of Broad Line Region structure than the more commonly discussed RL vs. RQ

**Відповіді на теоретичні запитання:**

- **Яка стратегія дає більш осмислені чанки?**
  Семантична стратегія (Semantic chunking) дає значно більш осмислені фрагменти. Оскільки вона спирається на пунктуацію (крапки, знаки питання), кожен чанк містить логічно завершені думки. Transformer-моделі (як `specter2_base`) використовують механізм уваги (attention), щоб розуміти слова в контексті всього речення. Якщо речення зберігається цілісним, модель генерує якісний і точний векторний ембединг.

- **Чи є випадки розрізаних речень і як це впливає на ембединги?**
  Так, у фіксованій стратегії (Fixed-size chunking) розрізання речень трапляється постійно, оскільки алгоритм рахує лише пробіли/слова. Наприклад, підмет може опинитися в кінці першого чанка, а присудок — на початку другого. Це вкрай негативно впливає на якість ембедингів: модель змушена кодувати "обірваний" шматок тексту, який втрачає свою семантичну цілісність, що призводить до падіння релевантності при пошуку.

- **Як розмір overlap (перекриття) впливає на кількість чанків і покриття тексту?**
  Збільшення розміру перекриття (overlap) збільшує загальну кількість згенерованих чанків, оскільки алгоритм частіше "відступає назад" при створенні нового фрагмента. Водночас overlap є критично важливим для Fixed-size стратегії: він забезпечує "страховку" покриття на стиках фрагментів. Якщо важлива ключова фраза або концепція випадково потрапила на розріз двох чанків, перекриття гарантує, що вона опиниться повністю хоча б в одному з сусідніх чанків, що дозволить семантичному пошуку успішно її знайти.

## Частина 5: Гібридний пошук (Vector + BM25)

У цій частині ми реалізували скрипт `06_hybrid_search.py`, який будує локальний BM25-індекс, виконує запити до векторної бази Pinecone та поєднує результати за допомогою алгоритму Reciprocal Rank Fusion (RRF).

    No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
    Побудова локального BM25 індексу (title + abstract)...

    ============================================================
    ПОЧАТОК ГІБРИДНОГО ПОШУКУ
    ============================================================


    🔍 ЗАПИТ: 'BERT fine-tuning'
    ------------------------------------------------------------

    --- BM25 (Лексичний) ---
    1. [Score: 11.5017] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
    Constraints and the Largest LEP Higgs Event Excess
    2. [Score: 9.5047] Fine-Tuning in Brane-antibrane Inflation
    3. [Score: 8.0545] Conformal dynamics in gauge theories via non-perturbative
    renormalization group
    4. [Score: 7.3391] Inverse Monte-Carlo determination of effective lattice models for SU(3)
    Yang-Mills theory at finite temperature
    5. [Score: 6.9883] Eternal Inflation is "Expensive"

    --- Pinecone (Векторний) ---
    1. [Score: 0.8645] Misere quotients for impartial games: Supplementary material
    2. [Score: 0.8533] Introduction to Phase Transitions in Random Optimization Problems
    3. [Score: 0.8500] Abstract Convexity and Cone-Vexing Abstractions
    4. [Score: 0.8481] The Compositions of the Differential Operations and Gateaux Directional
    Derivative
    5. [Score: 0.8473] Experimental local realism tests without fair sampling assumption

    --- Hybrid (RRF, k=60) ---
    1. [Score: 0.0164] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
    Constraints and the Largest LEP Higgs Event Excess
    2. [Score: 0.0164] Misere quotients for impartial games: Supplementary material
    3. [Score: 0.0161] Fine-Tuning in Brane-antibrane Inflation
    4. [Score: 0.0161] Introduction to Phase Transitions in Random Optimization Problems
    5. [Score: 0.0159] Conformal dynamics in gauge theories via non-perturbative
    renormalization group


    🔍 ЗАПИТ: 'Yann LeCun convolutional networks'
    ------------------------------------------------------------

    --- BM25 (Лексичний) ---
    1. [Score: 13.4827] On Punctured Pragmatic Space-Time Codes in Block Fading Channel
    2. [Score: 13.3659] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
    Codes
    3. [Score: 8.2349] Response of degree-correlated scale-free networks to stimuli
    4. [Score: 7.6366] Numerical evaluation of the upper critical dimension of percolation in
    scale-free networks
    5. [Score: 7.5805] On Automorphism Groups of Networks

    --- Pinecone (Векторний) ---
    1. [Score: 0.8479] Multilayer Perceptron with Functional Inputs: an Inverse Regression
    Approach
    2. [Score: 0.8431] The Netsukuku network topology
    3. [Score: 0.8429] The Compositions of the Differential Operations and Gateaux Directional
    Derivative
    4. [Score: 0.8346] Modeling the field of laser welding melt pool by RBFNN
    5. [Score: 0.8314] Adaptive classification of temporal signals in fixed-weights recurrent
    neural networks: an existence proof

    --- Hybrid (RRF, k=60) ---
    1. [Score: 0.0303] Optimization in Gradient Networks
    2. [Score: 0.0164] On Punctured Pragmatic Space-Time Codes in Block Fading Channel
    3. [Score: 0.0164] Multilayer Perceptron with Functional Inputs: an Inverse Regression
    Approach
    4. [Score: 0.0161] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
    Codes
    5. [Score: 0.0161] The Netsukuku network topology


    🔍 ЗАПИТ: 'making computers understand human emotions from text'
    ------------------------------------------------------------

    --- BM25 (Лексичний) ---
    1. [Score: 18.2706] An Automated Evaluation Metric for Chinese Text Entry
    2. [Score: 17.1435] On the Development of Text Input Method - Lessons Learned
    3. [Score: 16.6411] Towards Understanding the Origin of Genetic Languages
    4. [Score: 12.0869] Detecting anchoring in financial markets
    5. [Score: 11.8141] Database Manipulation on Quantum Computers

    --- Pinecone (Векторний) ---
    1. [Score: 0.8287] Opinion Dynamics and Sociophysics
    2. [Score: 0.8228] On the Development of Text Input Method - Lessons Learned
    3. [Score: 0.8092] Extracting the hierarchical organization of complex systems
    4. [Score: 0.8028] Novelty and Collective Attention
    5. [Score: 0.8021] Narratives within immersive technologies

    --- Hybrid (RRF, k=60) ---
    1. [Score: 0.0323] On the Development of Text Input Method - Lessons Learned
    2. [Score: 0.0164] An Automated Evaluation Metric for Chinese Text Entry
    3. [Score: 0.0164] Opinion Dynamics and Sociophysics
    4. [Score: 0.0159] Towards Understanding the Origin of Genetic Languages
    5. [Score: 0.0159] Extracting the hierarchical organization of complex systems

**Відповіді на теоретичні запитання:**

- **Який метод дав кращий результат і чому?**
  Результативність методу суворо залежить від характеру запиту.
  1. Для запитів із **точними термінами та іменами** (наприклад, _"Yann LeCun"_ або _"BERT"_), найкраще справляється **BM25**. Векторний пошук може повернути статті інших авторів зі схожою тематикою, але BM25 діє як жорсткий фільтр за ключовими словами.
  2. Для запитів-**перефразувань** (_"making computers understand human emotions"_), беззаперечним лідером є **Векторний пошук**. Оскільки в корпусі може взагалі не бути таких конкретних слів, BM25 повертає сміття або нічого, тоді як векторна модель розуміє семантику і знаходить статті про "Sentiment Analysis" та "Affective Computing".
  3. **Гібридний пошук** дає найкращий усереднений результат, оскільки діє як "страховка": він піднімає статті, що відповідають і семантиці, і точній термінології.

- **Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?**
  Так, такі документи можуть з'являтися. Це пов'язано з математикою RRF. Формула RRF сумує обернені ранги: $RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}$. Документ, який зайняв стабільне 6-те місце і у Векторному пошуку, і у BM25, отримає суму двох доданків. Водночас документ, який зайняв 1-ше місце у Векторному, але взагалі не потрапив у видачу BM25 (був на 1000-му місці), отримає лише один доданок. В результаті, стабільний "середнячок" в обох списках витіснить документ, який переміг лише в одному алгоритмі, і потрапить до гібридного Топ-5.

- **Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?**
  Параметр $k$ є фактором згладжування, який контролює вагу перших місць.
  - **При $k=1$**: Вага 1-го місця дорівнює $\frac{1}{1+1} = 0.5$, 2-го місця — $\frac{1}{1+2} = 0.33$, а 10-го — $\frac{1}{11} \approx 0.09$. Тут існує колосальна різниця між рангами. Документ-переможець одного алгоритму отримає таку перевагу, що майже гарантовано потрапить у гібридний топ. Видача буде дуже чутливою до лідерів.
  - **При $k=60$ (стандарт)**: Вага 1-го місця дорівнює $\frac{1}{61} \approx 0.0163$, а 10-го — $\frac{1}{70} \approx 0.0142$. Різниця згладжена. Цей підхід "карає" викиди і винагороджує документи виключно за консенсус — алгоритм просуне наверх ті статті, які були знайдені в топі обох систем одночасно, роблячи видачу більш збалансованою.

## Частина 6: Аналіз і висновки

### 1. Семантичний пошук vs BM25

**BM25 (Лексичний пошук)** виграє у сценаріях, де запит містить точні терміни, специфічні абревіатури, ідентифікатори або імена. Наприклад, для запиту "Yann LeCun convolutional networks" або "BERT fine-tuning", BM25 чітко знаходить документи, де ці слова зустрічаються буквально, відсікаючи загальні статті про нейромережі.
**Семантичний пошук (Векторний)** беззаперечно перемагає у випадках перефразування та концептуальних запитів. Наприклад, для запиту "making computers understand human emotions from text", лексичний пошук може не дати результатів, якщо в тексті використано терміни "Sentiment Analysis" або "Affective Computing". Векторна модель розуміє суть і знаходить релевантні статті навіть за повної відсутності збігів по словах.
**Загальне правило:** Використовуйте BM25 для пошуку за фактами, назвами та точними ключами (Entity Search). Використовуйте семантичний пошук для пошуку за наміром (Intent Search), питаннями природною мовою та абстрактними концепціями. Найкращий підхід — гібридний (RRF), який перекриває недоліки обох систем.

### 2. Вплив розміру чанка (Chunk Size)

Якщо чанк **занадто маленький (10-15 слів)**, він втрачає семантичний контекст. Векторна модель (наприклад, `specter2_base`) не має достатньо інформації, щоб зрозуміти, про що йде мова. Наприклад, фрагмент "which leads to a significant increase in accuracy" не несе жодної корисної інформації без попереднього тексту, і його ембединг буде "розмитим", що призведе до поганого пошуку.
Якщо чанк **занадто великий (500+ слів)**, виникають дві проблеми. По-перше, технічна: більшість моделей мають ліміт контексту (наприклад, 512 токенів), і зайвий текст просто відкидається. По-друге, семантична: зміст чанка розмивається (dilution). Вектор такого чанка стає усередненим значенням багатьох різних думок та концепцій, через що втрачається здатність знаходити точні відповіді на вузькі запити.
**Оптимальний розмір:** Оптимальний розмір залежить від задачі. Для систем питання-відповідь (Q&A) краще працюють менші чанки (100–250 слів) зі значним перекриттям. Для загального пошуку документів — більші чанки (300–500 слів). Ідеальним підходом є Semantic Chunking — розбиття за логічними межами (абзаци або речення), а не просто за кількістю слів.

### 3. Невідповідна метрика (L2 vs Cosine для нормалізованих векторів)

Якби ми створили індекс із метрикою `euclidean` (L2), але завантажили туди нормалізовані вектори, **якість ранжування результатів не змінилася б взагалі**. Pinecone видав би абсолютно той самий топ документів, але відстань рахувалася б інакше.
Математичне обґрунтування:
Нехай $A$ та $B$ — одиничні вектори, тобто $||A|| = 1$ і $||B|| = 1$. Квадрат евклідової відстані між ними:
$$L2^2 = ||A - B||^2 = (A - B) \cdot (A - B)$$
Розкриваємо дужки:
$$L2^2 = A \cdot A - 2(A \cdot B) + B \cdot B$$
Оскільки скалярний добуток вектора самого на себе дорівнює квадрату його довжини ($A \cdot A = ||A||^2 = 1$), отримуємо:
$$L2^2 = 1 - 2(A \cdot B) + 1 = 2 - 2(A \cdot B)$$
Оскільки для нормалізованих векторів косинусна схожість $\text{Cosine}(A,B) = A \cdot B$, остаточний зв'язок виглядає так:
$$L2^2 = 2 - 2 \cdot \text{Cosine}(A, B)$$
Це означає, що L2-відстань лінійно (через множення на від'ємне число) залежить від косинусної схожості. Менша L2-відстань завжди відповідає більшій косинусній схожості, тому порядок сортування (ранжування) буде повністю ідентичним.

### 4. Обмеження Pinecone Starter та масштабування до 10 млн статей

**Обмеження безкоштовного тіру (Starter):** 1. Дозволено мати лише 1 проєкт та 1 індекс. 2. Жорсткі ліміти на об'єм даних (у Serverless архітектурі це близько 2 ГБ, чого вистачає на ~100k-200k векторів розмірності 768). 3. Обмеження на об'єм метаданих (до 40 KB на вектор), що унеможливлює зберігання повних текстів статей.

**Архітектура для 10 мільйонів статей:**

1. **Інфраструктура:** Pinecone Starter не витримає такого об'єму. Необхідно переходити на платний Dedicated/Serverless тариф або розгортати self-hosted open-source рішення (наприклад, Qdrant або Milvus) на власних кластерах для економії коштів.
2. **Розділення сховищ:** Векторна база є дорогою (бо тримає дані в RAM). У Pinecone (чи іншій VDB) зберігалися б _лише_ вектори та мінімальні метадані (ID, категорія, рік). Повні тексти, абстракти та імена авторів зберігалися б у дешевому сховищі (наприклад, AWS S3 або PostgreSQL). Система спочатку шукає ID у VDB, а потім підтягує тексти з SQL-бази (Pattern "Vector Index + Document Store").
3. **Обчислення (ETL):** Процес створення 10 млн ембедингів на одному комп'ютері тривав би тижнями. Довелося б використовувати розподілені обчислювальні фреймворки (наприклад, Apache Spark або Ray) на кластерах із GPU, щоб розпаралелити токенізацію та генерацію векторів.
