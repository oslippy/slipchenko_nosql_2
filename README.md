## Відповіді на теоретичні питання

### 1. Pinecone проти Qdrant і Chroma

Найбільше ці три бази різняться тим, як їх запускають і кому належить код.

Pinecone — закритий хмарний сервіс. Своєї копії не піднімеш, працюєш тільки через API, а інфраструктуру тримає вендор. Зручно, бо адмініструвати нічого не треба. Але ти прив'язаний до постачальника і платиш за використання.

Qdrant — повна протилежність. Відкритий код (Apache 2.0), написаний на Rust. Можна підняти в себе через Docker чи Kubernetes, а можна взяти готову Qdrant Cloud. Тобто сам вирішуєш, де він живе і хто за ним стежить.

Chroma теж відкрита (Apache 2.0) і найпростіша зі стартом: вбудовується прямо в Python-застосунок і зберігає дані локально, без окремої бази. Серверний режим є, але зазвичай її беруть для роботи на своїй машині.

По швидкості все впирається в масштаб. На маленьких колекціях найшвидша Chroma, бо немає мережі й усе крутиться в пам'яті процесу. Коли даних більшає і хостиш сам, попереду Qdrant завдяки Rust. А на мільйонах векторів, де треба стабільна затримка й доступність без власного DevOps, практичніший Pinecone, бо масштабуванням займається хмара.

Тож обирав би я так. Chroma — щоб швидко зібрати прототип чи невеликий локальний пошук. Qdrant — коли треба відкрите рішення під своїм контролем: on-prem, складна фільтрація, гібрид хмара+локально. Pinecone — для продакшну, де важливіше не возитися з інфраструктурою, ніж мати відкритий код.

### 2. Чому specter2_base, а не all-MiniLM-L6-v2

all-MiniLM-L6-v2 — універсальна модель «на все». У картці так і написано: "It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search". Тренували її на суміші з понад мільярда звичайних пар речень з інтернету. Вона маленька і швидка, добре вловлює загальний зміст. А от наукову мову й зв'язки між статтями знає радше поверхово.

specter2_base зроблена саме під наукові тексти. У картці на HuggingFace прямо сказано, на чому її вчили: "SPECTER2 has been trained on over 6M triplets of scientific paper citations". Тобто моделі показували трійки статей (запит, реально процитована стаття і випадкова стороння) і привчали ставити їх так, щоб пов'язані цитуванням роботи опинялися поруч. Серед форматів задач, під які її готували, є й Proximity (Retrieval), тобто пошук найближчих сусідів — рівно те, що нам і потрібно.

Через це по наукових статтях specter2_base ранжує доречніше: вона розуміє термінологію і «знає», які роботи зазвичай цитують поруч. Платою стає розмір і швидкість. Це BERT на ~110М параметрів проти 22М у MiniLM, вектори по 768 значень замість 384 (більше місця в індексі), плюс лише англійська і потреба в бібліотеці `adapters`. Якби треба було швидкий загальний пошук на слабкій машині, я б узяв MiniLM. Але тут пошук саме по науці, тому логічніша specter2_base.

### 3. Яку метрику схожості брати і чому це важливо вирішити одразу

Чесно кажучи, конкретної метрики в картці specter2_base не названо. Там описано лише що модель кодує, а не чим міряти відстань між векторами. Зрозуміти її можна з того, як модель навчали. SPECTER і SPECTER2 тренують triplet loss'ом, де близькість рахується через евклідову відстань (L2): простір підганяють так, щоб споріднені статті були ближчими. Але на практиці для таких щільних векторів майже завжди беруть косинусну подібність, бо вона не залежить від довжини вектора, а на нормалізованих векторах дає те саме впорядкування, що й L2. Тому індекс під цю модель я створив би з `metric="cosine"` і `dimension=768` (стільки чисел у векторі specter2_base). Евклідова теж підійшла б. А dotproduct має сенс лише якщо самому нормалізувати вектори, тоді він зведеться до того ж косинуса.

Чому це треба продумати ще на старті? По-перше, у Pinecone метрика задається при створенні індексу і потім не змінюється: щоб перейти на іншу, індекс доводиться видаляти і заливати всі вектори наново. По-друге, метрика має відповідати тому, як модель розуміє схожість. Візьмеш не ту (скажімо, dotproduct на ненормалізованих векторах) і пошук тихо зіпсується: самі вектори нормальні, але сусіди шикуються неправильно, і релевантні статті просто не потраплять у топ. Запит і база при цьому в одній метриці, тож помилку навіть не видно як помилку, просто результати гірші, ніж могли б бути. Тому метрику підбирають під модель ще до того, як наповнювати індекс.

### 4. Чому для нормалізованих ембеддингів косинус дорівнює скалярному добутку

Косинусна схожість — це скалярний добуток векторів, поділений на добуток їхніх довжин: `cos(a, b) = (a * b) / (||a|| * ||b||)`. Ділення тут лише прибирає вплив довжини й лишає сам кут між векторами. Якщо вектори наперед нормалізовані до одиничної довжини, то `||a|| = ||b|| = 1`. Знаменник стає одиницею і зникає, лишається чистий чисельник `a * b`. Виходить, для одиничних векторів `cos(a, b) = a * b`: косинус і скалярний добуток дають те саме число.

Звідси й практична користь: коли вектори нормалізовані (саме це вмикає `normalize_embeddings=True`), індексу можна задати метрику `dotproduct` замість `cosine`. Ранжування вийде однаковим, зате рахується трохи швидше, бо не треба щоразу обчислювати норми й ділити на них.

### 5. Фільтрація за метаданими: чому видача A і B різна

Обидва приклади шукають "reinforcement learning", різниця лише у фільтрі. Приклад A (категорія `cs.LG` + рік >= 2021) повертає саме те, що очікуєш: свіжі роботи про RL зі скорами ~0.83-0.85, як-от «Operator Deep Q-Learning» (2022), «Bootstrapped Reward Shaping» (2025), «Toward Causal-Aware RL» (2022), «E-GRPO... for Flow Models» (2026). Усі п'ять із `cs.LG`, усі 2021+, усі справді про навчання з підкріпленням. Приклад B (будь-яка категорія, рік < 2015) дає старіший і строкатіший набір зі скорами ~0.79-0.82. Поряд із ранніми RL-роботами (Policy Evaluation 2013, Local Optimality of RL 2011) сюди потрапляють `math.OC`, `math.ST` і `cond-mat`, тобто статті, що зачепилися радше за слова «learning»/«temporal», ніж за саму тему. Виходить, звуження за категорією+роком тримає видачу гострою й на часі, а широкий фільтр впускає старші й дотичні лише за вектором роботи з сусідніх галузей.

### 6. Порівняння метрик: cosine, dot product і L2

`04_search.py` рахує всі три метрики локально по `embeddings.npy` для одного запиту — і результат збігається з теорією.

Cosine і dot product дають однаковий топ-5. Причому не просто той самий набір статей, а й ті самі числа: у прогоні обидва повернули paper_5055, 4404, 4376, 6852, 6048 зі скорами 0.853, 0.849, 0.847... Причина та сама, що в п. 4: вектори нормалізовані, тож `cos(a, b) = a * b`, це буквально одне й те саме значення.

L2 повертає той самий топ-5, лише в іншій шкалі. Числа інакші, це відстань (менше = ближче: 0.541, 0.550, 0.552...), але порядок статей ідентичний. Для одиничних векторів `||a - b||^2 = 2 - 2*(a * b)`, тобто відстань монотонно спадає з ростом косинуса: більший косинус, менша відстань. Ранжування від цього не змінюється, хоч метрика формально інша.

А якби вектори не були нормалізовані, метрики розійшлися б. Косинус від довжини не залежить, він усе одно ділить на норми й міряє тільки кут. Dot product натомість почав би залежати від довжини: довші вектори діставали б вищий скор незалежно від напрямку, і топ зсунувся б до статей із більшою нормою. L2 теж зреагувала б на різницю довжин, бо два колінеарні вектори різної довжини мали б ненульову відстань. Саме нормалізація прибирає вплив довжини й зводить усі три метрики до одного ранжування. Без неї cosine, dot і L2 загалом дали б три різні топ-5.

### 7. Chunking: fixed-size проти semantic

Осмисленіші чанки дає semantic. Він ріже текст по межах речень, тож кожен чанк лишається закінченою думкою, і ембеддинг кодує цілісний зміст. У прогоні це видно одразу: semantic-чанки починаються з початку речення («The resulting WR catalog includes...», «Variable non-thermal radio emission...»), а fixed — із середини («consistent with the shape of the mass power spectrum...», «an early epoch, and follows the merger history...»). Fixed просто відлічує по 50 слів, тому межі чанків випадкові.

Розрізаних речень у fixed-нарізці справді багато: майже кожен чанк, крім першого в статті, стартує з половини фрази. Для ембеддингів це радше шкодить, бо specter2 навчена на зв'язному тексті, а обірваний фрагмент типу «point source distribution correlates with the mass of the galaxy...» несе менше контексту, тож його вектор менш чітко представляє тему. Semantic цього здебільшого уникає, хоч і не ідеально: наївний розбивач по `.!?` спотикається на скороченнях і десяткових (`e.g.`, `0.5`), тому зрідка ріже не там.

Overlap прямо керує кількістю чанків і покриттям. Крок між чанками дорівнює `size - overlap`, тож більший overlap означає менший крок, більше чанків і більше дублювання на стиках: концепт, що ліг на межу, цілим потрапляє хоча б в один сусідній чанк. У прогоні fixed з overlap=10 на вікні 50 слів дав 245 чанків. Прибрати overlap — і крок зросте з 40 до 50, чанків стане помітно менше, зате зросте ризик, що думка на межі розділиться навпіл і недопредставиться в пошуку. По суті overlap — це розмін «більше векторів і обчислень» на «кращу повноту на межах».

### 8. Гібридний пошук: BM25, вектор і RRF

Який метод кращий — залежить від запиту, і це власне причина існування гібрида. На «BERT fine-tuning» виразно виграв BM25: точно збігшись із термінами, він підняв справжні статті про BERT і fine-tuning («On Explaining Your Explanations of BERT», «Adaptive Fine-tuning for Multiclass Classification», «Prefix-Tuning»), тоді як вектор з'їхав на загальний ML (RL, inference), бо ловив сенс, а не конкретну назву моделі. На «emotions from text» сильніший уже вектор. Він витягнув тематичні «An affective computational model for machine consciousness», «Bible sentiment analysis», «EmoGator», а BM25 чіплявся за слово «text» (Text2Avatar, Text-to-Image). Коротко: BM25 на точних термінах, вектор на сенсі, універсального переможця немає.

Гібрид не просто чергує два списки. RRF піднімає документи, на яких методи погоджуються, і це дає влучання, відсутні в топ-5 обох окремо. Найчіткіший приклад — запит про емоції. Гібрид #1 «Ensemble emotion recognizing with multiple modal physiological» (rrf 0.0301) не входить у топ-5 ні BM25, ні вектора. Він лежить десь на 6-7 місці в обох списках, але `1/(60+6) + 1/(60+7) ~ 0.030` обганяє `1/(60+1) ~ 0.016` у будь-якого лідера лише одного методу. Згода двох середніх сигналів перемагає один сильний. Те саме на «LeCun...»: нагору вийшли консенсусні «Handwritten Indic Character Recognition using Capsule Networks» (0.0318) і «GoogLe2Net» (0.0310), що були в обох списках.

k керує саме цим балансом. У `1/(k+rank)` велике k=60 майже зрівнює сусідні ранги, тож важливіше бути в обох списках, ніж першим в одному. Через це консенсусні документи й спливають угору. Мале k=1 робить формулу крутою (0.5, 0.33, 0.25...), і домінує перше місце: при k=1 гібрид на «emotions» очолив би вже не консенсусний «Ensemble emotion recognizing» (він лише ~6-7 у кожному списку, ваги по ~0.13), а той, хто №1 хоч в одному методі (вага 0.5). Так що великий k — довіряй згоді, малий — довіряй верхівці.

### 9. Семантичний пошук проти BM25: де який виграв

Це вже частково видно з п. 8, тож коротко. BM25 виграв на «BERT fine-tuning»: точно збігається з рідкісним терміном і піднімає саме статті про BERT, а вектор там з'їжджає на загальний ML, бо ловить сенс «навчання моделі», а не конкретну назву. На перефразуванні про емоції навпаки: вектор знаходить тематичні affective/sentiment-роботи, де слів із запиту майже немає, а BM25 чіпляється за буквальне «text».

Загальне правило таке. BM25 краще там, де важливе точне слово: абревіатури (BERT, RRF), імена авторів, ідентифікатори, рідкісний жаргон, коди, тобто коли користувач знає термін і хоче саме його. Вектор виграє на природних, описових запитах, перефразуваннях і синонімах, коли потрібний документ говорить про те саме іншими словами. А якщо тип запиту наперед невідомий, гібрид через RRF страхує обидва випадки.

### 10. Вплив розміру чанка

Занадто малий чанк (10-15 слів) майже не несе контексту: вектор такого уривка описує вирвану фразу, а не думку. У пошуку він або поверхово збігається з безліччю запитів, або ні з чим, тобто точність падає, а кількість векторів (і вартість) зростає. Занадто великий чанк (500+ слів) має дві біди. По-перше, він не влазить у 512-токенне вікно specter2/BERT, тож хвіст просто обрізається і частина тексту в ембеддинг не потрапляє. По-друге, велика порція змішує кілька підтем в один усереднений вектор: запит, що стосується однієї з них, дістає «розмитий» скор, а в результат повертається великий блок, де потрібне речення губиться.

Універсального розміру немає, він залежить від задачі. Орієнтир такий: чанк має бути достатньо великим, щоб лишатися самодостатньою думкою, і достатньо малим, щоб не мішати теми й не впиратися в ліміт токенів моделі. На практиці це десь 100-300 слів. Для точкового пошуку (знайти конкретний факт) кращі менші чанки, для широкого контексту (RAG, де уривок іде в LLM) — більші. І майже завжди розбивка по межах речень (semantic) виграє в арбітрарної фіксованої.

### 11. Невідповідна метрика: euclidean-індекс на нормалізованих векторах

Нічого б не зламалося: видача була б ідентичною cosine. Це випливає з математики. Для одиничних векторів `a`, `b` (`||a|| = ||b|| = 1`):

```
||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a * b) = 1 + 1 - 2*cos(a, b) = 2 - 2*cos(a, b)
```

Тобто `||a - b|| = sqrt(2 - 2*cos)`. Це строго спадна функція від косинуса: що більший `cos`, то менша L2-відстань. А отже впорядкування за зростанням L2 збігається з упорядкуванням за спаданням косинуса, тобто найближчі сусіди ті самі і топ-K той самий. Саме це показав `04_search.py` у п. 6: L2 повернув той самий топ-5, що й cosine, лише в шкалі відстані. Тож euclidean-індекс на нормалізованих векторах дає правильний результат, відрізняються тільки числові скори, не порядок. Небезпечною невідповідність стає лише на ненормалізованих векторах: тоді `||a-b||^2 = ||a||^2 + ||b||^2 - 2(a*b)` залежить ще й від довжин, і L2 з косинусом розходяться.

### 12. Обмеження Pinecone Starter і масштабування до 10 млн

Реальні межі Starter-тіру, в які впираєшся: лише один регіон (`aws us-east-1`), обмежена кількість індексів (а ми тримаємо вже три: `arxiv-papers` + два чанк-індекси), місячні ліміти на storage та read/write-одиниці, відсутність бекапів, плюс жорсткий ліміт 40 KB метаданих на вектор (через нього й ріжемо abstract до 500 символів). На 10k x 768 усе вміщається, але це майже стеля безкоштовного тіру.

Якби датасет був 10 млн статей, підхід довелося б міняти на кількох рівнях. Сховище: 10 млн x 768 x 4 байти, це ~30 ГБ сирих векторів, тобто вже платний serverless. Щоб збити вартість, я б застосував квантизацію (int8, у 4 рази менше) або меншу розмірність (384-вимірна модель чи Matryoshka-обрізання вектора). Обчислення: ембеддити 10 млн через specter2 на CPU нереально, потрібен GPU/батч-інференс або хостована inference-модель із паралельним завантаженням батчами. Метадані: повний текст тримати поза Pinecone (об'єктне сховище чи БД), а в індексі лишати тільки id + мінімум полів для фільтрів. Архітектурно — шардинг за роком/категорією через namespaces, щоб фільтровані запити не сканували все. А якщо вартість керованого сервісу стане критичною, можна піти на self-hosted Qdrant/Milvus із квантизацією на власному залізі.

## Порівняння методів пошуку (Частина 5)

Топ-1 результат кожного методу для трьох тестових запитів:

| Запит | BM25 | Вектор | Гібрид (RRF) |
|---|---|---|---|
| `BERT fine-tuning` | On Explaining Your Explanations of BERT | Reinforcement Learning for Flexibility Design | On Explaining... BERT — 0.0164 |
| `Yann LeCun convolutional networks` | Depth-Adaptive Computational Policies | Handwritten Indic... Capsule Networks | Handwritten Indic... Capsule Networks — 0.0318 |
| `making computers understand human emotions from text` | Text2Avatar | An affective computational model for machine consciousness | Ensemble emotion recognizing — 0.0301 |

Патерн: на термінному запиті (BERT) гібрид іде за BM25, а на семантичному (emotions) піднімає консенсусний документ, якого немає в топ-1 жодного методу окремо.

## Вивід скриптів

### `01_prepare_data.py`

```
Розподіл за категоріями (топ-10):
astro-ph             2724
hep-ph                320
quant-ph              293
cs.CV                 265
hep-th                225
cs.LG                 221
cond-mat.mtrl-sci     208
math.CO               194
gr-qc                 181
cond-mat.mes-hall     172

Розподіл за роками: 2000-2026, по 370 статей щороку (разом 9990)

Збережено в data/arxiv_subset.parquet
```

### `02_embed.py`

```
Оброблено текстів:       9990
Розмірність ембеддингів: 768
Норма першого вектора:   1.0000
Збережено у embeddings/embeddings.npy (shape=(9990, 768))
```

### `03_load_to_pinecone.py`

```
Завантаження завершено. Векторів в індексі 'arxiv-papers': 9990
```

### `04_search.py`

```
крок 3: семантичний пошук
запит: teaching machines to recognize objects in pictures
1. 0.853  cs.CV 2020  Butterfly Detection and Classification Based on Integrated YOLO Algorithm
2. 0.849  cs.LG 2018  Deep Learning for Identifying Potential Conceptual Shifts for Co-creative Drawing
3. 0.847  cs.CV 2018  Learning audio and image representations with bio-inspired trainable feature extractors
4. 0.844  cs.CV 2023  Hierarchical Explanations for Video Action Recognition
5. 0.842  eess.IV 2025  A Novel Approach using CapsNet and Deep Belief Network for Detection and Identification of Oral Leukopenia

крок 4: пошук з фільтром, запит: reinforcement learning
A) cs.LG, останні 5 років (рік >= 2021):
1. 0.847  cs.LG 2022  Operator Deep Q-Learning: Zero-Shot Reward Transferring in Reinforcement Learning
2. 0.844  cs.LG 2025  Bootstrapped Reward Shaping
3. 0.843  cs.LG 2022  Toward Causal-Aware RL: State-Wise Action-Refined Temporal Difference
4. 0.835  cs.LG 2023  On the Challenges of using Reinforcement Learning in Precision Drug Dosing
5. 0.832  cs.LG 2026  E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models

B) будь-яка категорія, рік < 2015:
1. 0.822  cs.LG 2013  Policy Evaluation with Variance Related Risk Criteria in Markov Decision Processes
2. 0.813  cs.LG 2011  The Local Optimality of Reinforcement Learning by Value Gradients
3. 0.800  math.OC 2013  On Distributed Online Classification in the Midst of Concept Drifts
4. 0.794  math.ST 2009  Adjustment coefficient for risk processes in some dependent contexts
5. 0.792  cond-mat.stat-mech 2014  Enhanced Sampling in Molecular Dynamics Using Metadynamics

категорії в A: ['cs.LG']
категорії в B: ['cond-mat.stat-mech', 'cs.LG', 'math.OC', 'math.ST']

крок 5: метрики для запиту: teaching machines to recognize objects in pictures
cosine:
   0.8535  paper_5055  Butterfly Detection and Classification Based on Integra
   0.8488  paper_4404  Deep Learning for Identifying Potential Conceptual Shif
   0.8474  paper_4376  Learning audio and image representations with bio-inspi
   0.8436  paper_6852  A Novel Approach using CapsNet and Deep Belief Network
   0.8432  paper_6048  Hierarchical Explanations for Video Action Recognition
dot:
   0.8535  paper_5055  Butterfly Detection and Classification Based on Integra
   0.8488  paper_4404  Deep Learning for Identifying Potential Conceptual Shif
   0.8474  paper_4376  Learning audio and image representations with bio-inspi
   0.8436  paper_6852  A Novel Approach using CapsNet and Deep Belief Network
   0.8432  paper_6048  Hierarchical Explanations for Video Action Recognition
l2 (відстань, менша = ближча):
   0.5412  paper_5055  Butterfly Detection and Classification Based on Integra
   0.5499  paper_4404  Deep Learning for Identifying Potential Conceptual Shif
   0.5524  paper_4376  Learning audio and image representations with bio-inspi
   0.5593  paper_6852  A Novel Approach using CapsNet and Deep Belief Network
   0.5599  paper_6048  Hierarchical Explanations for Video Action Recognition

топ-5 однаковий для всіх трьох метрик: True
```

### `05_chunking.py`

```
30 статей -> fixed: 245 чанків, semantic: 249 чанків

запит: dark matter distribution in galaxies
fixed:
1. 0.843  The power spectrum of galaxy clustering in the APM survey  [чанк 4]
2. 0.834  Dynamical evolution of intermediate mass black holes...     [чанк 1]
3. 0.834  White Paper: Radio Emission and Polarization Properties    [чанк 1]
4. 0.834  Dynamical evolution of intermediate mass black holes...     [чанк 7]
5. 0.830  Wolf-Rayet galaxies in SDSS-IV MaNGA. I. Catalog...        [чанк 5]
semantic:
1. 0.842  Wolf-Rayet galaxies in SDSS-IV MaNGA. I. Catalog...        [чанк 4]
2. 0.839  The power spectrum of galaxy clustering in the APM survey  [чанк 4]
3. 0.839  Dynamical evolution of intermediate mass black holes...     [чанк 5]
4. 0.837  Wolf-Rayet galaxies in SDSS-IV MaNGA. I. Catalog...        [чанк 5]
5. 0.836  White Paper: Radio Emission and Polarization Properties    [чанк 1]

запит: gamma-ray burst afterglow
fixed:
1. 0.833  Radio emission of the Galactic X-rays binaries...           [чанк 7]
2. 0.832  Kilohertz QPO Frequency Anti-Correlated with mHz QPO...     [чанк 7]
3. 0.830  Radio emission of the Galactic X-rays binaries...           [чанк 0]
4. 0.828  Radio emission of the Galactic X-rays binaries...           [чанк 6]
5. 0.822  Radio emission of the Galactic X-rays binaries...           [чанк 1]
semantic:
1. 0.843  Radio emission of the Galactic X-rays binaries...           [чанк 2]
2. 0.833  Radio emission of the Galactic X-rays binaries...           [чанк 7]
3. 0.828  Radio emission of the Galactic X-rays binaries...           [чанк 0]
4. 0.820  Radio emission of the Galactic X-rays binaries...           [чанк 6]
5. 0.819  Kilohertz QPO Frequency Anti-Correlated with mHz QPO...     [чанк 7]
```

### `06_hybrid_search.py`

```
запит: BERT fine-tuning
bm25:
1. On Explaining Your Explanations of BERT: An Empirical Study
2. Adaptive Fine-tuning for Multiclass Classification over Software
3. Alzheimer's disease detection based on large language model prompt
4. Prefix-Tuning: Optimizing Continuous Prompts for Generation
5. Analyzing Commonsense Emergence in Few-shot Knowledge Models
вектор:
1. Reinforcement Learning for Flexibility Design Problems
2. Thresholds of descending algorithms in inference problems
3. Quantum King-Ring Domination in Chess: A QAOA Approach
4. Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems
5. Generative Deep Learning for Virtuosic Classical Music
гібрид (rrf):
1. 0.0164  On Explaining Your Explanations of BERT: An Empirical Study
2. 0.0164  Reinforcement Learning for Flexibility Design Problems
3. 0.0161  Adaptive Fine-tuning for Multiclass Classification over Software
4. 0.0161  Thresholds of descending algorithms in inference problems
5. 0.0159  Alzheimer's disease detection based on large language model prompt

запит: Yann LeCun convolutional networks
bm25:
1. Depth-Adaptive Computational Policies for Efficient Visual Tracking
2. Self-Taught Convolutional Neural Networks for Short Text Clustering
3. Plugin Networks for Inference under Partial Evidence
4. GoogLe2Net: Going Transverse with Convolutions
5. Handwritten Indic Character Recognition using Capsule Networks
вектор:
1. Handwritten Indic Character Recognition using Capsule Networks
2. BERT-JEPA: Reorganizing CLS Embeddings for Language-Invariant Semantics
3. Biologically Inspired Hexagonal Deep Learning for Hexagonal Image
4. Lossless Compression of Deep Neural Networks
5. GoogLe2Net: Going Transverse with Convolutions
гібрид (rrf):
1. 0.0318  Handwritten Indic Character Recognition using Capsule Networks
2. 0.0310  GoogLe2Net: Going Transverse with Convolutions
3. 0.0164  Depth-Adaptive Computational Policies for Efficient Visual Tracking
4. 0.0161  Self-Taught Convolutional Neural Networks for Short Text Clustering
5. 0.0161  BERT-JEPA: Reorganizing CLS Embeddings for Language-Invariant Semantics

запит: making computers understand human emotions from text
bm25:
1. Text2Avatar: Text to 3D Human Avatar Generation with Codebook
2. Hierarchical Vision-Language Alignment for Text-to-Image Generation
3. Judge the Judges: A Large-Scale Evaluation Study of Neural Language
4. Second Thoughts are Best: Learning to Re-Align With Human Values
5. Reasoning based on symbolic and parametric knowledge bases: a survey
вектор:
1. An affective computational model for machine consciousness
2. Assessing Emoji Use in Modern Text Processing Tools
3. An Attentive Sequence Model for Adverse Drug Event Extraction
4. Large language model for Bible sentiment analysis: Sermon on the Mount
5. EmoGator: A New Open Source Vocal Burst Dataset with Baseline
гібрид (rrf):
1. 0.0301  Ensemble emotion recognizing with multiple modal physiological
2. 0.0297  EmoGator: A New Open Source Vocal Burst Dataset with Baseline
3. 0.0164  Text2Avatar: Text to 3D Human Avatar Generation with Codebook
4. 0.0164  An affective computational model for machine consciousness
5. 0.0161  Hierarchical Vision-Language Alignment for Text-to-Image Generation
```
