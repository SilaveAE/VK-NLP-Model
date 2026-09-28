# MIRACL NLP Reranker Project

Этот проект подготавливает MIRACL данные, обучает CrossEncoder reranker и выполняет инференс по запросам.

## Что реализовано

- `prepare_data.py` — формирует полный корпус `data/corpus.jsonl` и пары `data/train_pairs.jsonl`, `data/dev_pairs.jsonl`.
- `train_reranker.py` — обучает `CrossEncoder` на парах `(запрос, документ)`.
- `evaluate.py` — оценивает модель на dev-выборке по MRR@10, P@1 и Hit@10.
- `infer.py` — извлекает кандидатов BM25 и переранжирует их обученной моделью.
- `utils.py` — функции загрузки MIRACL данных и формирования тренировочных пар.

## Установка

1. Откройте терминал в папке проекта.
2. Создайте виртуальное окружение (рекомендуется):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Установите зависимости:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Подготовка полного датасета

Чтобы использовать весь набор данных, запустите:

```powershell
python prepare_data.py --corpus-dir corpus --topics-train topics\topics.miracl-v1.0-ru-train.tsv --qrels-train qrels\qrels.miracl-v1.0-ru-train.tsv --topics-dev topics\topics.miracl-v1.0-ru-dev.tsv --qrels-dev qrels\qrels.miracl-v1.0-ru-dev.tsv --output-dir data
```

Это создаст:

- `data/corpus.jsonl` — корпус документов, используемый для инференса и оценки.
- `data/train_pairs.jsonl` — тренировочные пары из всего train-сета.
- `data/dev_pairs.jsonl` — валидационные пары из dev-сета.

### Рекомендуемые опции

- `--negative-ratio 1` — стандартный баланс положительных и отрицательных примеров.
- Не указывайте `--max-docs`, чтобы загрузить все документы из `corpus`.
- `--include-all-negatives` — если хотите включить все отрицательные документы, связанные с запросом.

## Обучение на полном датасете

Для полноценной тренировки используйте весь `data/train_pairs.jsonl` и минимальные ограничения:

```powershell
python train_reranker.py --train-pairs data\train_pairs.jsonl --dev-pairs data\dev_pairs.jsonl --output-dir models\reranker --epochs 3 --batch-size 16 --device cuda
```

Если у вас мало VRAM или нет GPU, установите `--device cpu` и уменьшите `--batch-size` до `8` или `4`.

### Рекомендации для лучшего качества

- `--epochs 3` или `--epochs 5` — больше эпох обычно повышает качество.
- `--batch-size 16` или `8` — зависит от памяти GPU.
- `--model-name cross-encoder/ms-marco-MiniLM-L-6-v2` — быстрый стартовый вариант.
- При желании замените `--model-name` на более крупную или русскоязычную модель.

## Оценка модели

После обучения запустите оценку на dev-выборке:

```powershell
python evaluate.py --model-path models\reranker --qrels-dev qrels\qrels.miracl-v1.0-ru-dev.tsv --topics-dev topics\topics.miracl-v1.0-ru-dev.tsv --corpus-jsonl data\corpus.jsonl --top-k 10
```

Ожидаемые метрики:

- `MRR@10` — оценка ранжирования по первому релевантному документу.
- `P@1` — доля запросов, где релевантный документ оказался на первом месте.
- `Top-10 Hit` — доля запросов, где релевантный документ попал в топ-10.

## Инференс

Запустите модель для реального запроса:

```powershell
python infer.py --model-path models\reranker --corpus-jsonl data\corpus.jsonl --query "Кто такой Юрий Гагарин?" --top-k 3
```

Скрипт загружает корпус BM25, находит кандидатов и переранжирует их `CrossEncoder`.

## Улучшение качества

1. Использовать `--epochs 3` или `5` при обучении.
2. Тренировать на полном наборе `data/train_pairs.jsonl` без `--max-train-samples`.
3. Увеличить `--batch-size`, если хватает памяти.
4. Попробовать более мощную модель `--model-name`, например `cross-encoder/ms-marco-MiniLM-L-12-v2`.
5. Если нужно, повторить подготовку с `--include-all-negatives`.

## Быстрый старт

```powershell
python prepare_data.py --corpus-dir corpus --topics-train topics\topics.miracl-v1.0-ru-train.tsv --qrels-train qrels\qrels.miracl-v1.0-ru-train.tsv --topics-dev topics\topics.miracl-v1.0-ru-dev.tsv --qrels-dev qrels\qrels.miracl-v1.0-ru-dev.tsv --output-dir data
python train_reranker.py --train-pairs data\train_pairs.jsonl --dev-pairs data\dev_pairs.jsonl --output-dir models\reranker --epochs 3 --batch-size 16 --device cuda
python evaluate.py --model-path models\reranker --qrels-dev qrels\qrels.miracl-v1.0-ru-dev.tsv --topics-dev topics\topics.miracl-v1.0-ru-dev.tsv --corpus-jsonl data\corpus.jsonl --top-k 10
python infer.py --model-path models\reranker --corpus-jsonl data\corpus.jsonl --query "Кто такой Юрий Гагарин?" --top-k 3
```

## Файлы проекта

- `prepare_data.py`
- `train_reranker.py`
- `evaluate.py`
- `infer.py`
- `utils.py`
- `requirements.txt`
- `README.md`
