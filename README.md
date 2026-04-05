# Scene Analysis

Приложение получает monocular depth map, преобразует ее в непрерывную `obstacle heatmap` и оценивает качество heatmap по `Average Precision`.

`video -> frame reading -> preprocessing -> Depth Anything V2 -> road suppression -> obstacle heatmap -> evaluation`

## Стек

- Python 3.11-3.14
- Poetry
- OpenCV
- NumPy
- PyYAML
- Typer
- Loguru
- Pydantic v2
- pytest
- torch
- transformers
- pillow
- safetensors
- tqdm
- matplotlib
- pandas
- scikit-learn

## Установка

```bash
poetry install
```

## Запуск

### Обработка видеофайла

1. Положите видеофайл в `data/raw/sample.mp4`.
2. Запустите обработку:

```bash
poetry run scene-analysis run-video --config configs/base.yaml
```

Пример запуска с depth model override:

```bash
poetry run scene-analysis run-video \
  --config configs/base.yaml \
  --depth-model depth-anything/Depth-Anything-V2-Small-hf
```

Дополнительно можно переопределить параметры фильтров:

```bash
poetry run scene-analysis run-video \
  --config configs/base.yaml \
  --suppression-strength 1.0 \
  --bottom-strip-ratio 0.35 \
  --gamma 1.1
```

### Обработка потока изображений

Если у вас уже есть отдельные кадры, а не видео, можно сразу сгенерировать
`obstacle heatmap` predictions в `.npy` по stem имени файла:

```bash
poetry run scene-analysis generate-predictions --config configs/base.yaml
```

Пример с явными путями:

```bash
poetry run scene-analysis generate-predictions \
  --config configs/base.yaml \
  --dataset-root data/datasets/road_obstacle_21 \
  --images-dir images \
  --predictions-dir predictions \
  --output-dir data/artifacts/image_run_001
```

### Запуск evaluation

```bash
poetry run scene-analysis evaluate-heatmap --config configs/base.yaml
```

Пример с override путей:

```bash
poetry run scene-analysis evaluate-heatmap \
  --config configs/base.yaml \
  --dataset-root data/datasets/road_obstacle_21 \
  --predictions-dir predictions \
  --output-dir data/artifacts/eval_run_001
```

### Как подготовить predictions_dir

1. Подготовьте локальный датасет, например в `data/datasets/road_obstacle_21`.
2. Положите obstacle masks в `masks/`.
3. Положите предсказанные heatmap в `predictions/`.
4. Имена prediction и mask должны совпадать по `sample_id`, например:

- `predictions/frame_000001.npy`
- `masks/frame_000001.png`

## Результат запуска инференса

После выполнения команды в `output.output_dir` будут сохранены:

- исходные кадры;
- обработанные кадры;
- кадры с overlay;
- `depth_npy/frame_XXXXXX.npy`;
- `depth_colormap/frame_XXXXXX.png`;
- `obstacle_heatmap_npy/frame_XXXXXX.npy`;
- `obstacle_heatmap_png/frame_XXXXXX.png`;
- `results.jsonl` с metadata по каждому обработанному кадру.

## Результат запуска evaluation

После выполнения команды в `evaluation.outputs.output_dir` будут сохранены:

- `summary.json` с глобальным `Average Precision`;
- `per_sample.csv` с per-sample статистикой;
- `pr_curve.png` с precision-recall curve;
- `hard_examples.csv` со сложными sample.

## Что уже реализовано

- monocular depth inference через `Depth Anything V2`;
- преобразование depth map в `near-score`;
- подавление дороги и опорной поверхности через baseline;
- усиление вертикальных препятствий над дорогой;
- сглаживание и очистка итоговой obstacle heatmap;
- сохранение `obstacle_heatmap_npy/frame_XXXXXX.npy`;
- сохранение `obstacle_heatmap_png/frame_XXXXXX.png`;
- сохранение overlay с obstacle heatmap;
- сохранение `results.jsonl` с metadata по depth и obstacle heatmap.

## Как работает

Оценивается не depth map, а именно `obstacle heatmap`.

- основная итоговая метрика: `Average Precision`;
- evaluation работает по локальному датасету;
- ground truth поддерживает три состояния пикселей: obstacle, background, ignore;
- dataset-level AP считается по всем валидным пикселям всех sample вместе;
- предсказания по умолчанию ожидаются в `predictions_dir` как `.npy` heatmap файлы.

## Поддерживаемые модели

По умолчанию используется:

- `depth-anything/Depth-Anything-V2-Small-hf`

Также поддерживаются совместимые модели, например:

- `depth-anything/Depth-Anything-V2-Base-hf`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf`
