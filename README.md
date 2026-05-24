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
- torchvision
- albumentations

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
  --dataset-root data/datasets/road_obstacle_21_raw \
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
  --dataset-root data/datasets/road_obstacle_21_raw \
  --predictions-dir predictions \
  --output-dir data/artifacts/eval_run_001
```

## Student pipeline

Student-модели работают по RGB-кадру без Depth Anything V2, depth map и road suppression:

`RGB image -> student model -> obstacle_logits + roi_logits -> sigmoid -> final_heatmap = obstacle_prob * roi_prob`

Teacher pipeline нужен только на этапе подготовки датасета: он генерирует `teacher_heatmaps/*.npy`, которые используются в distillation loss при обучении.

Поддерживаются три student-варианта:

- `student_s` — MobileNetV3 Small;
- `student_m` — ShuffleNetV2 x1.0;
- `student_q` — EfficientNet-B0.

### Подготовка student dataset

Ожидаемая сырая структура по умолчанию:

```text
data/datasets/road_obstacle_21_raw/
├── images/
└── masks/
```

Маски: `0` — background, `1` — obstacle, `255` — ignore. Если layout отличается, настройте `images_dir`, `masks_dir`, `image_suffix`, `mask_suffix` в `configs/student_train.yaml`.

```bash
poetry run scene-analysis prepare-student-data --config configs/student_train.yaml
```

Результат:

```text
data/datasets/road_obstacle_21_prepared/
├── split/train.txt
├── split/val.txt
├── train/images masks teacher_heatmaps
├── val/images masks teacher_heatmaps
├── teacher_previews/
└── prepare_summary.json
```

Повторный запуск не пересчитывает teacher heatmaps. Для пересчета:

```bash
poetry run scene-analysis prepare-student-data \
  --config configs/student_train.yaml \
  --overwrite-teacher-heatmaps
```

### Обучение student-моделей

Для полноценного обучения используйте CUDA GPU: Colab, Kaggle Notebook или локальный CUDA. В конфигах стоит `device: auto`: если доступна CUDA, она будет использована; если CUDA нет, программа предупредит, что CPU/MPS режим подходит в основном для smoke теста или дебага

Обучить все модели из конфига:

```bash
poetry run scene-analysis train-students --config configs/student_train.yaml
```

Обучить одну модель:

```bash
poetry run scene-analysis train-students \
  --config configs/student_train.yaml \
  --student student_s
```

### Smoke test

CPU-friendly проверка на 1 эпоху и несколько batch:

```bash
poetry run scene-analysis train-students --config configs/student_smoke.yaml --smoke
```

### Запуск student на папке видео

```bash
poetry run scene-analysis run-student-video-folder \
  --config configs/student_inference.yaml \
  --student student_s \
  --checkpoint data/artifacts/student_runs/student_heatmap_distillation/student_s/checkpoints/best.pt \
  --input-dir data/input_videos \
  --output-dir data/artifacts/student_video_folder/student_s_demo
```

### Запуск student на live camera

```bash
poetry run scene-analysis run-student-camera \
  --config configs/student_inference.yaml \
  --student student_s \
  --checkpoint data/artifacts/student_runs/student_heatmap_distillation/student_s/checkpoints/best.pt \
  --camera-index 0 \
  --output-dir data/artifacts/student_camera/student_s_demo
```

Клавиша `q` закрывает live OpenCV окно.

### Ожидаемые student-артефакты

После обучения:

```text
data/artifacts/student_runs/student_heatmap_distillation/<student_name>/
├── checkpoints/best.pt
├── checkpoints/last.pt
├── history.csv
├── summary.json
├── pr_curve.png
├── previews/
└── config_resolved.yaml
```

После video folder inference:

```text
data/artifacts/student_video_folder/<run_name>/<student_name>/
├── videos/*_overlay.mp4
├── heatmaps_png/
├── results.jsonl
└── summary.json
```

После camera inference:

```text
data/artifacts/student_camera/<run_name>/<student_name>/
├── camera_overlay.mp4
├── results.jsonl
└── summary.json
```

### Как интерпретировать AP и overlay

Основная метрика — global `Average Precision` по всем валидным пикселям validation split. Ignore-пиксели `255` исключаются из AP и не участвуют в BCE/Dice/distillation. Если в маленьком smoke-наборе нет одновременно positive и negative пикселей, AP будет `n/a`, но pipeline все равно сохранит checkpoint и diagnostic artifacts.

Визуально проверяйте `overlay` видео и `heatmaps_png`: яркие области показывают пиксели, где student ожидает препятствие; зеленый контур соответствует порогу `visualization.threshold`.

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
