# Scene Analysis

На данном этапе в приложение подключена реальная depth модель через Hugging Face Transformers API

`video -> frame reading -> preprocessing -> Depth Anything V2 -> artifact saving`

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

## Установка

```bash
poetry install
```

## Запуск

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

Дополнительно можно переопределить устройство и fp16:

```bash
poetry run scene-analysis run-video \
  --config configs/base.yaml \
  --device cuda \
  --fp16
```

## Что уже реализовано

- загрузка и валидация YAML-конфига;
- чтение видео через OpenCV;
- базовый препроцессинг кадров: ROI, resize, normalization;
- единые типизированные dataclass-модели результатов; 
- интеграция `Depth Anything V2` через `AutoImageProcessor` и `AutoModelForDepthEstimation`;
- обработка глубины через `DepthEstimator`;
- сохранение сырого файла глубины `.npy` и цветовой карты `.png`;
- сохранение PNG-артефактов и JSONL-результатов;
- CLI на Typer;
- базовые тесты.

## Поддерживаемые модели

По умолчанию используется:

- `depth-anything/Depth-Anything-V2-Small-hf`

Также поддерживаются совместимые модели, например:

- `depth-anything/Depth-Anything-V2-Base-hf`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf`

## Что будет дальше

- генерация карты глубины;
- детекция динамических объектов;
- teacher/student архитектура, distillation и training pipeline.

## Результат запуска

После выполнения команды в `output.output_dir` будут сохранены:

- исходные кадры;
- обработанные кадры;
- кадры с оверлеем;
- `depth_npy/frame_XXXXXX.npy`;
- `depth_colormap/frame_XXXXXX.png`;
- `results.jsonl` с метаданными по каждому обработанному кадру.
