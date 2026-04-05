# Scene Analysis

Приложение получает monocular depth map и преобразует ее в непрерывную `obstacle heatmap`.

`video -> frame reading -> preprocessing -> Depth Anything V2 -> road suppression -> obstacle heatmap -> artifact saving`

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

Дополнительно можно переопределить параметры фильтров:

```bash
poetry run scene-analysis run-video \
  --config configs/base.yaml \
  --suppression-strength 0.9 \
  --bottom-strip-ratio 0.35 \
  --gamma 1.1
```

## Что уже реализовано

- загрузка и валидация YAML-конфига для obstacle heatmap;
- inference monocular depth через `Depth Anything V2`;
- преобразование depth map в `near-score`;
- подавление дороги и опорной поверхности через baseline;
- усиление вертикальных препятствий над дорогой;
- сглаживание и очистка итоговой obstacle heatmap;
- сохранение `obstacle_heatmap_npy/frame_XXXXXX.npy`;
- сохранение `obstacle_heatmap_png/frame_XXXXXX.png`;
- сохранение overlay с obstacle heatmap;
- сохранение `results.jsonl` с metadata по depth и obstacle heatmap.

## Что важно

- отдельные объекты не детектируются;
- дорога подавляется, а не выделяется как класс;

## Поддерживаемые модели

По умолчанию используется:

- `depth-anything/Depth-Anything-V2-Small-hf`

Также поддерживаются совместимые модели, например:

- `depth-anything/Depth-Anything-V2-Base-hf`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf`

## Результат запуска

После выполнения команды в `output.output_dir` будут сохранены:

- исходные кадры;
- обработанные кадры;
- кадры с overlay;
- `depth_npy/frame_XXXXXX.npy`;
- `depth_colormap/frame_XXXXXX.png`;
- `obstacle_heatmap_npy/frame_XXXXXX.npy`;
- `obstacle_heatmap_png/frame_XXXXXX.png`;
- `results.jsonl` с metadata по каждому обработанному кадру.
