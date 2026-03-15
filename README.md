# Scene Analysis

На текущем этапе проект не использует реальные ML-модели. Вместо этого реализован полный рабочий путь обработки:

`video -> frame reading -> preprocessing -> pipeline stubs -> artifact saving`

## Стек

- Python 3.11+
- Poetry
- OpenCV
- NumPy
- PyYAML
- Typer
- Loguru
- Pydantic v2
- pytest

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

Пример запуска с переопределением параметров:

```bash
poetry run scene-analysis run-video \
  --config configs/base.yaml \
  --input data/raw/sample.mp4 \
  --output-dir data/artifacts/run_manual \
  --max-frames 50 \
  --sample-every-n 2
```

## Что уже реализовано

- загрузка и валидация YAML-конфига;
- чтение видео через OpenCV;
- базовый препроцессинг кадров: ROI, resize, optional normalization;
- единые типизированные dataclass-модели результатов;
- dummy depth / obstacle map / dynamic detection модули;
- MVP pipeline с overlay;
- сохранение PNG-артефактов и JSONL-результатов;
- CLI на Typer;
- базовые тесты.

## Что будет дальше

- подключение реальных depth-моделей;
- генерация obstacle map и occupancy/cost maps;
- детекция и трекинг динамических объектов;
- временная логика между кадрами;
- teacher/student архитектура, distillation и training pipeline.

## Структура проекта

```text
project/
├── pyproject.toml
├── README.md
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
├── src/
│   └── scene_analysis/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_setup.py
│       ├── types.py
│       ├── utils.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── video_reader.py
│       │   └── artifact_writer.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── frame_preprocessor.py
│       ├── depth/
│       │   ├── __init__.py
│       │   └── base.py
│       ├── obstacle_map/
│       │   ├── __init__.py
│       │   └── base.py
│       ├── dynamic/
│       │   ├── __init__.py
│       │   └── base.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── mvp_pipeline.py
│       └── app/
│           ├── __init__.py
│           └── cli.py
├── tests/
│   ├── test_config.py
│   ├── test_preprocessing.py
│   └── test_pipeline.py
└── .gitignore
```

## Результат запуска

После выполнения команды в `output.output_dir` будут сохранены:

- исходные кадры;
- препроцессированные кадры;
- overlay-кадры;
- `results.jsonl` с метаданными по каждому обработанному кадру.
