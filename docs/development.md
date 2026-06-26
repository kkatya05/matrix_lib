# Руководство для разработчиков

Этот документ описывает, как устроен проект, как его модифицировать, тестировать и публиковать.

## Структура проекта

matlib_lib/
├── .gitignore
├── .python-version
├── README.md
├── REPORT.md
├── mkdocs.yml
├── pyproject.toml
├── uv.lock
├── src/
|   ├──init.py
│   └── matlib_z3342_tem1_msgvk/
│       ├── init.py
│       ├── py.typed
│       ├── base.py # абстрактный класс Matrix
│       ├── dense.py # FullMatrix
│       ├── band.py # (заготовка)
│       ├── symmetric.py # SymmetricMatrix
│       ├── block.py # BlockMatrix
│       ├── solvers.py # solve, PLU
│       └── errors.py # исключения
├── tests/ # модульные тесты (pytest)
|   ├── test_dense.py
│   ├── test_structured.py
│   ├── test_block.py
│   ├── test_solvers.py
│   └── test_integration.py
├── docs/ # документация (MkDocs)
│   ├── index.md
│   ├── tutorial.md
│   ├── api.md
│   └── development.md
└── examples/ # примеры использования
    ├── poisson_1d.py
    └── block_matrix_demo.py

## Настройка окружения

Проект управляется через **uv**. Убедитесь, что `uv` установлен, затем:

```bash
# Создать виртуальное окружение и установить зависимости
uv sync
```

## Запуск тестов

```bash
uv run pytest
# Для запуска конкретного тестового файла
uv run pytest tests/test_dense.py
```

## Форматированние и линтинг

Использование ruff:

```bash
# Форматирование кода
uv run ruff format .

# Проверка стиля
uv run ruff check 
```

## Локальный просмотр документации

```bash
uv run mkdocs serve
```

## Сборка документации в статические файлы

Результат появится в папке site/.
```bash
uv run mkdocs build
```

## Сборка пакета

Будут созданы дистрибутивы в dist/.
```bash
uv build
```

## Публикация на TestPyPl

Зарегистрируйтесь на TestPyPI.
Настройте токен API или используйте имя пользователя/пароль.
Загрузите пакет:
```bash
uv publish --publish-url #нужно вставить нужный URL
```

## Как добавить новый матричный тип

- Создайте новый модуль (например, triangular.py) в папке src/matlib_z3342_tem1_msgvk/.

- Определите класс, наследующий от Matrix из base.py.

- Реализуйте абстрактные методы:

    shape, dtype (свойства)
    to_dense()
    matvec(x)
    empty_like()
    __getitem__, __setitem__

- Добавьте экспорт класса в __init__.py пакета.

- Напишите тесты в отдельном файле tests/test_<new>.py.

- Обновите документацию (API, tutorial).

## Запуск примеров

Примеры находятся в папке examples/. Запустить любой из них можно так:

```bash
uv run python examples/poisson_1d.py
uv run python examples/block_matrix_demo.py
```