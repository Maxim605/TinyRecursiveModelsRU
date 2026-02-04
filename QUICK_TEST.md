# Быстрый тест обучения (10 минут)

## Решение проблем

### Проблема 1: adam-atan2 не установлен

Если вы получили ошибку `ModuleNotFoundError: No module named 'adam_atan2_backend'`, код автоматически использует `torch.optim.AdamW` вместо `adam-atan2`.

### Проблема 2: GPU не найден в WSL

Если вы получили ошибку `RuntimeError: Found no NVIDIA driver on your system`, это означает, что WSL не видит вашу GPU.

**Быстрое решение:**
1. Установите NVIDIA драйвер для WSL на Windows (версия 515+)
2. Или запустите обучение на Windows напрямую (если драйверы уже установлены)
3. Или используйте CPU (очень медленно, но работает для теста)

Подробные инструкции см. в `WSL_GPU_SETUP.md`

**Временное решение для CPU теста:**
Код теперь автоматически определяет доступное устройство. Если GPU недоступен, будет использован CPU (с предупреждением). 

⚠️ **Внимание:** Обучение на CPU будет **очень медленным** (в 10-100 раз медленнее GPU). Для 10 эпох на маленьком датасете может занять 30-60 минут вместо 10 минут.

**Рекомендация:** Лучше настроить GPU в WSL (см. `WSL_GPU_SETUP.md`) или запустить на Windows напрямую.

Если вы получили ошибку `ModuleNotFoundError: No module named 'adam_atan2_backend'`, это означает, что пакет `adam-atan2` не скомпилировался правильно. 

### Вариант 1: Переустановить adam-atan2 (рекомендуется)

```bash
# Установить необходимые инструменты компиляции
sudo apt-get update
sudo apt-get install -y build-essential ninja-build

# Переустановить adam-atan2
pip uninstall adam-atan2 -y
pip install --no-cache-dir --no-build-isolation adam-atan2
```

### Вариант 2: Использовать стандартный AdamW (быстрое решение)

Код уже обновлен для автоматического fallback на `torch.optim.AdamW`, если `adam-atan2` недоступен. Просто запустите команду ниже - она будет работать с AdamW вместо AdamATan2.

**Примечание:** AdamW может быть немного медленнее, но для теста это не критично.

## Шаг 1: Создать минимальный тестовый датасет

Создайте очень маленький датасет для быстрого теста:

```bash
# Создать минимальный датасет Sudoku (10 примеров, без аугментаций)
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-test-mini \
  --subsample-size 10 \
  --num-aug 0
```

Это создаст датасет всего с 10 примерами, что достаточно для проверки работоспособности.

## Шаг 2: Запустить быстрый тест обучения

**Код теперь автоматически использует `torch.optim.AdamW` если `adam-atan2` недоступен!**

```bash
# Быстрый тест - обучение за ~10 минут
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-test-mini]" \
evaluators="[]" \
epochs=10 \
eval_interval=10 \
global_batch_size=4 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=1 \
arch.H_cycles=1 \
arch.L_cycles=2 \
arch.hidden_size=128 \
arch.halt_max_steps=4 \
lr_warmup_steps=10 \
ema=False \
+run_name=quick_test_10min
```

**Важно:** `eval_interval` должен быть делителем `epochs`. Для `epochs=10` используйте `eval_interval=10`, `5`, `2`, или `1`.

**Если увидите предупреждение `WARNING: adam-atan2 not available, using torch.optim.AdamW instead` - это нормально, обучение будет работать!**

## Параметры для быстрого теста:

- **epochs=10** - всего 10 эпох (вместо 50000)
- **global_batch_size=4** - минимальный батч
- **eval_interval=10** - оценка выполнится в конце (должен быть делителем epochs)
- **L_layers=1** - всего 1 слой (вместо 2)
- **H_cycles=1, L_cycles=2** - минимальные циклы
- **hidden_size=128** - маленькая модель (вместо 512)
- **halt_max_steps=4** - максимум 4 шага ACT (вместо 16)
- **lr_warmup_steps=10** - быстрый warmup
- **ema=False** - отключено для ускорения
- **evaluators="[]"** - без оценщиков

## Ожидаемое время:

- **Создание датасета**: ~1-2 минуты
- **Обучение**: ~5-10 минут на 16GB GPU
- **Итого**: ~10-12 минут

## Что проверить:

1. ✅ Модель загружается без ошибок
2. ✅ Обучение начинается (видны шаги в консоли)
3. ✅ Loss уменьшается (даже немного)
4. ✅ Нет ошибок памяти (OOM)
5. ✅ Чекпоинт сохраняется

## Если нужно еще быстрее (5 минут):

```bash
# Экстремально быстрый тест
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-test-mini]" \
evaluators="[]" \
epochs=5 \
eval_interval=5 \
global_batch_size=2 \
lr=1e-3 puzzle_emb_lr=1e-3 weight_decay=0.1 puzzle_emb_weight_decay=0.1 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=1 \
arch.H_cycles=1 \
arch.L_cycles=1 \
arch.hidden_size=64 \
arch.halt_max_steps=2 \
lr_warmup_steps=5 \
ema=False \
+run_name=quick_test_5min
```

**Примечание:** `eval_interval=5` должен быть делителем `epochs=5`

## Альтернатива: Использовать существующий датасет с ограничением

Если у вас уже есть датасет, можно ограничить количество эпох:

```bash
# Использовать существующий датасет, но обучить только 1 эпоху
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=1 \
eval_interval=1 \
global_batch_size=8 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=1 \
arch.H_cycles=1 \
arch.L_cycles=2 \
arch.hidden_size=128 \
arch.halt_max_steps=4 \
lr_warmup_steps=10 \
ema=False \
+run_name=quick_test_1epoch
```

**Примечание:** `eval_interval=1` должен быть делителем `epochs=1`

Это пройдет 1 эпоху, что займет ~5-15 минут в зависимости от размера датасета.
