# Руководство по использованию модели для инференса

## Что умеет модель?

Модель TRM (Tiny Recursive Reasoning Model) специализируется на решении **структурированных головоломок**:

### 1. **Sudoku (Судоку)**
- Решает судоку 9x9
- Вход: частично заполненная доска судоку
- Выход: полностью решенная доска

### 2. **ARC-AGI (Визуальные головоломки)**
- Решает визуальные паттерн-головоломки
- Вход: сетка с входным паттерном
- Выход: сетка с решением паттерна
- Достигает **45% точности на ARC-AGI-1** и **8% на ARC-AGI-2**

### 3. **Maze (Лабиринты)**
- Решает задачи с лабиринтами
- Вход: представление лабиринта
- Выход: решение лабиринта

## ⚠️ Важно: Модель НЕ является LLM

**Модель НЕ понимает текстовые вопросы!** Это не ChatGPT или другой языковой моделью.

- ❌ Нельзя задать вопрос текстом: "Реши судоку..."
- ❌ Не понимает естественный язык
- ✅ Работает только с **структурированными данными** (сетки, матрицы)
- ✅ Требует данные в **специфическом формате**

## Как работает инференс?

### Процесс инференса:

1. **Инициализация состояния:**
   ```python
   carry = model.initial_carry(batch)
   ```

2. **Рекурсивное рассуждение:**
   ```python
   while not all_finish:
       carry, loss, metrics, preds, all_finish = model(
           carry=carry, 
           batch=batch, 
           return_keys=["preds"]
       )
   ```

3. **Получение предсказаний:**
   - Предсказания находятся в `preds["preds"]`
   - Модель автоматически решает, когда остановиться (ACT)

## Как использовать обученную модель?

### Вариант 1: Использовать встроенную оценку (рекомендуется)

Модель уже имеет встроенную систему оценки:

```bash
# Оценка на тестовом датасете
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
data_paths_test="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
load_checkpoint=checkpoints/your_model.pth \
epochs=1 \
eval_interval=1 \
global_batch_size=4
```

### Вариант 2: Создать скрипт для инференса

Создайте файл `inference.py`:

```python
import torch
import numpy as np
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.losses import ACTLossHead
from utils.functions import load_model_class

# Загрузка модели
def load_model(checkpoint_path: str, config):
    """Загружает обученную модель из чекпоинта."""
    # Создать модель с той же конфигурацией, что и при обучении
    model_cls = load_model_class("trm")
    loss_head_cls = load_model_class("ACTLossHead")
    
    model = model_cls(config)
    model = loss_head_cls(model, **config.loss_config)
    
    # Загрузить веса
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def solve_sudoku(model, sudoku_board: np.ndarray):
    """
    Решает судоку.
    
    Параметры:
        model: Обученная модель
        sudoku_board: Доска судоку 9x9 (0 = пустая клетка, 1-9 = цифры)
    
    Возвращает:
        Решенная доска судоку 9x9
    """
    # Преобразовать в формат батча
    # (нужно преобразовать в последовательность согласно формату датасета)
    batch = {
        "inputs": torch.tensor([sudoku_board.flatten()], dtype=torch.long),
        "labels": torch.zeros((1, 81), dtype=torch.long),  # Заглушка
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long),  # Заглушка
    }
    
    # Инференс
    with torch.no_grad():
        carry = model.initial_carry(batch)
        
        while True:
            carry, loss, metrics, preds, all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=["preds"]
            )
            
            if all_finish:
                break
        
        # Получить предсказание
        solution = preds["preds"][0].cpu().numpy()
        solution = solution.reshape(9, 9)
        
    return solution

# Пример использования
if __name__ == "__main__":
    # Загрузить модель
    # model = load_model("checkpoints/your_model.pth", config)
    
    # Пример судоку (частично заполненная)
    sudoku = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    # Решить
    # solution = solve_sudoku(model, sudoku)
    # print(solution)
```

### Вариант 3: Использовать через pretrain.py с оценкой

Для ARC-AGI можно использовать встроенный оценщик:

```bash
python pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
data_paths_test="[data/arc1concept-aug-1000]" \
evaluators="[arc@ARC]" \
load_checkpoint=checkpoints/your_model.pth \
epochs=1 \
eval_interval=1 \
global_batch_size=4
```

Это создаст файл `submission.json` с предсказаниями.

## Формат данных

### Для Sudoku:
- **Вход:** Массив 9x9, где 0 = пустая клетка, 1-9 = цифры
- **Выход:** Массив 9x9 с решением

### Для ARC-AGI:
- **Вход:** Сетка 30x30 с паттерном
- **Выход:** Сетка 30x30 с решением

### Для Maze:
- **Вход:** Представление лабиринта в формате датасета
- **Выход:** Решение лабиринта

## Примеры использования

### Пример 1: Решение судоку

```python
import numpy as np
import torch

# Создать частично заполненную доску
sudoku_input = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    # ... остальные строки
])

# Преобразовать в формат модели и решить
# (требуется правильная подготовка данных согласно формату датасета)
```

### Пример 2: Оценка на тестовом наборе

```bash
# Запустить оценку на тестовом датасете
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
data_paths_test="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
load_checkpoint=checkpoints/step_1000 \
epochs=1 \
eval_interval=1 \
global_batch_size=8
```

## Ограничения

1. **Специализированная модель:** Работает только с обученными типами задач
2. **Формат данных:** Требует данные в специфическом формате датасета
3. **Не универсальная:** Не может решать произвольные задачи
4. **Требует обучения:** Нужна обученная модель для каждого типа задачи

## Рекомендации

1. **Для быстрого теста:** Используйте встроенную оценку через `pretrain.py`
2. **Для кастомного использования:** Создайте скрипт `inference.py` по примеру выше
3. **Для ARC-AGI:** Используйте встроенный оценщик `arc@ARC`
4. **Для production:** Адаптируйте код под ваши нужды

## Вывод

- ✅ Модель решает **структурированные головоломки** (судоку, ARC, лабиринты)
- ❌ Модель **НЕ понимает текстовые вопросы**
- ✅ Используйте встроенную оценку или создайте скрипт для инференса
- ✅ Требует данные в **специфическом формате** датасета
