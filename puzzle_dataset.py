"""
Модуль для работы с датасетом головоломок.
Реализует итерируемый датасет для обучения моделей на головоломках.
"""
import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    """
    Формирует батч примеров из головоломок.
    
    Параметры:
        rng: Генератор случайных чисел
        group_order: Массив порядка групп для выборки
        puzzle_indices: Индексы начала каждой головоломки
        group_indices: Индексы начала каждой группы
        start_index: Начальный индекс в group_order
        global_batch_size: Размер глобального батча
    
    Возвращает:
        Кортеж (новый_start_index, batch_indices, batch_puzzle_indices):
        - новый_start_index: Индекс для следующего батча
        - batch_indices: Индексы примеров в батче
        - batch_puzzle_indices: Индексы головоломок для каждого примера
    """
    # Упаковка примеров в полный батч
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Выбираем группу и головоломку из этой группы
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Получаем диапазон головоломки
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Добавляем в батч
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    """
    Конфигурация для датасета головоломок.
    """
    seed: int  # Семя для генератора случайных чисел
    dataset_paths: List[str]  # Список путей к датасетам
    global_batch_size: int  # Глобальный размер батча
    test_set_mode: bool  # Режим тестового набора (True) или обучающего (False)
    epochs_per_iter: int  # Количество эпох на итерацию (для уменьшения накладных расходов)
    rank: int  # Ранг процесса в распределенном обучении
    num_replicas: int  # Количество реплик в распределенном обучении

class PuzzleDataset(IterableDataset):
    """
    Итерируемый датасет для головоломок.
    Поддерживает распределенное обучение и ленивую загрузку данных.
    """
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        """
        Параметры:
            config: Конфигурация датасета
            split: Раздел датасета ('train' или 'test')
        """
        super().__init__()
        self.config = config
        self.split = split

        # Объединение метаданных из нескольких датасетов
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # Проверки
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # Состояние
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        """
        Загружает метаданные датасета из JSON файла.
        
        Параметры:
            dataset_path: Путь к директории датасета
        
        Возвращает:
            Метаданные датасета
        """
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        """
        Ленивая загрузка данных датасета (загружается только при первом обращении).
        Использует memory-mapped файлы для больших массивов.
        """
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Храним индексы в памяти
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Загрузка данных
        self._data = {}
        for set_name in self.metadata.sets:  # Загрузка подмножества
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }


    def _collate_batch(self, batch):
        """
        Обрабатывает батч данных: преобразует типы, заменяет игнорируемые метки и добавляет padding.
        
        Параметры:
            batch: Словарь с данными батча
        
        Возвращает:
            Словарь с обработанными тензорами PyTorch
        """
        # Преобразование типов
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Преобразование идентификаторов игнорируемых меток
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Добавление padding
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # Преобразование в тензоры
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        """
        Итератор для тестового режима.
        Загружает примеры последовательно без перемешивания.
        
        Yields:
            Кортеж (set_name, batch, global_batch_size):
            - set_name: Название набора данных
            - batch: Батч данных
            - global_batch_size: Эффективный размер глобального батча
        """
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Загрузка примеров по одному
            start_index = 0
            while start_index < total_examples:
                # Вычисление индексов
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Получение батча примеров и идентификаторов головоломок
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index
                
                # Переход к следующему батчу
                start_index += self.config.global_batch_size

    def _iter_train(self):
        """
        Итератор для обучающего режима.
        Случайно перемешивает группы головоломок и формирует батчи.
        
        Yields:
            Кортеж (set_name, batch, global_effective_batch_size):
            - set_name: Название набора данных
            - batch: Батч данных
            - global_effective_batch_size: Эффективный размер глобального батча
        """
        for set_name, dataset in self._data.items():  # type: ignore
            # Увеличение счетчика эпох
            self._iters += 1

            # Случайное перемешивание групп
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Выбор текущего ранга и формирование батча
                global_effective_batch_size = batch_puzzle_indices.size  # Глобальный эффективный размер батча, исключая padding

                # Пропуск последнего неполного батча
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        """
        Основной итератор датасета.
        Выбирает режим итерации в зависимости от конфигурации.
        
        Yields:
            Батчи данных в зависимости от режима (train/test)
        """
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Итерация с использованием указанного режима
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()

