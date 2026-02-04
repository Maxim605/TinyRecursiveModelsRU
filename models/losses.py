"""
Модуль с функциями потерь для обучения моделей.
"""
from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100  # Идентификатор для игнорируемых меток


def s(x, epsilon=1e-30):
    """
    Вспомогательная функция для стабильного вычисления softmax.
    
    Параметры:
        x: Входной тензор
        epsilon: Малое значение для численной стабильности
    
    Возвращает:
        Преобразованный тензор
    """
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    """
    Вычисляет логарифм стабильного softmax.
    
    Параметры:
        x: Входной тензор с логитами
        dim: Размерность для применения softmax
    
    Возвращает:
        Логарифм вероятностей
    """
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    """
    Вычисляет кросс-энтропию с использованием стабильного softmax.
    
    Параметры:
        logits: Выходные логиты модели
        labels: Истинные метки
        ignore_index: Индекс для игнорируемых меток
        valid_mask: Маска валидных элементов (если None, вычисляется автоматически)
    
    Возвращает:
        Тензор потерь для каждого элемента
    """
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    """
    Вычисляет стандартную кросс-энтропию с softmax.
    
    Параметры:
        logits: Выходные логиты модели
        labels: Истинные метки
        ignore_index: Индекс для игнорируемых меток
    
    Возвращает:
        Тензор потерь для каждого элемента
    """
    # Приведение логитов к float32
    # Развертывание логитов
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    """
    Голова потерь для ACT (Adaptive Computation Time) модели.
    Вычисляет потери для языкового моделирования и Q-обучения для остановки.
    """
    def __init__(self, model: nn.Module, loss_type: str):
        """
        Параметры:
            model: Базовая модель для обертки
            loss_type: Тип функции потерь ('stablemax_cross_entropy' или 'softmax_cross_entropy')
        """
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        """
        Инициализирует состояние переноса модели.
        
        Возвращает:
            Начальное состояние переноса
        """
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Прямой проход через модель с вычислением потерь.
        
        Параметры:
            return_keys: Ключи выходных данных для возврата
            **model_kwargs: Аргументы для модели (carry, batch и т.д.)
        
        Возвращает:
            Кортеж (new_carry, loss, metrics, outputs, all_finish):
            - new_carry: Новое состояние переноса
            - loss: Общая функция потерь
            - metrics: Словарь метрик
            - outputs: Выходные данные (только запрошенные ключи)
            - all_finish: Флаг, что все последовательности завершены
        """
        # Логиты модели
        # Размерность: B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Предсказания
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Корректность
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Избегаем деления на ноль

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Метрики (только для остановленных последовательностей)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Потери

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (потеря бутстраппинга для целевого Q); Alexia: Это соответствует Q-learning, но кажется совершенно ненужным
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Фильтрация выходных данных для возврата
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

