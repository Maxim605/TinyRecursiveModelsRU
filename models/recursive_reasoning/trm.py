"""
Модуль Tiny Recursive Reasoning Model (TRM) - модель рекурсивного рассуждения.
Реализует рекурсивное улучшение предсказаний с использованием небольшой нейронной сети.
"""
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100  # Идентификатор для игнорируемых меток

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    """
    Внутреннее состояние переноса модели TRM.
    Содержит скрытые состояния для высокого (H) и низкого (L) уровней рекурсии.
    """
    z_H: torch.Tensor  # Скрытое состояние высокого уровня
    z_L: torch.Tensor  # Скрытое состояние низкого уровня


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    """
    Полное состояние переноса модели TRM с ACT (Adaptive Computation Time).
    Содержит внутреннее состояние, счетчики шагов, флаги остановки и текущие данные.
    """
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry  # Внутреннее состояние
    
    steps: torch.Tensor  # Количество выполненных шагов для каждого примера
    halted: torch.Tensor  # Флаги остановки для каждого примера
    
    current_data: Dict[str, torch.Tensor]  # Текущие данные батча


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    """
    Конфигурация модели Tiny Recursive Reasoning Model с ACT.
    """
    batch_size: int  # Размер батча
    seq_len: int  # Длина последовательности
    puzzle_emb_ndim: int = 0  # Размерность эмбеддингов головоломок (0 = отключено)
    num_puzzle_identifiers: int  # Количество идентификаторов головоломок
    vocab_size: int  # Размер словаря

    H_cycles: int  # Количество циклов на высоком уровне рекурсии
    L_cycles: int  # Количество циклов на низком уровне рекурсии

    H_layers: int  # Игнорируется (не используется)
    L_layers: int  # Количество слоев на низком уровне

    # Конфигурация трансформера
    hidden_size: int  # Размерность скрытого состояния
    expansion: float  # Коэффициент расширения для feed-forward слоев
    num_heads: int  # Количество голов внимания
    pos_encodings: str  # Тип позиционных эмбеддингов ('rope', 'learned' или 'none')

    rms_norm_eps: float = 1e-5  # Эпсилон для RMS нормализации
    rope_theta: float = 10000.0  # Базовое значение для RoPE
    
    # Конфигурация Q-learning для остановки
    halt_max_steps: int  # Максимальное количество шагов до принудительной остановки
    halt_exploration_prob: float  # Вероятность исследования для остановки

    forward_dtype: str = "bfloat16"  # Тип данных для прямого прохода

    # Дополнительные параметры (добавлены Alexia)
    mlp_t: bool = False  # Использовать MLP на L вместо трансформера
    puzzle_emb_len: int = 16  # Если не ноль, задается это значение
    no_ACT_continue: bool = True  # Не использовать continue ACT loss, только сигмоиду halt (более логично)

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    """
    Блок модели TRM.
    Может использовать либо self-attention, либо MLP в зависимости от конфигурации.
    """
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        """
        Параметры:
            config: Конфигурация модели
        """
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,  # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через блок.
        
        Параметры:
            cos_sin: Кортеж (cos, sin) для RoPE или None
            hidden_states: Входные скрытые состояния [batch_size, seq_len, hidden_size]
        
        Возвращает:
            Выходные скрытые состояния после применения блока
        """
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Полносвязный слой
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    """
    Модуль рассуждения модели TRM.
    Состоит из нескольких блоков и применяет инъекцию входных данных.
    """
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        """
        Параметры:
            layers: Список блоков для применения
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Прямой проход через модуль рассуждения.
        
        Параметры:
            hidden_states: Скрытые состояния для обработки
            input_injection: Входные данные для инъекции (добавляются к скрытым состояниям)
            **kwargs: Дополнительные аргументы (например, cos_sin для RoPE)
        
        Возвращает:
            Обработанные скрытые состояния
        """
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    """
    Внутренняя модель TRM.
    Реализует основную логику рекурсивного рассуждения с циклами H и L.
    """
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        """
        Параметры:
            config: Конфигурация модели
        """
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Вход/Выход

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Специальная инициализация Q head
        # Инициализация Q почти нулями для ускорения обучения при бутстраппинге
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """
        Создает входные эмбеддинги из токенов и идентификаторов головоломок.
        
        Параметры:
            input: Входные токены
            puzzle_identifiers: Идентификаторы головоломок
        
        Возвращает:
            Эмбеддинги входных данных
        """
        # Эмбеддинги токенов
        embedding = self.embed_tokens(input.to(torch.int32))

        # Эмбеддинги головоломок
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Позиционные эмбеддинги
        if self.config.pos_encodings == "learned":
            # Масштабирование на 1/sqrt(2) для сохранения дисперсии прямого прохода
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Масштабирование
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        """
        Создает пустое состояние переноса.
        
        Параметры:
            batch_size: Размер батча
        
        Возвращает:
            Пустое состояние переноса
        """
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        """
        Сбрасывает состояние переноса для указанных примеров.
        
        Параметры:
            reset_flag: Флаги сброса для каждого примера в батче
            carry: Текущее состояние переноса
        
        Возвращает:
            Новое состояние переноса с сброшенными значениями
        """
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Прямой проход через внутреннюю модель.
        Выполняет рекурсивные циклы H и L для улучшения предсказаний.
        
        Параметры:
            carry: Состояние переноса с z_H и z_L
            batch: Батч данных с inputs и puzzle_identifiers
        
        Возвращает:
            Кортеж (new_carry, output, (q_halt_logits, q_continue_logits)):
            - new_carry: Новое состояние переноса (без градиентов)
            - output: Логиты языковой модели
            - (q_halt_logits, q_continue_logits): Логиты Q-функций для остановки и продолжения
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Кодирование входных данных
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Итерации прямого прохода
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 без градиентов (для экономии памяти)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 с градиентами
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Выходы языковой модели
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # Новое состояние без градиентов
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # Q-head; использует первую позицию puzzle_emb
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """
    Обертка модели TRM с ACT (Adaptive Computation Time).
    Управляет остановкой вычислений на основе Q-learning.
    """

    def __init__(self, config_dict: dict):
        """
        Параметры:
            config_dict: Словарь с конфигурацией модели
        """
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        """
        Возвращает модуль эмбеддингов головоломок.
        """
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """
        Инициализирует начальное состояние переноса для батча.
        
        Параметры:
            batch: Батч данных
        
        Возвращает:
            Начальное состояние переноса (все последовательности остановлены)
        """
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Пустое состояние ожидается, будет сброшено в первом проходе, так как все последовательности остановлены
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # По умолчанию остановлены
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # Если обучение и ACT включен
            if self.training and (self.config.halt_max_steps > 1):

                # Сигнал остановки
                # ПРИМЕЧАНИЕ: Во время оценки всегда используем максимальное количество шагов,
                # это гарантирует одинаковые шаги остановки внутри батча для целей батчинга
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Исследование
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Вычисление целевого Q
                    # ПРИМЕЧАНИЕ: Нет буфера воспроизведения и целевых сетей для вычисления целевого Q-значения.
                    # Так как размер батча большой, есть много параллельных окружений.
                    # Похожая концепция как в PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
