"""
Модуль с базовыми слоями нейронной сети: линейные слои, эмбеддинги, внимание и нормализация.
"""
from typing import Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F

# Попытка импорта Flash Attention (закомментировано)
#try:
#    from flash_attn_interface import flash_attn_func  # type: ignore[import]
#except ImportError:
#    # Резервный вариант - FlashAttention 2
#    from flash_attn import flash_attn_func  # type: ignore[import]
from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]  # Тип для косинуса и синуса позиционных эмбеддингов


def _find_multiple(a, b):
    """
    Находит наименьшее число, кратное b, которое >= a.
    
    Параметры:
        a: Исходное число
        b: Множитель
    
    Возвращает:
        Наименьшее кратное b, которое >= a
    """
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """
    Поворачивает половину скрытых размерностей входного тензора.
    Используется для RoPE (Rotary Position Embedding).
    
    Параметры:
        x: Входной тензор
    
    Возвращает:
        Тензор с повернутыми половинами размерностей
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Применяет ротационные позиционные эмбеддинги (RoPE) к запросам и ключам.
    
    Параметры:
        q: Тензор запросов [batch_size, seq_len, num_heads, head_dim]
        k: Тензор ключей [batch_size, seq_len, num_heads, head_dim]
        cos: Косинусы позиционных эмбеддингов [seq_len, head_dim]
        sin: Синусы позиционных эмбеддингов [seq_len, head_dim]
    
    Возвращает:
        Кортеж (q_embed, k_embed) - запросы и ключи с примененными позиционными эмбеддингами
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    """
    Линейный слой с приведением типов весов к типу входных данных.
    Использует усеченную нормальную инициализацию LeCun.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        """
        Параметры:
            in_features: Размерность входных признаков
            out_features: Размерность выходных признаков
            bias: Использовать ли смещение (bias)
        """
        super().__init__()
        # Усеченная нормальная инициализация LeCun
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Инициализация смещения нулями
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через линейный слой.
        
        Параметры:
            input: Входной тензор
        
        Возвращает:
            Результат линейного преобразования
        """
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    """
    Слой эмбеддингов с приведением типов весов к указанному типу данных.
    Использует усеченную нормальную инициализацию LeCun.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        """
        Параметры:
            num_embeddings: Количество эмбеддингов (размер словаря)
            embedding_dim: Размерность эмбеддингов
            init_std: Стандартное отклонение для инициализации
            cast_to: Тип данных, к которому приводятся веса
        """
        super().__init__()
        self.cast_to = cast_to

        # Усеченная нормальная инициализация LeCun
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Получает эмбеддинги для входных индексов.
        
        Параметры:
            input: Тензор с индексами токенов
        
        Возвращает:
            Эмбеддинги для указанных индексов
        """
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """
    Модуль для генерации ротационных позиционных эмбеддингов (RoPE).
    """
    def __init__(self, dim, max_position_embeddings, base, device=None):
        """
        Параметры:
            dim: Размерность эмбеддингов
            max_position_embeddings: Максимальная длина последовательности
            base: Базовое значение для вычисления частот
            device: Устройство для размещения тензоров
        """
        super().__init__()

        # Вычисление обратных частот для RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Отличается от статьи, но использует другую перестановку для получения того же результата
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        """
        Возвращает кэшированные косинусы и синусы позиционных эмбеддингов.
        
        Возвращает:
            Кортеж (cos, sin) позиционных эмбеддингов
        """
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """
    Модуль механизма внимания (self-attention) с поддержкой RoPE и масштабированного скалярного произведения.
    """
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        """
        Параметры:
            hidden_size: Размерность скрытого состояния
            head_dim: Размерность одной головы внимания
            num_heads: Количество голов внимания для запросов
            num_key_value_heads: Количество голов внимания для ключей и значений
            causal: Использовать ли каузальную маску (для автогрессивных моделей)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через механизм внимания.
        
        Параметры:
            cos_sin: Кортеж (cos, sin) для RoPE или None
            hidden_states: Входные скрытые состояния [batch_size, seq_len, hidden_size]
        
        Возвращает:
            Выходные скрытые состояния после применения внимания
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Проекция в пространство запросов, ключей и значений
        qkv = self.qkv_proj(hidden_states)

        # Разделение на головы
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # Применение ротационных позиционных эмбеддингов
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Масштабированное скалярное произведение внимания
        # Перестановка нужна для scaled_dot_product_attention, но не для flash_attn_func
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value))
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    """
    Линейный слой с активацией SiLU (Swish).
    Может применяться в двух порядках: Linear(SiLU(x)) или SiLU(Linear(x)).
    """
    def __init__(self, hidden_size: int, reverse=False):
        """
        Параметры:
            hidden_size: Размерность скрытого состояния
            reverse: Если True, применяет SiLU после линейного слоя, иначе до
        """
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        """
        Прямой проход через слой.
        
        Параметры:
            x: Входной тензор
        
        Возвращает:
            Результат применения линейного слоя и SiLU
        """
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    """
    Модуль SwiGLU (Swish-Gated Linear Unit) - активация с воротами.
    Используется в качестве feed-forward слоя в трансформерах.
    """
    def __init__(self, hidden_size: int, expansion: float):
        """
        Параметры:
            hidden_size: Размерность скрытого состояния
            expansion: Коэффициент расширения внутренней размерности
        """
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        """
        Прямой проход через SwiGLU.
        
        Параметры:
            x: Входной тензор
        
        Возвращает:
            Результат применения SwiGLU: down_proj(SiLU(gate) * up)
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """
    Применяет RMS нормализацию (Root Mean Square Normalization) к тензору.
    
    Параметры:
        hidden_states: Входной тензор для нормализации
        variance_epsilon: Малое значение для численной стабильности
    
    Возвращает:
        Нормализованный тензор
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
