"""
Модуль для экспоненциального скользящего среднего (EMA) весов модели.
"""
import copy
import torch.nn as nn

class EMAHelper(object):
    """
    Вспомогательный класс для управления экспоненциальным скользящим средним весов модели.
    EMA помогает стабилизировать обучение и улучшить качество модели.
    """
    def __init__(self, mu=0.999):
        """
        Параметры:
            mu: Коэффициент EMA (близок к 1.0, обычно 0.999)
        """
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        """
        Регистрирует параметры модуля для отслеживания EMA.
        
        Параметры:
            module: Модуль PyTorch для регистрации
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """
        Обновляет EMA весов на основе текущих весов модуля.
        
        Параметры:
            module: Модуль PyTorch для обновления EMA
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """
        Применяет EMA веса к модулю (заменяет текущие веса на EMA).
        
        Параметры:
            module: Модуль PyTorch для применения EMA весов
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """
        Создает копию модуля с примененными EMA весами.
        
        Параметры:
            module: Модуль PyTorch для копирования
        
        Возвращает:
            Копия модуля с EMA весами
        """
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        """
        Возвращает словарь состояния EMA (тени весов).
        
        Возвращает:
            Словарь с EMA весами
        """
        return self.shadow

    def load_state_dict(self, state_dict):
        """
        Загружает словарь состояния EMA.
        
        Параметры:
            state_dict: Словарь с EMA весами для загрузки
        """
        self.shadow = state_dict

