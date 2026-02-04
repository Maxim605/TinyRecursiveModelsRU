"""
Утилиты для загрузки классов моделей и получения путей к исходным файлам.
"""
import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    """
    Загружает класс модели по идентификатору в формате 'module_path@ClassName'.
    
    Параметры:
        identifier: Идентификатор модели в формате 'module_path@ClassName'
                   (например, 'recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1')
        prefix: Префикс для пути модуля (по умолчанию 'models.')
    
    Возвращает:
        Класс модели
    """
    module_path, class_name = identifier.split('@')

    # Импорт модуля
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    """
    Получает путь к исходному файлу модуля модели.
    
    Параметры:
        identifier: Идентификатор модели в формате 'module_path@ClassName'
        prefix: Префикс для пути модуля (по умолчанию 'models.')
    
    Возвращает:
        Путь к исходному файлу модуля
    """
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)
