# src/models/base.py
"""
Абстрактный базовый класс для всех динамических систем.
Обеспечивает унифицированный интерфейс для различных моделей хаотических систем.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np


class DynamicalSystem(ABC):
    """
    Абстрактный базовый класс для динамической системы.
    
    Любая новая модель должна наследоваться от этого класса и реализовать
    метод compute_derivatives().
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Инициализация системы.
        
        Args:
            name: Имя системы (например, "Lorenz")
            description: Описание системы
        """
        self.name = name
        self.description = description
        self.parameters: Dict[str, float] = {}
        self.default_parameters: Dict[str, float] = {}
        self.parameter_ranges: Dict[str, Tuple[float, float]] = {}
    
    @abstractmethod
    def compute_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Вычисляет производные для системы дифференциальных уравнений.
        
        Args:
            t: Текущее время (может не использоваться в автономных системах)
            state: Текущее состояние системы [x, y, z, ...]
        
        Returns:
            Массив производных [dx/dt, dy/dt, dz/dt, ...]
        """
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> bool:
        """
        Устанавливает параметры системы с проверкой допустимости.
        
        Args:
            **kwargs: Параметры системы
        
        Returns:
            True если все параметры валидны, False иначе
        """
        pass
    
    def get_parameters(self) -> Dict[str, float]:
        """Возвращает текущие параметры системы."""
        return self.parameters.copy()
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Возвращает допустимые диапазоны параметров."""
        return self.parameter_ranges.copy()
    
    def validate_state(self, state: np.ndarray) -> bool:
        """
        Проверяет валидность начального состояния.
        По умолчанию проверяет, что все значения конечны.
        """
        return np.all(np.isfinite(state))
    
    def get_default_parameters(self) -> Dict[str, float]:
        """Возвращает параметры по умолчанию."""
        return self.default_parameters.copy()
    
    def reset_to_defaults(self):
        """Сбрасывает параметры на значения по умолчанию."""
        self.parameters = self.get_default_parameters()
    
    def __repr__(self) -> str:
        return f"<{self.name}: {self.description}>"
