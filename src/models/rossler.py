# src/models/rossler.py
"""
Заготовка для реализации аттрактора Рёсслера.

Аттрактор Рёсслера - ещё один классический пример странного аттрактора.
Уравнения:
    dx/dt = -y - z
    dy/dt = x + ay
    dz/dt = b + z(x - c)
"""

import numpy as np
from .base import DynamicalSystem
from logger import get_logger


class RosslerSystem(DynamicalSystem):
    """
    Реализация аттрактора Рёсслера.
    
    Более простой по структуре странный аттрактор в сравнении с Лоренцем.
    """
    
    def __init__(self):
        super().__init__(
            name="Рёсслер",
            description="Аттрактор Рёсслера"
        )
        self.logger = get_logger("model_rossler")
        
        # Параметры по умолчанию
        self.default_parameters = {
            'a': 0.1,
            'b': 0.1,
            'c': 14.0
        }
        
        # Допустимые диапазоны
        self.parameter_ranges = {
            'a': (0.01, 1.0),
            'b': (0.01, 1.0),
            'c': (1.0, 30.0)
        }
        
        self.parameters = self.default_parameters.copy()
    
    def compute_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Вычисляет производные для системы Рёсслера.
        
        Args:
            t: время
            state: [x, y, z]
        
        Returns:
            [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        
        dx_dt = -y - z
        dy_dt = x + a * y
        dz_dt = b + z * (x - c)
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def set_parameters(self, **kwargs) -> bool:
        """Устанавливает параметры системы с проверкой."""
        temp_params = self.parameters.copy()
        
        try:
            for key, value in kwargs.items():
                if key not in self.parameters:
                    self.logger.error(f"Неизвестный параметр: {key}")
                    return False
                
                value = float(value)
                min_val, max_val = self.parameter_ranges[key]
                
                if not (min_val <= value <= max_val):
                    self.logger.error(
                        f"Параметр {key}={value} вне диапазона [{min_val}, {max_val}]"
                    )
                    return False
                
                self.parameters[key] = value
            
            return True
        
        except (ValueError, TypeError) as exc:
            self.logger.error(f"Ошибка при установке параметров: {exc}")
            self.parameters = temp_params
            return False

    def get_info(self) -> str:
        """Возвращает информацию о системе и текущих параметрах."""
        info = f"""
Система: {self.name}
{self.description}

Уравнения:
  dx/dt = -y - z
  dy/dt = x + a*y
  dz/dt = b + z*(x - c)

Текущие параметры:
  a = {self.parameters['a']:.4f}   (диапазон {self.parameter_ranges['a']})
  b = {self.parameters['b']:.4f}   (диапазон {self.parameter_ranges['b']})
  c = {self.parameters['c']:.4f}   (диапазон {self.parameter_ranges['c']})
        """
        return info
