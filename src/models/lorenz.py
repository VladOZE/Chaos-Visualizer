# src/models/lorenz.py
"""
Реализация классической системы Лоренца.

Система Лоренца описывается следующей системой ОДУ:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

где σ (sigma) - число Прандтля, ρ (rho) - число Рэлея, β (beta) - параметр геометрии.
"""

import numpy as np
from .base import DynamicalSystem
from logger import get_logger


class LorenzSystem(DynamicalSystem):
    """
    Реализация системы Лоренца.
    
    Классический пример хаотической системы, описывающей упрощённую модель
    атмосферной конвекции.
    """
    
    def __init__(self):
        super().__init__(
            name="Лоренц",
            description="Система Лоренца (атмосферная конвекция)"
        )
        self.logger = get_logger("model_lorenz")
        
        # Параметры по умолчанию - классический хаотический режим
        self.default_parameters = {
            'sigma': 10.0,      # Число Прандтля
            'rho': 28.0,        # Число Рэлея (параметр интенсивности нагрева)
            'beta': 8.0/3.0     # Параметр геометрии
        }
        
        # Допустимые диапазоны параметров
        self.parameter_ranges = {
            'sigma': (0.1, 100.0),
            'rho': (0.1, 100.0),
            'beta': (0.1, 50.0)
        }
        
        self.parameters = self.default_parameters.copy()
    
    def compute_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Вычисляет производные для системы Лоренца.
        
        Args:
            t: время (не используется в данной системе)
            state: [x, y, z] - текущее состояние
        
        Returns:
            [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        sigma = self.parameters['sigma']
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def set_parameters(self, **kwargs) -> bool:
        """
        Устанавливает параметры системы с проверкой.
        
        Args:
            sigma: Число Прандтля (должно быть > 0)
            rho: Число Рэлея (должно быть > 0)
            beta: Параметр геометрии (должен быть > 0)
        
        Returns:
            True если все параметры валидны, False иначе
        """
        temp_params = self.parameters.copy()
        
        try:
            for key, value in kwargs.items():
                if key not in self.parameters:
                    self.logger.error(f"Неизвестный параметр: {key}")
                    return False
                
                # Проверка на числовой тип и допустимый диапазон
                value = float(value)
                min_val, max_val = self.parameter_ranges[key]
                
                if not (min_val <= value <= max_val):
                    self.logger.error(
                        f"Параметр {key}={value} вне диапазона [{min_val}, {max_val}]"
                    )
                    return False
                
                self.parameters[key] = value
            
            return True
        
        except (ValueError, TypeError) as e:
            self.logger.error(f"Ошибка при установке параметров: {e}")
            self.parameters = temp_params
            return False
    
    def get_info(self) -> str:
        """Возвращает информацию о системе и текущих параметрах."""
        info = f"""
Система: {self.name}
{self.description}

Уравнения:
  dx/dt = σ(y - x)
  dy/dt = x(ρ - z) - y
  dz/dt = xy - βz

Текущие параметры:
  σ (sigma) = {self.parameters['sigma']:.4f}  (Прандтль, диапазон {self.parameter_ranges['sigma']})
  ρ (rho)   = {self.parameters['rho']:.4f}    (Рэлей, диапазон {self.parameter_ranges['rho']})
  β (beta)  = {self.parameters['beta']:.4f}   (геометрия, диапазон {self.parameter_ranges['beta']})
        """
        return info
