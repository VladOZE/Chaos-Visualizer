"""
Реализация системы Чуа (Chua circuit).

Система дифференциальных уравнений:
    dx/dt = a*(y - x - h(x))
    dy/dt = x - y + z
    dz/dt = -b*y
    
где h(x) - нелинейная функция Чуа
"""

import numpy as np
from .base import DynamicalSystem
from logger import get_logger


class ChuaSystem(DynamicalSystem):
    """Хаотическая система Чуа (Chua's circuit)."""

    def __init__(self):
        super().__init__(
            name="Чуа",
            description="Система Чуа (хаотическая электрическая цепь)"
        )
        self.logger = get_logger("model_chua")
        
        # Стандартные параметры для хаотического режима
        self.default_parameters = {
            "a": 11.0,
            "b": 14.0,
            "m0": -0.71,
            "m1": -0.46,
        }
        
        self.parameter_ranges = {
            "a": (0.1, 50.0),
            "b": (1.0, 100.0),
            "m0": (-5.0, 0.0),
            "m1": (-2.0, 0.0),
        }
        
        self.parameters = self.default_parameters.copy()

    def _h(self, x):
        """
        Нелинейная функция Чуа h(x).
        Кусочно-линейная функция:
            h(x) = m1*x + (m0 - m1)*sgn(x) * ((|x| - 1)/2) если |x| > 1
            h(x) = m0*x если |x| <= 1
        """
        m0 = self.parameters["m0"]
        m1 = self.parameters["m1"]
        
        if isinstance(x, np.ndarray):
            h = np.zeros_like(x, dtype=float)
            mask_small = np.abs(x) <= 1.0
            mask_large = np.abs(x) > 1.0
            
            h[mask_small] = m0 * x[mask_small]
            
            # Для |x| > 1
            x_large = x[mask_large]
            sign = np.sign(x_large)
            h[mask_large] = m1 * x_large + (m0 - m1) * sign * (np.abs(x_large) - 1.0)
            
            return h
        else:
            x = float(x)
            if np.abs(x) <= 1.0:
                return m0 * x
            else:
                sign = np.sign(x)
                return m1 * x + (m0 - m1) * sign * (np.abs(x) - 1.0)

    def compute_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Вычисляет производные для системы Чуа.
        
        Args:
            t: Текущее время (не используется)
            state: [x, y, z]
        
        Returns:
            [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state[0], state[1], state[2]
        a = self.parameters["a"]
        b = self.parameters["b"]
        
        h_x = self._h(x)
        
        dx = a * (y - x - h_x)
        dy = x - y + z
        dz = -b * y
        
        return np.array([dx, dy, dz], dtype=float)

    def set_parameters(self, **kwargs) -> bool:
        """Устанавливает параметры системы с валидацией."""
        temp = self.parameters.copy()
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
            self.parameters = temp
            return False
