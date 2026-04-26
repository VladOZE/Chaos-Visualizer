"""
Пользовательская система ОДУ с уравнениями, заданными строками.
"""

import math
import numpy as np
from .base import DynamicalSystem
from logger import get_logger


class CustomEquationSystem(DynamicalSystem):
    """Система, в которой пользователь задаёт dx/dt, dy/dt, dz/dt."""

    def __init__(self):
        super().__init__(
            name="Пользовательская",
            description="Пользовательские уравнения для интегрирования"
        )
        self.logger = get_logger("model_custom")
        self.default_parameters = {}
        self.parameter_ranges = {}
        self.parameters = {}
        self.equations = {
            "dx": "10*(y-x)",
            "dy": "x*(28-z)-y",
            "dz": "x*y-2.6666666667*z",
        }
        self._safe_env = {
            "np": np,
            "math": math,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "pi": np.pi,
        }

    def set_equations(self, dx_expr: str, dy_expr: str, dz_expr: str) -> bool:
        try:
            self.equations = {
                "dx": dx_expr.strip(),
                "dy": dy_expr.strip(),
                "dz": dz_expr.strip(),
            }
            if not all(self.equations.values()):
                raise ValueError("Уравнения не должны быть пустыми")
            # Пробная проверка выражений.
            self.compute_derivatives(0.0, np.array([0.1, 0.1, 0.1], dtype=float))
            return True
        except Exception as exc:
            self.logger.error(f"Ошибка в пользовательских уравнениях: {exc}")
            return False

    def compute_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state[0], state[1], state[2]
        local_env = {"t": t, "x": x, "y": y, "z": z}
        try:
            dx = eval(self.equations["dx"], {"__builtins__": {}}, {**self._safe_env, **local_env})
            dy = eval(self.equations["dy"], {"__builtins__": {}}, {**self._safe_env, **local_env})
            dz = eval(self.equations["dz"], {"__builtins__": {}}, {**self._safe_env, **local_env})
            result = np.array([float(dx), float(dy), float(dz)], dtype=float)
            if not np.all(np.isfinite(result)):
                raise ValueError("Уравнения вернули NaN/Inf")
            return result
        except Exception as exc:
            raise ValueError(f"Некорректные пользовательские уравнения: {exc}") from exc

    def set_parameters(self, **kwargs) -> bool:
        # Параметры задаются непосредственно в формулах.
        return True
