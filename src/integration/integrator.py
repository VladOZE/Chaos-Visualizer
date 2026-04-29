# src/integration/integrator.py
"""
Численный интегратор для решения систем обыкновенных дифференциальных уравнений.
Использует методы из SciPy, в частности классический RK4.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable
import warnings

from logger import get_logger


class ODEIntegrator:
    """
    Численный интегратор для систем ОДУ на основе SciPy.
    
    Поддерживает различные методы интегрирования, включая адаптивный RK45 и классический RK4.
    """
    
    def __init__(self, method: str = 'RK45', verbose: bool = False):
        """
        Инициализация интегратора.
        
        Args:
            method: Метод интегрирования ('RK45', 'RK23', 'Radau', 'BDF')
                    RK45 - явный метод 4-5 порядка (рекомендуется)
            verbose: Выводить ли информацию о процессе
        """
        self.method = method
        self.verbose = verbose
        self.supported_methods = ['RK45', 'RK23', 'Radau', 'BDF']
        self.logger = get_logger("integration")
        
        if method not in self.supported_methods:
            raise ValueError(f"Метод {method} не поддерживается. "
                           f"Используйте: {self.supported_methods}")
    
    def integrate(self,
                 dynamics_func: Callable,
                 initial_state: np.ndarray,
                 t_span: Tuple[float, float],
                 t_eval: np.ndarray = None,
                 max_step: float = None,
                 rtol: float = 1e-6,
                 atol: float = 1e-9,
                 dense_output: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Интегрирует систему ОДУ от начального состояния.
        
        Args:
            dynamics_func: Функция, вычисляющая производные f(t, y)
            initial_state: Начальное состояние [x0, y0, z0, ...]
            t_span: (t0, tf) - начальное и конечное время интегрирования
            t_eval: Точки времени для вычисления решения. Если None, используется
                   собственная сетка интегратора
            max_step: Максимальный шаг интегрирования
            rtol: Относительная точность
            atol: Абсолютная точность
            dense_output: Использовать ли плотный вывод (гладкую интерполяцию)
        
        Returns:
            (t, y) - массивы времени и решения shape(n_points, n_vars)
        """
        
        # Валидация начального состояния
        if not np.all(np.isfinite(initial_state)):
            error_msg = "Начальное состояние содержит NaN или бесконечность"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Валидация t_eval
        if t_eval is not None and len(t_eval) == 0:
            error_msg = "t_eval не может быть пустым массивом"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Генерируем сетку для оценки если не задана
            if t_eval is None:
                n_points = 5000
                t_eval = np.linspace(t_span[0], t_span[1], n_points)
            
            # Параметры для solve_ivp
            solve_kwargs = {
                'method': self.method,
                'rtol': rtol,
                'atol': atol,
                'dense_output': dense_output,
                't_eval': t_eval
            }
            
            if max_step is not None:
                solve_kwargs['max_step'] = max_step
            
            # Интегрирование
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution = solve_ivp(
                    dynamics_func,
                    t_span,
                    initial_state,
                    **solve_kwargs
                )
            
            # Проверка успеха интегрирования
            if not solution.success:
                warn_msg = f"Интегрирование завершилось с предупреждением: {solution.message}"
                self.logger.warning(warn_msg)
                if self.verbose:
                    print(warn_msg)
            
            # Убеждаемся, что solution.y - это numpy array, а не список
            y_result = solution.y
            if not isinstance(y_result, np.ndarray):
                y_result = np.array(y_result, dtype=float)
            
            # Обработка edge case: если solution.y имеет shape (n_vars,) вместо (n_vars, n_points)
            # это происходит при одной точке или в некоторых edge cases
            if y_result.ndim == 1:
                y_result = y_result.reshape(-1, 1)
            
            # Убеждаемся, что solution.t - это numpy array
            t_result = solution.t
            if not isinstance(t_result, np.ndarray):
                t_result = np.array(t_result, dtype=float)
            
            # Возвращаем транспонированное решение: (n_points, n_vars)
            # Гарантируем, что результат - это numpy array
            result_t = np.asarray(t_result, dtype=float)
            result_y = np.asarray(y_result.T, dtype=float)
            
            return result_t, result_y
        
        except Exception as e:
            error_msg = f"Ошибка при интегрировании: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def rk4_step(self,
                 f: Callable,
                 t: float,
                 y: np.ndarray,
                 h: float) -> np.ndarray:
        """
        Один шаг классического метода Рунге-Кутты 4-го порядка.
        
        Может использоваться для простых операций, когда не нужна SciPy.
        
        Args:
            f: Функция f(t, y), вычисляющая производные
            t: Текущее время
            y: Текущее состояние
            h: Шаг интегрирования
        
        Returns:
            Новое состояние y(t + h)
        """
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)
        
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


class FixedStepIntegrator:
    """
    Простой интегратор с фиксированным шагом, использующий RK4.
    Полезен для быстрых расчётов и валидации.
    """
    
    def __init__(self, step_size: float = 0.01):
        """
        Args:
            step_size: Размер шага интегрирования
        """
        self.step_size = step_size
    
    def integrate(self,
                 dynamics_func: Callable,
                 initial_state: np.ndarray,
                 num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Интегрирует с фиксированным шагом RK4.
        
        Args:
            dynamics_func: f(t, y) -> dy/dt
            initial_state: Начальное состояние
            num_steps: Количество шагов
        
        Returns:
            (t_array, solution_array) - массивы времени и решения
        """
        
        t_array = np.zeros(num_steps)
        solution = np.zeros((num_steps, len(initial_state)))
        
        solution[0] = initial_state
        t = 0.0
        
        for i in range(1, num_steps):
            # RK4 шаг
            y = solution[i-1]
            k1 = dynamics_func(t, y)
            k2 = dynamics_func(t + self.step_size/2, y + self.step_size*k1/2)
            k3 = dynamics_func(t + self.step_size/2, y + self.step_size*k2/2)
            k4 = dynamics_func(t + self.step_size, y + self.step_size*k3)
            
            solution[i] = y + (self.step_size/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += self.step_size
            t_array[i] = t
        
        return t_array, solution
