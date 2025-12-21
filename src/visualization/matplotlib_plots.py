# src/visualization/matplotlib_plots.py
"""
2D визуализация с использованием Matplotlib.
Включает фазовые портреты и графики эволюции во времени.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from typing import Tuple, Optional


class MatplotlibPlotter:
    """Класс для создания 2D графиков с Matplotlib."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        """
        Args:
            figsize: Размер фигуры (width, height) в дюймах
            dpi: Разрешение
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_3d_projection(self,
                          trajectory: np.ndarray,
                          title: str = "3D фазовый портрет",
                          save_path: Optional[str] = None) -> Figure:
        """
        Рисует 3D проекции траектории на 2D плоскостях (x-y, x-z, y-z).
        
        Args:
            trajectory: shape (n_points, 3) или (n_points, n_vars)
            title: Название графика
            save_path: Путь для сохранения (опционально)
        
        Returns:
            Figure объект Matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Извлекаем координаты
        x = trajectory[:, 0]
        y = trajectory[:, 1] if trajectory.shape[1] > 1 else None
        z = trajectory[:, 2] if trajectory.shape[1] > 2 else None
        
        # x-y фазовый портрет
        axes[0, 0].plot(x, y, 'b-', linewidth=0.5, alpha=0.8)
        axes[0, 0].scatter(x[0], y[0], c='green', s=100, marker='o', label='Начало', zorder=5)
        axes[0, 0].scatter(x[-1], y[-1], c='red', s=100, marker='x', label='Конец', zorder=5)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Фазовый портрет: x-y')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # x-z фазовый портрет
        if z is not None:
            axes[0, 1].plot(x, z, 'g-', linewidth=0.5, alpha=0.8)
            axes[0, 1].scatter(x[0], z[0], c='green', s=100, marker='o', label='Начало', zorder=5)
            axes[0, 1].scatter(x[-1], z[-1], c='red', s=100, marker='x', label='Конец', zorder=5)
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('z')
            axes[0, 1].set_title('Фазовый портрет: x-z')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # y-z фазовый портрет
            axes[1, 0].plot(y, z, 'r-', linewidth=0.5, alpha=0.8)
            axes[1, 0].scatter(y[0], z[0], c='green', s=100, marker='o', label='Начало', zorder=5)
            axes[1, 0].scatter(y[-1], z[-1], c='red', s=100, marker='x', label='Конец', zorder=5)
            axes[1, 0].set_xlabel('y')
            axes[1, 0].set_ylabel('z')
            axes[1, 0].set_title('Фазовый портрет: y-z')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # График эволюции во времени (только x)
        t = np.arange(len(x))
        axes[1, 1].plot(t, x, 'b-', linewidth=0.8, label='x(t)')
        if y is not None:
            axes[1, 1].plot(t, y, 'g-', linewidth=0.8, label='y(t)')
        if z is not None:
            axes[1, 1].plot(t, z, 'r-', linewidth=0.8, label='z(t)')
        axes[1, 1].set_xlabel('Время шага')
        axes[1, 1].set_ylabel('Значение координат')
        axes[1, 1].set_title('Время эволюции')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_2d_phase_portrait(self,
                              trajectory: np.ndarray,
                              x_idx: int = 0,
                              y_idx: int = 1,
                              title: str = "2D фазовый портрет",
                              colormap: str = 'viridis',
                              save_path: Optional[str] = None) -> Figure:
        """
        Рисует 2D фазовый портрет с раскраской по времени.
        
        Args:
            trajectory: shape (n_points, n_vars)
            x_idx, y_idx: Индексы координат для осей
            title: Название
            colormap: Названи цветовой схемы
            save_path: Путь для сохранения
        
        Returns:
            Figure объект
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        x = trajectory[:, x_idx]
        y = trajectory[:, y_idx]
        t = np.arange(len(x))
        
        # Раскраска по времени
        scatter = ax.scatter(x, y, c=t, cmap=colormap, s=2, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Время шага')
        
        # Маркеры начала и конца
        ax.scatter(x[0], y[0], c='green', s=200, marker='o', label='Начало',
                  edgecolors='black', linewidths=1, zorder=10)
        ax.scatter(x[-1], y[-1], c='red', s=200, marker='x', label='Конец',
                  linewidths=2, zorder=10)
        
        ax.set_xlabel(f'$x_{x_idx}$', fontsize=12)
        ax.set_ylabel(f'$x_{y_idx}$', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def get_figure_canvas(figure: Figure, parent=None) -> QWidget:
        """
        Конвертирует Matplotlib Figure в PyQt5 виджет.
        
        Args:
            figure: Matplotlib Figure
            parent: Родительский QWidget
        
        Returns:
            QWidget с встроенным Canvas
        """
        widget = QWidget(parent)
        layout = QVBoxLayout()
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        widget.setLayout(layout)
        return widget
