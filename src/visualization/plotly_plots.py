# src/visualization/plotly_plots.py
"""
Интерактивная 3D визуализация с использованием Plotly.
Позволяет вращать, масштабировать и экспортировать 3D графики.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional


class PlotlyVisualizer:
    """Класс для интерактивной 3D визуализации с Plotly."""
    
    @staticmethod
    def plot_3d_attractor(trajectory: np.ndarray,
                         title: str = "3D аттрактор",
                         show_axes: bool = True,
                         colormap: str = 'Viridis',
                         save_path: Optional[str] = None) -> go.Figure:
        """
        Создаёт интерактивный 3D график аттрактора.
        
        Args:
            trajectory: shape (n_points, 3) или больше
            title: Название графика
            show_axes: Показывать ли оси координат
            colormap: Цветовая схема Plotly
            save_path: Путь для сохранения (HTML)
        
        Returns:
            Plotly Figure объект
        """
        x = trajectory[:, 0]
        y = trajectory[:, 1] if trajectory.shape[1] > 1 else np.zeros_like(x)
        z = trajectory[:, 2] if trajectory.shape[1] > 2 else np.zeros_like(x)
        
        # Раскраска по времени
        t = np.arange(len(x))
        
        fig = go.Figure()
        
        # Основная траектория
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                color=t,
                colorscale=colormap,
                width=2,
                showscale=True,
                colorbar=dict(title="Время шага")
            ),
            name='Траектория',
            hoverinfo='x+y+z'
        ))
        
        # Маркер начала траектории
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers+text',
            marker=dict(size=10, color='green', symbol='circle'),
            text=['START'],
            textposition='top center',
            name='Точка начала',
            hoverinfo='text',
            showlegend=True
        ))
        
        # Маркер конца траектории
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='x'),
            text=['END'],
            textposition='top center',
            name='Точка окончания',
            hoverinfo='text',
            showlegend=True
        ))
        
        # Обновляем layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1000,
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_2d_scatter(trajectory: np.ndarray,
                       x_idx: int = 0,
                       y_idx: int = 1,
                       title: str = "2D фазовый портрет",
                       colormap: str = 'Viridis',
                       save_path: Optional[str] = None) -> go.Figure:
        """
        Создаёт интерактивный 2D scatter plot с раскраской по времени.
        
        Args:
            trajectory: shape (n_points, n_vars)
            x_idx, y_idx: Индексы координат
            title: Название
            colormap: Цветовая схема
            save_path: Путь для сохранения
        
        Returns:
            Plotly Figure
        """
        x = trajectory[:, x_idx]
        y = trajectory[:, y_idx]
        t = np.arange(len(x))
        
        fig = go.Figure()
        
        # Scatter с раскраской по времени
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=4,
                color=t,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(title="Время шага"),
                opacity=0.7
            ),
            text=[f"t={ti}<br>x={xi:.2f}<br>y={yi:.2f}" 
                  for ti, xi, yi in zip(t, x, y)],
            hoverinfo='text',
            name='Траектория'
        ))
        
        # Начало и конец
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='circle'),
            text=['START'],
            textposition='top center',
            name='Начало',
            hoverinfo='text'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='x'),
            text=['END'],
            textposition='top center',
            name='Окончание',
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title=f'x[{x_idx}]',
            yaxis_title=f'x[{y_idx}]',
            width=900,
            height=700,
            hovermode='closest',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def export_to_formats(fig: go.Figure,
                         base_path: str,
                         formats: list = ['html', 'png']) -> dict:
        """
        Экспортирует Plotly Figure в различные форматы.
        
        Args:
            fig: Plotly Figure
            base_path: Путь без расширения
            formats: Список форматов ('html', 'png', 'jpg', 'svg')
        
        Returns:
            Словарь {формат: путь_к_файлу}
        """
        results = {}
        
        for fmt in formats:
            try:
                file_path = f"{base_path}.{fmt}"
                if fmt == 'html':
                    fig.write_html(file_path)
                else:
                    fig.write_image(file_path)
                results[fmt] = file_path
            except Exception as e:
                print(f"Ошибка экспорта в {fmt}: {e}")
        
        return results
