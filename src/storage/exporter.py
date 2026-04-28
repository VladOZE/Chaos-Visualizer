# src/storage/exporter.py
"""
Экспорт результатов моделирования в различные форматы:
- Изображения: PNG, SVG
- Данные: JSON, CSV
- Видео/анимация: MP4, GIF (опционально)
"""

import json
import csv
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from logger import get_logger


class ResultsExporter:
    """Класс для экспорта результатов моделирования."""
    
    def __init__(self, output_dir: str = "data"):
        """
        Args:
            output_dir: Директория для сохранения результатов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = get_logger("exporter")
    
    def save_trajectory_csv(self,
                           trajectory: np.ndarray,
                           filename: str = "trajectory.csv",
                           labels: Optional[List[str]] = None) -> str:
        """
        Сохраняет траекторию в CSV.
        
        Args:
            trajectory: shape (n_points, n_vars)
            filename: Имя файла
            labels: Названия координат (по умолчанию ['x', 'y', 'z', ...])
        
        Returns:
            Полный путь к файлу
        """
        file_path = self.output_dir / filename
        
        if labels is None:
            labels = [f'x{i}' for i in range(trajectory.shape[1])]
        
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(labels)
                writer.writerows(trajectory)
            self.logger.info(f"Траектория сохранена: {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении CSV: {e}")
            return ""

    def save_trajectory_json(self,
                             trajectory: np.ndarray,
                             filename: str = "trajectory.json",
                             labels: Optional[List[str]] = None) -> str:
        """
        Сохраняет траекторию в JSON.
        """
        file_path = self.output_dir / filename
        try:
            data = {
                "labels": labels or [f"x{i}" for i in range(trajectory.shape[1])],
                "trajectory": trajectory.tolist()
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Траектория сохранена (json): {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении JSON траектории: {e}")
            return ""
    
    def save_parameters_json(self,
                            parameters: Dict,
                            system_name: str = "system",
                            initial_conditions: Optional[Dict] = None,
                            filename: Optional[str] = None) -> str:
        """
        Сохраняет параметры эксперимента в JSON.
        
        Args:
            parameters: Словарь параметров системы
            system_name: Имя системы
            initial_conditions: Начальные условия
            filename: Имя файла (по умолчанию f"{system_name}_params.json")
        
        Returns:
            Полный путь к файлу
        """
        if filename is None:
            filename = f"{system_name}_params.json"
        
        file_path = self.output_dir / filename
        
        data = {
            'system': system_name,
            'parameters': parameters,
            'initial_conditions': initial_conditions or {}
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Параметры сохранены: {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении JSON: {e}")
            return ""
    
    def save_figure(self,
                   figure: Figure,
                   filename: str,
                   formats: List[str] = ['png']) -> Dict[str, str]:
        """
        Сохраняет Matplotlib Figure в различные форматы.
        
        Args:
            figure: Matplotlib Figure
            filename: Имя файла без расширения
            formats: Список форматов ('png', 'svg', 'pdf', etc.)
        
        Returns:
            Словарь {формат: полный_путь}
        """
        results = {}
        
        for fmt in formats:
            try:
                file_path = self.output_dir / f"{filename}.{fmt}"
                figure.savefig(str(file_path), dpi=150, bbox_inches='tight')
                self.logger.info(f"Рисунок сохранен: {file_path}")
                results[fmt] = str(file_path)
            except Exception as e:
                self.logger.error(f"Ошибка при сохранении в {fmt}: {e}")
        
        return results
    
    def save_plotly_html(self,
                        plotly_figure,
                        filename: str) -> str:
        """
        Сохраняет Plotly Figure в интерактивный HTML.
        
        Args:
            plotly_figure: Plotly Figure
            filename: Имя файла
        
        Returns:
            Полный путь к файлу
        """
        file_path = self.output_dir / filename
        
        try:
            plotly_figure.write_html(str(file_path))
            self.logger.info(f"HTML сохранён: {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении HTML: {e}")
            return ""
    
    def save_animation(self,
                      trajectory: np.ndarray,
                      filename: str = "animation.gif",
                      interval: int = 50) -> Optional[str]:
        """
        Создаёт анимацию формирования аттрактора (GIF или MP4).
        
        Требует imageio и imageio-ffmpeg.
        
        Args:
            trajectory: shape (n_points, 3)
            filename: Имя файла (расширение определяет формат)
            interval: Интервал между кадрами в миллисекундах
        
        Returns:
            Полный путь к файлу или None при ошибке
        """
        try:
            file_path = self.output_dir / filename
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            step = max(1, len(trajectory) // 240)  # до ~240 кадров
            frame_indices = list(range(1, len(trajectory), step))
            if not frame_indices:
                frame_indices = [len(trajectory) - 1]
            elif frame_indices[-1] != len(trajectory) - 1:
                frame_indices.append(len(trajectory) - 1)

            x = trajectory[:, 0]
            y = trajectory[:, 1] if trajectory.shape[1] > 1 else np.zeros_like(x)
            z = trajectory[:, 2] if trajectory.shape[1] > 2 else np.zeros_like(x)
            line, = ax.plot([], [], [], 'b-', linewidth=0.8, alpha=0.9)
            point = ax.scatter([], [], [], c='red', s=40)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y), np.max(y))
            ax.set_zlim(np.min(z), np.max(z))

            def update(frame_idx):
                segment = trajectory[:frame_idx + 1]
                line.set_data(segment[:, 0], segment[:, 1])
                line.set_3d_properties(segment[:, 2] if segment.shape[1] > 2 else np.zeros(len(segment)))
                point._offsets3d = ([segment[-1, 0]], [segment[-1, 1]], [segment[-1, 2] if segment.shape[1] > 2 else 0.0])
                ax.set_title(f'Attractor Formation (step {frame_idx})')
                return line, point

            anim = FuncAnimation(
                fig,
                update,
                frames=frame_indices,
                interval=interval,
                blit=False,
                repeat=False,
            )

            fps = max(1, 1000 // interval)
            if filename.endswith('.gif'):
                anim.save(str(file_path), writer=PillowWriter(fps=fps))
            elif filename.endswith('.mp4'):
                anim.save(str(file_path), writer=FFMpegWriter(fps=fps))
            else:
                raise ValueError("Поддерживаются только .gif и .mp4")

            plt.close(fig)
            self.logger.info(f"Анимация сохранена: {file_path}")
            return str(file_path)
        
        except ImportError:
            self.logger.error("Для сохранения анимаций требуется pillow (GIF) и ffmpeg (MP4)")
            return None
        except Exception as e:
            self.logger.error(f"Ошибка при создании анимации: {e}")
            return None

    def load_trajectory_csv(self, filepath: str) -> Optional[np.ndarray]:
        """Загружает траекторию из CSV."""
        try:
            arr = np.loadtxt(filepath, delimiter=',', skiprows=1)
            return arr
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке CSV траектории: {e}")
            return None

    def load_trajectory_json(self, filepath: str) -> Optional[np.ndarray]:
        """Загружает траекторию из JSON."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            traj = np.array(data.get("trajectory"))
            return traj
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке JSON траектории: {e}")
            return None
    
    def load_parameters_json(self, filepath: str) -> Optional[Dict]:
        """
        Загружает параметры из JSON файла.
        
        Args:
            filepath: Путь к файлу
        
        Returns:
            Словарь параметров или None при ошибке
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке JSON: {e}")
            return None
