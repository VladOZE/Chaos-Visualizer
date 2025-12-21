# src/main.py
"""
Точка входа в приложение Chaos Visualizer.
Запускает PyQt5 интерфейс для моделирования и визуализации хаотических систем.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from ui.main_window import ChaosVisualizerApp
from logger import get_logger

logger = get_logger("main")


def main():
    """Главная функция для запуска приложения."""
    # Создание приложения Qt
    app = QApplication(sys.argv)
    
    # Установка стиля приложения
    app.setStyle('Fusion')
    
    # Создание и отображение главного окна
    window = ChaosVisualizerApp()
    window.show()
    logger.info("Приложение запущено")
    
    # Запуск event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    # Добавляем текущую директорию в путь для импортов
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
