"""
Загрузчик конфигурации из JSON файла.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("config_loader")


class ConfigLoader:
    """Загружает и предоставляет доступ к конфигурации приложения."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Загружает конфигурацию из JSON файла.
        
        Args:
            config_path: Путь к файлу конфигурации. 
                        По умолчанию ищет config/default_params.json
        
        Returns:
            Словарь с конфигурацией
        """
        if config_path is None:
            # Ищем конфиг относительно текущего файла
            config_path = Path(__file__).parent.parent / "config" / "default_params.json"
        else:
            config_path = Path(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            logger.info(f"✓ Конфигурация загружена: {config_path}")
            return self._config
        except FileNotFoundError:
            logger.error(f"✗ Файл конфигурации не найден: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"✗ Ошибка при парсинге JSON: {e}")
            raise
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Получает значение по пути в конфигурации.
        
        Args:
            path: Путь вида "systems.lorenz.parameters.sigma.value"
            default: Значение по умолчанию если ключ не найден
        
        Returns:
            Значение или default
        """
        if self._config is None:
            self.load_config()
        
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_system_config(self, system_name: str) -> Dict[str, Any]:
        """Получает конфигурацию системы."""
        return self.get(f"systems.{system_name}", {})
    
    def get_system_parameters(self, system_name: str) -> Dict[str, Dict[str, float]]:
        """Получает параметры системы."""
        return self.get(f"systems.{system_name}.parameters", {})
    
    def get_system_initial_conditions(self, system_name: str) -> Dict[str, float]:
        """Получает начальные условия для системы."""
        return self.get(f"systems.{system_name}.initial_conditions", {})
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Получает конфигурацию интеграции."""
        return self.get("integration", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Получает конфигурацию визуализации."""
        return self.get("visualization", {})
    
    def get_export_config(self) -> Dict[str, Any]:
        """Получает конфигурацию экспорта."""
        return self.get("export", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Получает конфигурацию UI."""
        return self.get("ui", {})
    
    def get_custom_equation_examples(self) -> Dict[str, Dict[str, str]]:
        """Получает примеры пользовательских уравнений."""
        return self.get("systems.custom.example_equations", {})
    
    def get_default_steps(self) -> int:
        """Получает количество шагов интеграции по умолчанию."""
        return self.get("integration.default_steps", 3000)
    
    def get_chunk_size(self) -> int:
        """Получает размер чанка интеграции."""
        return self.get("integration.chunk_size", 500)
    
    def get_rk45_tolerance(self) -> tuple:
        """Получает допуски для метода RK45 (rtol, atol)."""
        rtol = self.get("integration.rtol", 1e-6)
        atol = self.get("integration.atol", 1e-9)
        return rtol, atol


def get_config_loader() -> ConfigLoader:
    """Получить экземпляр конфиг-загрузчика."""
    return ConfigLoader()


__all__ = ["ConfigLoader", "get_config_loader"]
