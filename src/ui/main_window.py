# src/ui/main_window.py
"""
Главное окно приложения PyQt5.
Объединяет интерфейс пользователя с численным моделированием и визуализацией.
"""

import sys
import time
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget,
    QMessageBox, QFileDialog, QGroupBox, QFormLayout, QProgressBar,
    QSlider, QScrollArea, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Импорты из нашего проекта
sys.path.insert(0, '..')
from models.lorenz import LorenzSystem
from models.rossler import RosslerSystem
from models.chua import ChuaSystem
from models.custom_system import CustomEquationSystem
from integration.integrator import ODEIntegrator
from visualization.matplotlib_plots import MatplotlibPlotter
from visualization.plotly_plots import PlotlyVisualizer
from storage.exporter import ResultsExporter
from config_loader import get_config_loader
from logger import get_logger
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

ui_logger = get_logger("ui")


class IntegrationWorker(QThread):
    """Рабочий поток для численного интегрирования."""
    
    progress_signal = pyqtSignal(int)
    completed_signal = pyqtSignal(np.ndarray, np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self, dynamics_system, integrator, initial_state, num_points):
        super().__init__()
        self.dynamics_system = dynamics_system
        self.integrator = integrator
        self.initial_state = initial_state
        self.num_points = num_points
        self._cancel_requested = False
    
    def request_cancel(self):
        """Запрашивает остановку вычислений."""
        self._cancel_requested = True
        self.requestInterruption()
    
    def run(self):
        try:
            self.progress_signal.emit(10)
            
            if self._cancel_requested or self.isInterruptionRequested():
                self.error_signal.emit("Моделирование остановлено пользователем")
                return
            
            # Интегрирование с возможностью отмены
            if hasattr(self.dynamics_system, "simulate"):
                # Для дискретных систем
                t, trajectory = self.dynamics_system.simulate(self.initial_state, self.num_points)
            else:
                # Для непрерывных систем - интегрируем частями
                dt = 0.01
                chunk_size = 500
                total_points = int(self.num_points)
                t_parts = []
                y_parts = []
                current_state = np.array(self.initial_state, dtype=float)
                start_idx = 0
                
                while start_idx < total_points:
                    if self._cancel_requested or self.isInterruptionRequested():
                        self.error_signal.emit("Моделирование остановлено пользователем")
                        return
                    
                    end_idx = min(start_idx + chunk_size, total_points)
                    local_count = end_idx - start_idx
                    t0 = start_idx * dt
                    t1 = end_idx * dt
                    t_eval = np.linspace(t0, t1, local_count)
                    
                    t_chunk, y_chunk = self.integrator.integrate(
                        self.dynamics_system.compute_derivatives,
                        current_state,
                        (t0, t1),
                        t_eval=t_eval
                    )
                    
                    if start_idx > 0:
                        t_chunk = t_chunk[1:]
                        y_chunk = y_chunk[1:]
                    
                    if len(t_chunk) == 0 or len(y_chunk) == 0:
                        raise RuntimeError("Интегратор вернул пустой результат")
                    
                    t_parts.append(t_chunk)
                    y_parts.append(y_chunk)
                    current_state = y_chunk[-1]
                    start_idx = end_idx
                    
                    # Отправляем прогресс
                    progress = 10 + int(80 * start_idx / total_points)
                    self.progress_signal.emit(min(90, progress))
                
                t = np.concatenate(t_parts)
                trajectory = np.concatenate(y_parts)
            
            self.progress_signal.emit(100)
            self.completed_signal.emit(t, trajectory)
        
        except Exception as e:
            ui_logger.error(f"Ошибка в рабочем потоке интегрирования: {e}")
            self.error_signal.emit(str(e))


class ChaosVisualizerApp(QMainWindow):
    """Главное приложение для визуализации хаотических систем."""
    
    def __init__(self):
        super().__init__()
        
        # Загрузка конфигурации
        self.config = get_config_loader()
        ui_config = self.config.get_ui_config()
        
        # Инициализация окна из конфига
        window_config = ui_config.get('window', {})
        geom = window_config.get('geometry', {'width': 1600, 'height': 900, 'x': 100, 'y': 100})
        self.setWindowTitle(window_config.get('title', 'Хаотические системы'))
        self.setGeometry(geom['x'], geom['y'], geom['width'], geom['height'])
        
        # Инициализация компонентов
        self.systems = {
            'Лоренц': LorenzSystem(),
            'Рёсслер': RosslerSystem(),
            'Чуа': ChuaSystem(),
            'Пользовательская': CustomEquationSystem(),
        }
        self.current_system = self.systems['Лоренц']
        self.integrator = ODEIntegrator(method='RK45', verbose=True)
        self.plotter = MatplotlibPlotter()
        self.visualizer = PlotlyVisualizer()
        self.exporter = ResultsExporter('data')
        self.logger = ui_logger
        self.live_timer = QTimer()
        self.live_timer.setInterval(30)
        self.live_timer.timeout.connect(self.update_live_frame)
        self.live_index = 0
        self.live_line = None          # линия для 3D анимации
        self.live_line_2d = None       # линия для 2D анимации
        self.anim2d_x_idx = 0
        self.anim2d_y_idx = 1
        self.start_wall_time = None
        self.live_step = 50
        self.speed_factor = 1.0
        self.scale_factor = 1.0
        self.scale_factor_2d = 1.0
        
        self.trajectory = None
        self.time_array = None
        self.worker = None
        
        # Построение интерфейса
        self.setup_ui()
    
    def setup_ui(self):
        """Создаёт интерфейс приложения."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # ============= ЛЕВАЯ ПАНЕЛЬ (управление) =============
        left_panel = QVBoxLayout()
        
        # Выбор модели
        model_group = QGroupBox("Модель системы")
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.systems.keys())
        self.model_combo.currentTextChanged.connect(self.on_system_changed)
        model_layout.addRow("Система:", self.model_combo)
        
        model_group.setLayout(model_layout)
        left_panel.addWidget(model_group)
        
        # Параметры системы
        self.params_group = QGroupBox("Параметры")
        self.params_layout = QFormLayout()
        self.param_inputs = {}
        
        self.update_parameter_inputs()
        self.params_group.setLayout(self.params_layout)
        left_panel.addWidget(self.params_group)
        
        # Пользовательские уравнения
        self.custom_eq_group = QGroupBox("Пользовательские уравнения")
        custom_eq_layout = QFormLayout()
        
        # Установим пример уравнений из конфига
        eq_examples = self.config.get_custom_equation_examples()
        lorenz_example = eq_examples.get('lorenz_copy', {})
        default_dx = lorenz_example.get('dx', '10*(y-x)')
        default_dy = lorenz_example.get('dy', 'x*(28-z)-y')
        default_dz = lorenz_example.get('dz', 'x*y-2.6666666667*z')
        
        self.dx_input = QLineEdit()
        self.dx_input.setPlaceholderText("Например: 10*(y-x)")
        self.dx_input.setText(default_dx)
        custom_eq_layout.addRow("dx/dt:", self.dx_input)
        
        self.dy_input = QLineEdit()
        self.dy_input.setPlaceholderText("Например: x*(28-z)-y")
        self.dy_input.setText(default_dy)
        custom_eq_layout.addRow("dy/dt:", self.dy_input)
        
        self.dz_input = QLineEdit()
        self.dz_input.setPlaceholderText("Например: x*y-2.6666666667*z")
        self.dz_input.setText(default_dz)
        custom_eq_layout.addRow("dz/dt:", self.dz_input)
        
        self.custom_eq_group.setLayout(custom_eq_layout)
        left_panel.addWidget(self.custom_eq_group)
        self.custom_eq_group.setVisible(False)
        
        # Начальные условия
        init_group = QGroupBox("Начальные условия")
        init_layout = QFormLayout()
        
        # Получаем начальные условия для текущей системы (Lorenz)
        init_conds = self.config.get_system_initial_conditions('lorenz')
        
        self.x0_input = QDoubleSpinBox()
        self.x0_input.setRange(-100, 100)
        self.x0_input.setValue(init_conds.get('x0', 1.0))
        init_layout.addRow("x₀:", self.x0_input)
        
        self.y0_input = QDoubleSpinBox()
        self.y0_input.setRange(-100, 100)
        self.y0_input.setValue(init_conds.get('y0', 1.0))
        init_layout.addRow("y₀:", self.y0_input)
        
        self.z0_input = QDoubleSpinBox()
        self.z0_input.setRange(-100, 100)
        self.z0_input.setValue(init_conds.get('z0', 1.0))
        init_layout.addRow("z₀:", self.z0_input)
        
        init_group.setLayout(init_layout)
        left_panel.addWidget(init_group)
        
        # Параметры интегрирования
        integ_group = QGroupBox("Интегрирование")
        integ_layout = QFormLayout()
        
        self.num_points_spinbox = QSpinBox()
        self.num_points_spinbox.setRange(100, 100000)
        default_steps = self.config.get_default_steps()
        self.num_points_spinbox.setValue(default_steps)
        integ_layout.addRow("Число итераций:", self.num_points_spinbox)
        
        integ_group.setLayout(integ_layout)
        left_panel.addWidget(integ_group)
        
        # Кнопки управления
        run_stop_layout = QHBoxLayout()
        self.run_button = QPushButton("▶ Запустить моделирование")
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_button.setToolTip("Запустить численное моделирование с текущими параметрами")
        self.run_button.clicked.connect(self.run_simulation)
        run_stop_layout.addWidget(self.run_button)
        
        self.stop_button = QPushButton("⏹ Остановить")
        self.stop_button.setStyleSheet("background-color: #d92d20; color: white; font-weight: bold;")
        self.stop_button.setToolTip("Остановить текущее моделирование")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        run_stop_layout.addWidget(self.stop_button)
        left_panel.addLayout(run_stop_layout)

        self.reset_button = QPushButton("↻ Сбросить параметры")
        self.reset_button.setStyleSheet("background-color: #d92d20; color: white; font-weight: bold;")
        self.reset_button.setToolTip("Сбросить параметры к значениям по умолчанию")
        self.reset_button.clicked.connect(self.reset_parameters)
        left_panel.addWidget(self.reset_button)

        self.clear_plots_button = QPushButton("🧹 Очистить графики")
        self.clear_plots_button.setToolTip("Очистить текущие 2D и 3D графики, не изменяя параметры")
        self.clear_plots_button.clicked.connect(self.clear_plots)
        left_panel.addWidget(self.clear_plots_button)
        
        # Блок анимации
        self.anim_3d_group = QGroupBox("Анимация")
        anim_group_layout = QVBoxLayout()
        
        anim_buttons = QHBoxLayout()
        self.animate_button = QPushButton("▶ Анимация")
        self.animate_button.setToolTip("Запустить отрисовку аттрактора по шагам")
        self.animate_button.clicked.connect(self.start_live_draw)
        self.pause_animation_button = QPushButton("⏸ Пауза")
        self.pause_animation_button.setToolTip("Поставить анимацию на паузу")
        self.pause_animation_button.clicked.connect(self.pause_live_draw)
        self.resume_animation_button = QPushButton("⏵ Продолжить")
        self.resume_animation_button.setToolTip("Продолжить анимацию с текущего места")
        self.resume_animation_button.clicked.connect(self.resume_live_draw)
        anim_buttons.addWidget(self.animate_button)
        anim_buttons.addWidget(self.pause_animation_button)
        anim_buttons.addWidget(self.resume_animation_button)
        anim_group_layout.addLayout(anim_buttons)
        
        # Ползунок скорости анимации
        self.speed_group = QGroupBox("Скорость анимации")
        speed_layout = QVBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)    # 0.1x
        self.speed_slider.setMaximum(50)   # 5.0x
        self.speed_slider.setValue(10)     # 1.0x
        self.speed_slider.setToolTip("Изменить скорость проигрывания анимации 3D аттрактора")
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        self.speed_group.setLayout(speed_layout)
        anim_group_layout.addWidget(self.speed_group)

        # Ползунок масштаба 3D
        self.scale_group = QGroupBox("Масштаб 3D")
        scale_layout = QVBoxLayout()
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(10)   # 0.5x
        self.scale_slider.setMaximum(300)  # 2.0x
        self.scale_slider.setValue(100)    # 1.0x
        self.scale_slider.setToolTip("Масштабировать 3D аттрактор относительно центра")
        self.scale_slider.valueChanged.connect(self.on_scale_change)
        self.scale_label = QLabel("100%")
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(self.scale_label)
        self.scale_group.setLayout(scale_layout)
        anim_group_layout.addWidget(self.scale_group)

        # Ползунок масштаба 2D
        self.scale2d_group = QGroupBox("Масштаб 2D")
        scale2d_layout = QVBoxLayout()
        self.scale2d_slider = QSlider(Qt.Horizontal)
        self.scale2d_slider.setMinimum(10)   # 0.5x
        self.scale2d_slider.setMaximum(300)  # 3.0x
        self.scale2d_slider.setValue(100)    # 1.0x
        self.scale2d_slider.setToolTip("Масштабировать 2D фазовый портрет относительно центра")
        self.scale2d_slider.valueChanged.connect(self.on_scale2d_change)
        self.scale2d_label = QLabel("100%")
        scale2d_layout.addWidget(self.scale2d_slider)
        scale2d_layout.addWidget(self.scale2d_label)
        self.scale2d_group.setLayout(scale2d_layout)
        anim_group_layout.addWidget(self.scale2d_group)

        # Управление 2D-анимацией (выбор проекции)
        self.anim2d_group = QGroupBox("Проекция")
        anim2d_layout = QVBoxLayout()
        self.anim2d_combo = QComboBox()
        self.anim2d_combo.addItem("x-y", (0, 1))
        self.anim2d_combo.addItem("x-z", (0, 2))
        self.anim2d_combo.addItem("y-z", (1, 2))
        self.anim2d_combo.setToolTip("Выберите проекцию для 2D фазового портрета и анимации")
        self.anim2d_combo.currentIndexChanged.connect(self.on_anim2d_projection_change)
        anim2d_layout.addWidget(self.anim2d_combo)
        self.anim2d_group.setLayout(anim2d_layout)
        anim_group_layout.addWidget(self.anim2d_group)

        self.anim_3d_group.setLayout(anim_group_layout)
        left_panel.addWidget(self.anim_3d_group)
        
               
        # Экспорт
        export_group = QGroupBox("Экспорт результатов")
        export_layout = QVBoxLayout()
        self.export_combo = QComboBox()
        self.export_combo.addItem("PNG", "png")
        self.export_combo.addItem("SVG", "svg")
        self.export_combo.addItem("CSV", "csv")
        self.export_combo.addItem("HTML (3D)", "html")
        self.export_combo.addItem("JSON", "json")
        self.export_combo.addItem("GIF (анимация)", "gif")
        self.export_combo.addItem("MP4 (анимация)", "mp4")
        self.export_combo.setToolTip("Выберите формат, в который нужно экспортировать результаты")
        export_layout.addWidget(self.export_combo)
        self.export_run_button = QPushButton("Экспортировать")
        self.export_run_button.setToolTip("Выполнить экспорт в выбранный формат")
        self.export_run_button.clicked.connect(self.export_selected)
        export_layout.addWidget(self.export_run_button)
        export_group.setLayout(export_layout)
        left_panel.addWidget(export_group)

        # Импорт
        import_group = QGroupBox("Импорт результатов")
        import_layout = QVBoxLayout()
        self.import_csv_button = QPushButton("⬇ Импорт CSV")
        self.import_csv_button.setToolTip("Загрузить сохранённую траекторию из CSV")
        self.import_csv_button.clicked.connect(self.import_csv)
        import_layout.addWidget(self.import_csv_button)

        self.import_json_button = QPushButton("⬇ Импорт JSON")
        self.import_json_button.setToolTip("Загрузить сохранённую траекторию из JSON")
        self.import_json_button.clicked.connect(self.import_json)
        import_layout.addWidget(self.import_json_button)
        import_group.setLayout(import_layout)
        left_panel.addWidget(import_group)
        
        left_panel.addStretch()
        
        # ============= ПРАВАЯ ПАНЕЛЬ (визуализация) =============
        self.tabs = QTabWidget()
        
        # Вкладка 2D графики
        self.fig_2d = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.toolbar_2d = NavigationToolbar(self.canvas_2d, self)
        tab2d_widget = QWidget()
        tab2d_layout = QVBoxLayout()
        tab2d_layout.setContentsMargins(0, 0, 0, 0)
        tab2d_layout.addWidget(self.toolbar_2d)
        tab2d_layout.addWidget(self.canvas_2d)
        tab2d_widget.setLayout(tab2d_layout)
        self.tabs.addTab(tab2d_widget, "2D фазовые портреты")
        
        # Вкладка 3D графики
        self.fig_3d = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.tabs.addTab(self.canvas_3d, "3D аттрактор")
        self.tabs.currentChanged.connect(self.update_controls_visibility)
        
        # Показываем сообщение на пустых графиках при инициализации
        self.show_empty_plot_message(self.fig_2d, self.canvas_2d)
        self.show_empty_plot_message(self.fig_3d, self.canvas_3d)
        
        # Прокручиваемая левая панель
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_widget)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                width: 10px;
                background: transparent;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e1;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Нижняя фиксированная панель статусов
        status_container = QWidget()
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_container.setLayout(status_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        self.info_label = QLabel("Готово к моделированию")
        self.info_label.setWordWrap(True)
        status_layout.addWidget(self.info_label)
        self.time_label = QLabel("Время моделирования: -")
        status_layout.addWidget(self.time_label)

        # Левый контейнер с прокруткой + статус
        left_wrapper = QVBoxLayout()
        left_wrapper.addWidget(scroll)
        left_wrapper.addWidget(status_container)
        left_wrapper.setStretch(0, 1)
        left_wrapper.setStretch(1, 0)
        left_container = QWidget()
        left_container.setLayout(left_wrapper)
        
        # Основной макет
        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(self.tabs, 2)
        self.setFont(QFont("Segoe UI", 10))
        self.apply_app_styles()
        
        central_widget.setLayout(main_layout)
        self.update_controls_visibility()
        

    def update_parameter_inputs(self):
        """Обновляет входные поля параметров в соответствии с выбранной системой."""
        # Очищаем старые параметры
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
        self.param_inputs.clear()
        
        # Добавляем новые параметры
        ranges = self.current_system.get_parameter_ranges()
        params = self.current_system.get_parameters()
        
        for param_name, (min_val, max_val) in ranges.items():
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(params[param_name])
            spinbox.setSingleStep((max_val - min_val) / 100)
            
            self.param_inputs[param_name] = spinbox
            self.params_layout.addRow(f"{param_name}:", spinbox)
    
    def on_system_changed(self, system_name):
        """Обработчик смены модели системы."""
        self.current_system = self.systems[system_name]
        self.update_parameter_inputs()
        
        # Обновляем начальные условия из конфига в зависимости от системы
        system_key = {
            'Лоренц': 'lorenz',
            'Рёсслер': 'rossler',
            'Чуа': 'chua',
            'Пользовательская': 'custom'
        }.get(system_name, 'lorenz')
        
        init_conds = self.config.get_system_initial_conditions(system_key)
        if init_conds:
            self.x0_input.setValue(init_conds.get('x0', 1.0))
            self.y0_input.setValue(init_conds.get('y0', 1.0))
            self.z0_input.setValue(init_conds.get('z0', 1.0))
        
        self.custom_eq_group.setVisible(system_name == "Пользовательская")
        self.params_group.setVisible(system_name != "Пользовательская")
        self.info_label.setText(f"Выбрана система: {system_name}")
        self.logger.info(f"Пользователь выбрал систему: {system_name}")
    
    
    def reset_parameters(self):
        """Сбрасывает параметры на значения по умолчанию."""
        self.current_system.reset_to_defaults()
        self.update_parameter_inputs()
        self.info_label.setText("Параметры сброшены")

    def show_empty_plot_message(self, fig, canvas, message="Запустите моделирование для отображения графика"):
        """Отображает сообщение на пустом графике."""
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, message, 
                horizontalalignment='center', 
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                color='#666666',
                style='italic')
        canvas.draw()
    
    def clear_plots(self):
        """Очищает графики, не изменяя параметры системы."""
        self.info_label.setText("Графики очищены")
        self.time_label.setText("Время моделирования: -")
        self.trajectory = None
        self.time_array = None
        self.show_empty_plot_message(self.fig_2d, self.canvas_2d)
        self.show_empty_plot_message(self.fig_3d, self.canvas_3d)
    
    def run_simulation(self):
        """Запускает численное моделирование."""
        try:
            # Получаем параметры
            params = {name: spinbox.value() for name, spinbox in self.param_inputs.items()}
            
            if not self.current_system.set_parameters(**params):
                error_msg = f"Ошибка при установке параметров: проверьте значения {list(params.keys())}"
                self.info_label.setText(f"✗ {error_msg}")
                QMessageBox.critical(self, "Ошибка параметров", error_msg)
                return
            
            # Для пользовательской системы устанавливаем уравнения
            if isinstance(self.current_system, CustomEquationSystem):
                if not self.current_system.set_equations(
                    self.dx_input.text(),
                    self.dy_input.text(),
                    self.dz_input.text(),
                ):
                    error_msg = "Некорректные пользовательские уравнения. Проверьте синтаксис выражений."
                    self.info_label.setText(f"✗ {error_msg}")
                    QMessageBox.critical(self, "Ошибка уравнений", error_msg)
                    return
            
            # Начальные условия
            initial_state = np.array([
                self.x0_input.value(),
                self.y0_input.value(),
                self.z0_input.value()
            ])
            
            if not self.current_system.validate_state(initial_state):
                error_msg = f"Невалидное начальное состояние: {initial_state}. Используйте конечные числовые значения."
                self.info_label.setText(f"✗ {error_msg}")
                QMessageBox.critical(self, "Ошибка начальных условий", error_msg)
                return
            
            num_points = self.num_points_spinbox.value()
            if num_points < 10:
                error_msg = "Число итераций должно быть не менее 10"
                self.info_label.setText(f"✗ {error_msg}")
                QMessageBox.warning(self, "Ошибка конфигурации", error_msg)
                return
            
            self.info_label.setText(f"⏳ Моделирование в процессе ({self.current_system.name})...")
            self.progress_bar.setValue(0)
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.start_wall_time = time.perf_counter()
            self.stop_live_draw()
            self.logger.info(
                f"Запуск моделирования системы {self.current_system.name} "
                f"с параметрами {params} и начальным состоянием {initial_state}"
            )
            
            # Запускаем интегрирование в отдельном потоке
            self.worker = IntegrationWorker(
                self.current_system,
                self.integrator,
                initial_state,
                num_points
            )
            self.worker.progress_signal.connect(self.update_progress)
            self.worker.completed_signal.connect(self.on_simulation_complete)
            self.worker.error_signal.connect(self.on_simulation_error)
            self.worker.start()
        
        except Exception as e:
            self.logger.error(f"Ошибка при запуске моделирования: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при запуске моделирования:\n{str(e)}")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def stop_simulation(self):
        """Прерывает моделирование по запросу пользователя."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel()
            self.info_label.setText("Остановка моделирования...")
            # Fallback: если поток не остановился сам, завершаем принудительно
            QTimer.singleShot(2000, self._force_stop_worker_if_needed)
    
    def _force_stop_worker_if_needed(self):
        """Принудительно завершает рабочий поток, если он всё ещё запущен."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(500)
            self.info_label.setText("Моделирование остановлено пользователем")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def update_progress(self, value):
        """Обновляет прогресс-бар."""
        self.progress_bar.setValue(value)
    
    def on_simulation_complete(self, t, trajectory):
        """Обработчик завершения моделирования."""
        self.time_array = t
        self.trajectory = trajectory
        
        # Анализируем качество траектории
        finite_mask = np.all(np.isfinite(trajectory), axis=1)
        valid_points = np.sum(finite_mask)
        total_points = len(trajectory)
        quality_pct = 100.0 * valid_points / total_points if total_points > 0 else 0
        
        # Форматируем информацию о параметрах
        params_str = ", ".join([f"{k}={v:.4g}" for k, v in self.current_system.get_parameters().items()])
        
        self.info_label.setText(
            f"✓ Моделирование завершено! "
            f"Точек: {valid_points}/{total_points} ({quality_pct:.1f}%) | "
            f"Параметры: {params_str}"
        )
        
        self.logger.info(
            f"Моделирование {self.current_system.name} завершено успешно. "
            f"Вычислено {len(trajectory)} точек ({valid_points} конечных). Качество: {quality_pct:.1f}%"
        )
        
        # Время моделирования (реальное)
        if self.start_wall_time is not None:
            elapsed = time.perf_counter() - self.start_wall_time
            self.time_label.setText(f"Время моделирования: {elapsed:.2f} c")
        
        try:
            self.plot_results()
        except Exception as exc:
            self.logger.error(f"Ошибка при отображении результатов: {exc}")
            self.info_label.setText(f"✗ Ошибка визуализации: {str(exc)[:60]}...")
            QMessageBox.critical(self, "Ошибка визуализации", str(exc))
        finally:
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_bar.setValue(100)
    
    def on_simulation_error(self, error_msg):
        """Обработчик ошибки моделирования."""
        self.logger.error(f"Ошибка моделирования: {error_msg}")
        self.info_label.setText(f"✗ Ошибка: {error_msg[:70]}...")
        QMessageBox.critical(self, "Ошибка при моделировании", error_msg)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def plot_results(self):
        """Отображает результаты моделирования."""
        if self.trajectory is None:
            QMessageBox.warning(self, "Нет данных", "Сначала запустите моделирование")
            return
        
        self.logger.info("Отрисовка результатов...")
        self.stop_live_draw()
        self.render_2d_plot()
        self.render_3d_plot()
        self.logger.info("Отрисовка завершена")

    def render_2d_plot(self):
        """Рисует статический 2D фазовый портрет в выбранной проекции."""
        if self.trajectory is None:
            return
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        x_idx, y_idx = self.anim2d_x_idx, self.anim2d_y_idx
        x = self.trajectory[:, x_idx]
        y = self.trajectory[:, y_idx]
        ax.plot(x, y, 'b-', linewidth=0.5, alpha=0.8)
        labels = ['x', 'y', 'z']
        lbl_x = labels[x_idx] if x_idx < len(labels) else f'axis {x_idx}'
        lbl_y = labels[y_idx] if y_idx < len(labels) else f'axis {y_idx}'
        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)
        ax.set_title(f"{self.current_system.name} - 2D фазовый портрет ({lbl_x}-{lbl_y})")
        ax.grid(True, alpha=0.3)
        self.canvas_2d.draw()

    def render_3d_plot(self):
        """Рисует статический 3D аттрактор."""
        if self.trajectory is None or len(self.trajectory) == 0:
            self.show_empty_plot_message(self.fig_3d, self.canvas_3d, "Нет данных для отображения")
            return
        
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')
        self.ax3d = ax
        
        x = self.trajectory[:, 0]
        y = self.trajectory[:, 1]
        z = self.trajectory[:, 2]
        
        # Проверка на NaN/Inf значения
        finite_mask = np.all(np.isfinite(self.trajectory), axis=1)
        if not np.any(finite_mask):
            self.logger.error("Траектория содержит только NaN/Inf значения")
            self.show_empty_plot_message(self.fig_3d, self.canvas_3d, 
                                       "Ошибка: траектория содержит только NaN значения")
            self.info_label.setText("✗ Ошибка: траектория содержит NaN/Inf значения")
            return
        
        # Используем только конечные значения для отрисовки
        x_clean = x[finite_mask]
        y_clean = y[finite_mask]
        z_clean = z[finite_mask]
        
        if len(x_clean) > 1:
            ax.plot(x_clean, y_clean, z_clean, 'b-', linewidth=0.6, alpha=0.9)
        
        # Отмечаем начальную и конечную точки
        if np.isfinite(x[0]) and np.isfinite(y[0]) and np.isfinite(z[0]):
            ax.scatter(x[0], y[0], z[0], c='green', s=30, label='Начало')
        if np.isfinite(x[-1]) and np.isfinite(y[-1]) and np.isfinite(z[-1]):
            ax.scatter(x[-1], y[-1], z[-1], c='red', s=30, label='Конец')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{self.current_system.name} - 3D аттрактор")
        ax.legend()
        
        try:
            self.apply_axis_scale(ax, x_clean, y_clean, z_clean)
        except Exception as e:
            self.logger.warning(f"Ошибка при применении масштаба: {e}. Используются автоматические пределы.")
        
        self.canvas_3d.draw()
        self.update_controls_visibility()
    
    def export_via_menu(self, fmt: str):
        """Экспорт через меню (png, svg, csv, html, json, gif, mp4)."""
        if self.trajectory is None:
            error_msg = "Сначала запустите моделирование"
            self.info_label.setText(f"✗ {error_msg}")
            QMessageBox.warning(self, "Нет данных", error_msg)
            return
        
        suggested = {
            "png": "attractor.png",
            "svg": "attractor.svg",
            "csv": "trajectory.csv",
            "html": "attractor.html",
            "json": "trajectory.json",
            "gif": "animation.gif",
            "mp4": "animation.mp4",
        }.get(fmt, "export.dat")
        
        filter_map = {
            "png": "PNG (*.png)",
            "svg": "SVG (*.svg)",
            "csv": "CSV (*.csv)",
            "html": "HTML (*.html)",
            "json": "JSON (*.json)",
            "gif": "GIF (*.gif)",
            "mp4": "MP4 (*.mp4)",
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Экспорт {fmt.upper()}",
            suggested,
            filter_map.get(fmt, "All Files (*.*)")
        )
        if not file_path:
            return
        
        try:
            self.info_label.setText(f"⏳ Экспорт в {fmt.upper()}...")
            
            if fmt in ["png", "svg"]:
                fig = self.plotter.plot_3d_projection(self.trajectory)
                self.exporter.save_figure(fig, file_path.replace(f".{fmt}", ""), formats=[fmt])
                plt.close(fig)
                self.logger.info(f"Экспортировано в {fmt.upper()}: {len(self.trajectory)} точек")
            elif fmt == "csv":
                self.exporter.save_trajectory_csv(self.trajectory, file_path)
                self.logger.info(f"CSV экспортирован: {file_path}")
            elif fmt == "json":
                self.exporter.save_trajectory_json(self.trajectory, file_path)
                self.logger.info(f"JSON экспортирован: {file_path}")
            elif fmt == "html":
                fig = self.visualizer.plot_3d_attractor(
                    self.trajectory,
                    title=f"{self.current_system.name} 3D аттрактор"
                )
                self.exporter.save_plotly_html(fig, file_path)
                self.logger.info(f"HTML экспортирован: {file_path}")
            elif fmt in ["gif", "mp4"]:
                self.exporter.save_animation(self.trajectory, filename=file_path, interval=50)
                self.logger.info(f"Анимация {fmt.upper()} экспортирована: {file_path} ({len(self.trajectory)} кадров)")
            
            self.info_label.setText(f"✓ Экспортировано в {fmt.upper()}")
            QMessageBox.information(self, "Успех", f"Файл сохранён:\n{file_path}")
        except Exception as e:
            error_msg = str(e)[:70]
            self.logger.error(f"Ошибка при экспорте {fmt}: {e}")
            self.info_label.setText(f"✗ Ошибка экспорта: {error_msg}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте: {e}")

    def export_selected(self):
        """Обработчик кнопки 'Экспортировать'."""
        fmt = self.export_combo.currentData()
        if fmt:
            self.export_via_menu(fmt)

    def import_csv(self):
        """Импортирует траекторию из CSV и отображает."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Импорт CSV", "", "CSV (*.csv)")
        if not file_path:
            return
        try:
            traj = self.exporter.load_trajectory_csv(file_path)
            if traj is None:
                error_msg = "Не удалось импортировать CSV - проверьте формат файла"
                self.logger.error(f"CSV import failed: {file_path}")
                self.info_label.setText(f"✗ {error_msg}")
                QMessageBox.critical(self, "Ошибка импорта", error_msg)
                return
            self.logger.info(f"CSV импортирован успешно: {file_path} ({len(traj)} точек)")
            self.apply_imported_trajectory(traj)
        except Exception as e:
            error_msg = f"Ошибка при импорте CSV: {str(e)[:60]}"
            self.logger.error(f"CSV import exception: {e}")
            self.info_label.setText(f"✗ {error_msg}")
            QMessageBox.critical(self, "Ошибка", error_msg)

    def import_json(self):
        """Импортирует траекторию из JSON и отображает."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Импорт JSON", "", "JSON (*.json)")
        if not file_path:
            return
        try:
            traj = self.exporter.load_trajectory_json(file_path)
            if traj is None:
                error_msg = "Не удалось импортировать JSON - проверьте формат файла"
                self.logger.error(f"JSON import failed: {file_path}")
                self.info_label.setText(f"✗ {error_msg}")
                QMessageBox.critical(self, "Ошибка импорта", error_msg)
                return
            self.logger.info(f"JSON импортирован успешно: {file_path} ({len(traj)} точек)")
            self.apply_imported_trajectory(traj)
        except Exception as e:
            error_msg = f"Ошибка при импорте JSON: {str(e)[:60]}"
            self.logger.error(f"JSON import exception: {e}")
            self.info_label.setText(f"✗ {error_msg}")
            QMessageBox.critical(self, "Ошибка", error_msg)

    def apply_imported_trajectory(self, traj: np.ndarray):
        """Применяет импортированную траекторию и обновляет графики."""
        if traj.ndim != 2 or traj.shape[1] < 2:
            error_msg = f"Некорректные данные: ожидается матрица N x 3, получено {traj.shape}"
            self.logger.error(error_msg)
            self.info_label.setText(f"✗ {error_msg}")
            QMessageBox.warning(self, "Некорректные данные", error_msg)
            return
        self.trajectory = traj
        self.time_array = np.arange(len(traj))
        
        # Анализируем качество импортированной траектории
        finite_mask = np.all(np.isfinite(traj), axis=1)
        valid_points = np.sum(finite_mask)
        quality_pct = 100.0 * valid_points / len(traj)
        
        self.info_label.setText(f"✓ Импортировано {len(traj)} точек ({valid_points} конечных, {quality_pct:.1f}%)")
        try:
            self.plot_results()
        except Exception as exc:
            self.logger.error(f"Ошибка при отображении импортированных данных: {exc}")
            QMessageBox.critical(self, "Ошибка визуализации", str(exc))

    def start_live_draw(self):
        """Запускает пошаговую отрисовку 3D траектории."""
        if self.trajectory is None or len(self.trajectory) == 0:
            QMessageBox.warning(self, "Нет данных", "Сначала запустите моделирование")
            return
        
        self.stop_live_draw()
        self.logger.info("Запуск анимации 3D аттрактора")
        self.live_index = 1

        # Настройка 3D для анимации
        self.fig_3d.clear()
        self.ax3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title(f"{self.current_system.name} - отрисовка 3D")
        self.live_line, = self.ax3d.plot([], [], [], 'b-', linewidth=0.8, alpha=0.9)
        
        # Проверка валидности первой и последней точек
        first_pt_valid = all(np.isfinite([self.trajectory[0, 0], self.trajectory[0, 1], self.trajectory[0, 2]]))
        last_pt_valid = all(np.isfinite([self.trajectory[-1, 0], self.trajectory[-1, 1], self.trajectory[-1, 2]]))
        
        if first_pt_valid:
            self.ax3d.scatter(self.trajectory[0, 0], self.trajectory[0, 1], self.trajectory[0, 2],
                            c='green', s=30, label='Начало')
        if last_pt_valid:
            self.ax3d.scatter(self.trajectory[-1, 0], self.trajectory[-1, 1], self.trajectory[-1, 2],
                            c='red', s=30, label='Конец')
        
        self.ax3d.legend()
        
        try:
            self.apply_axis_scale(self.ax3d,
                                self.trajectory[:, 0],
                                self.trajectory[:, 1],
                                self.trajectory[:, 2])
        except Exception as e:
            self.logger.warning(f"Не удалось применить масштаб: {e}")
        
        self.canvas_3d.draw()

        # Настройка 2D для анимации
        self.fig_2d.clear()
        self.ax2d = self.fig_2d.add_subplot(111)
        # Берём выбранную проекцию
        x_idx, y_idx = self.anim2d_x_idx, self.anim2d_y_idx
        full_x = self.trajectory[:, x_idx]
        full_y = self.trajectory[:, y_idx]
        
        # Фильтруем конечные значения
        finite_mask = np.isfinite(full_x) & np.isfinite(full_y)
        full_x_clean = full_x[finite_mask]
        full_y_clean = full_y[finite_mask]
        
        # Устанавливаем пределы так, чтобы вся траектория была в рамках с небольшим запасом
        if full_x_clean.size > 0 and full_y_clean.size > 0:
            margin_x = 0.05 * (full_x_clean.max() - full_x_clean.min() or 1.0)
            margin_y = 0.05 * (full_y_clean.max() - full_y_clean.min() or 1.0)
            cx = 0.5 * (full_x_clean.max() + full_x_clean.min())
            cy = 0.5 * (full_y_clean.max() + full_y_clean.min())
            half_range_x = 0.5 * (full_x_clean.max() - full_x_clean.min() + 2 * margin_x) * self.scale_factor_2d
            half_range_y = 0.5 * (full_y_clean.max() - full_y_clean.min() + 2 * margin_y) * self.scale_factor_2d
            self.ax2d.set_xlim(cx - half_range_x, cx + half_range_x)
            self.ax2d.set_ylim(cy - half_range_y, cy + half_range_y)

        labels = ['x', 'y', 'z']
        lbl_x = labels[x_idx] if x_idx < len(labels) else f'axis {x_idx}'
        lbl_y = labels[y_idx] if y_idx < len(labels) else f'axis {y_idx}'
        self.ax2d.set_xlabel(lbl_x)
        self.ax2d.set_ylabel(lbl_y)
        self.ax2d.set_title(f"{self.current_system.name} - отрисовка 2D ({lbl_x}-{lbl_y})")
        self.ax2d.grid(True, alpha=0.3)
        self.live_line_2d, = self.ax2d.plot([], [], 'b-', linewidth=0.8, alpha=0.9)
        self.canvas_2d.draw()

        self.live_step = max(1, len(self.trajectory) // 300)
        self.live_timer.start()
        self.animate_button.setEnabled(False)
        self.pause_animation_button.setEnabled(True)
        self.resume_animation_button.setEnabled(False)
        self.update_controls_visibility()

    def update_live_frame(self):
        """Обновляет кадр анимации."""
        if self.trajectory is None or self.live_line is None:
            self.stop_live_draw()
            return
        
        if self.live_index >= len(self.trajectory):
            self.stop_live_draw()
            return
        
        segment = self.trajectory[:self.live_index]
        # Обновляем 3D линию
        self.live_line.set_data(segment[:, 0], segment[:, 1])
        self.live_line.set_3d_properties(segment[:, 2])

        # Обновляем 2D линию (выбранная проекция)
        if self.live_line_2d is not None:
            x_idx, y_idx = self.anim2d_x_idx, self.anim2d_y_idx
            self.live_line_2d.set_data(segment[:, x_idx], segment[:, y_idx])

        self.live_index += self.live_step
        self.apply_axis_scale(self.ax3d,
                              self.trajectory[:, 0],
                              self.trajectory[:, 1],
                              self.trajectory[:, 2])
        self.canvas_2d.draw()
        self.canvas_3d.draw()

    def stop_live_draw(self):
        """Останавливает анимацию 3D."""
        if self.live_timer.isActive():
            self.live_timer.stop()
        self.live_line = None
        self.live_line_2d = None
        if hasattr(self, "animate_button"):
            self.animate_button.setEnabled(True)
        if hasattr(self, "pause_animation_button"):
            self.pause_animation_button.setEnabled(False)
        if hasattr(self, "resume_animation_button"):
            self.resume_animation_button.setEnabled(False)
        self.update_controls_visibility()

    def pause_live_draw(self):
        """Ставит анимацию на паузу."""
        if self.live_timer.isActive():
            self.live_timer.stop()
        if hasattr(self, "resume_animation_button"):
            self.resume_animation_button.setEnabled(True)
        if hasattr(self, "pause_animation_button"):
            self.pause_animation_button.setEnabled(False)

    def resume_live_draw(self):
        """Продолжает анимацию с текущей позиции."""
        if self.trajectory is None or self.live_line is None:
            # если анимация ещё не запускалась, стартуем с нуля
            self.start_live_draw()
            return
        if not self.live_timer.isActive():
            self.live_timer.start()
        if hasattr(self, "resume_animation_button"):
            self.resume_animation_button.setEnabled(False)
        if hasattr(self, "pause_animation_button"):
            self.pause_animation_button.setEnabled(True)

    def apply_axis_scale(self, ax, x, y, z):
        """Применяет масштабирование осей 3D по текущему scale_factor."""
        try:
            # Проверка на NaN/Inf значения
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
                self.logger.warning("apply_axis_scale: обнаружены NaN/Inf значения, используются автопределы")
                return
            
            # Вычисляем диапазоны с защитой от нулевых значений
            x_range = np.ptp(x)
            y_range = np.ptp(y)
            z_range = np.ptp(z)
            
            # Если какой-то диапазон близок к нулю, используем 1.0 как минимум
            max_range = max(
                x_range if x_range > 1e-10 else 1.0,
                y_range if y_range > 1e-10 else 1.0,
                z_range if z_range > 1e-10 else 1.0
            ) * self.scale_factor
            
            # Вычисляем центры
            cx = np.mean(x)
            cy = np.mean(y)
            cz = np.mean(z)
            
            # Проверка что центры конечны
            if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(cz)):
                self.logger.warning("apply_axis_scale: центры содержат NaN/Inf")
                return
            
            ax.set_xlim(cx - max_range/2, cx + max_range/2)
            ax.set_ylim(cy - max_range/2, cy + max_range/2)
            ax.set_zlim(cz - max_range/2, cz + max_range/2)
        except Exception as e:
            self.logger.error(f"Критическая ошибка в apply_axis_scale: {e}")
            raise

    def on_scale_change(self, value):
        """Обновляет фактор масштаба 3D в реальном времени."""
        self.scale_factor = value / 100.0
        self.scale_label.setText(f"{value}%")
        if self.trajectory is not None and self.canvas_3d:
            # обновляем текущие оси, если есть
            if hasattr(self, "ax3d") and self.ax3d:
                self.apply_axis_scale(self.ax3d,
                                      self.trajectory[:, 0],
                                      self.trajectory[:, 1],
                                      self.trajectory[:, 2])
                self.canvas_3d.draw()

    def on_scale2d_change(self, value):
        """Обновляет масштаб 2D фазового портрета в реальном времени."""
        self.scale_factor_2d = value / 100.0
        self.scale2d_label.setText(f"{value}%")
        if self.trajectory is not None:
            # перерасчёт пределов по той же логике, что и при старте анимации 2D
            if hasattr(self, "ax2d") and self.ax2d:
                x_idx, y_idx = self.anim2d_x_idx, self.anim2d_y_idx
                full_x = self.trajectory[:, x_idx]
                full_y = self.trajectory[:, y_idx]
                if full_x.size > 0 and full_y.size > 0:
                    margin_x = 0.05 * (full_x.max() - full_x.min() or 1.0)
                    margin_y = 0.05 * (full_y.max() - full_y.min() or 1.0)
                    cx = 0.5 * (full_x.max() + full_x.min())
                    cy = 0.5 * (full_y.max() + full_y.min())
                    half_range_x = 0.5 * (full_x.max() - full_x.min() + 2 * margin_x) * self.scale_factor_2d
                    half_range_y = 0.5 * (full_y.max() - full_y.min() + 2 * margin_y) * self.scale_factor_2d
                    self.ax2d.set_xlim(cx - half_range_x, cx + half_range_x)
                    self.ax2d.set_ylim(cy - half_range_y, cy + half_range_y)
                    self.canvas_2d.draw()

    def on_speed_change(self, value):
        """Меняет скорость анимации (0.5x - 5x)."""
        self.speed_factor = value / 10.0
        self.speed_label.setText(f"{self.speed_factor:.1f}x")
        base_interval = 30
        new_interval = max(5, int(base_interval / self.speed_factor))
        self.live_timer.setInterval(new_interval)

    def on_anim2d_projection_change(self, index: int):
        """Меняет проекцию для 2D анимации."""
        data = self.anim2d_combo.itemData(index)
        if isinstance(data, tuple) and len(data) == 2:
            self.anim2d_x_idx, self.anim2d_y_idx = data
            # если уже есть траектория и сейчас не идёт анимация – просто перерисуем 2D
            if self.trajectory is not None and self.live_line is None:
                self.render_2d_plot()

    def update_controls_visibility(self):
        """Показывает элементы управления в зависимости от активной вкладки."""
        current = self.tabs.currentWidget()
        # 2D вкладка: tab2d_widget (создана в setup_ui), 3D вкладка: canvas_3d
        is_2d = current is self.tabs.widget(0)
        is_3d = current is self.canvas_3d

        # Проекция только на 2D
        self.anim2d_group.setVisible(is_2d)
        self.scale2d_group.setVisible(is_2d)  # Масштаб 2D только на вкладке 2D
        # Анимация и скорость доступны и на 2D, и на 3D
        show_anim_controls = is_2d or is_3d
        self.speed_group.setVisible(show_anim_controls)
        self.scale_group.setVisible(is_3d)  # Масштаб 3D только на вкладке 3D
        self.animate_button.setVisible(show_anim_controls)
        self.pause_animation_button.setVisible(show_anim_controls)
        self.resume_animation_button.setVisible(show_anim_controls)

        # Если нет данных – блокируем запуск анимации
        has_data = self.trajectory is not None and len(self.trajectory) > 0
        self.animate_button.setEnabled(show_anim_controls and has_data)

    def apply_app_styles(self):
        """Устанавливает базовые стили приложения."""
        palette_styles = """
        QPushButton {
            background-color: #2563eb;
            color: white;
            border-radius: 6px;
            padding: 8px 10px;
        }
        QPushButton:hover {
            background-color: #1d4ed8;
        }
        QPushButton:disabled {
            background-color: #9ca3af;
        }
        QGroupBox {
            border: 1px solid #d0d7e2;
            border-radius: 6px;
            margin-top: 8px;
            font-weight: 600;
            padding: 6px;
            background: #f9fafb;
        }
        QGroupBox:title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QLabel {
            color: #1f2933;
        }
        QTabWidget::pane { border: 1px solid #d0d7e2; }
        QTabBar::tab {
            background: #e5e7eb;
            padding: 6px 10px;
            border-radius: 4px;
            margin-right: 2px;
        }
        QTabBar::tab:selected { background: #ffffff; }
        QSlider::groove:horizontal {
            height: 6px;
            background: #e5e7eb;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #2563eb;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        """
        self.setStyleSheet(palette_styles)

