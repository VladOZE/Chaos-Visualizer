# src/__init__.py
"""
Chaos Visualizer - приложение для визуализации хаотических динамических систем.
"""

__version__ = "1.0.0"
__author__ = "Lebedev Vladislav"

from .models.lorenz import LorenzSystem
from .models.rossler import RosslerSystem
from .models.chua import ChuaSystem
from .models.custom_system import CustomEquationSystem
from .integration.integrator import ODEIntegrator, FixedStepIntegrator
from .visualization.matplotlib_plots import MatplotlibPlotter
from .visualization.plotly_plots import PlotlyVisualizer
from .storage.exporter import ResultsExporter
