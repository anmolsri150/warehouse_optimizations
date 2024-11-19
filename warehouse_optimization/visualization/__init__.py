"""
Visualization module for warehouse optimization system.
Provides comprehensive visualization tools for warehouse analytics and monitoring.
"""

from .dashboard import (
    WarehouseDashboard
)
from .heatmap import (
    WarehouseHeatmap,
    HeatmapConfig
)
from .layout import (
    WarehouseLayout,
    LayoutConfig,
    ViewMode
)
from .performance import (
    PerformanceVisualizer,
    PerformanceConfig
)
import plotly.graph_objects as go

# Version information
__version__ = "0.1.0"

# Module level docstring
__doc__ = """
Warehouse Optimization Visualization Module
========================================

This module provides comprehensive visualization tools:

1. Dashboard:
   - Real-time monitoring
   - Interactive dashboards
   - Multi-metric displays
   - Custom layouts

2. Heatmaps:
   - Picking frequency
   - Space utilization
   - Traffic patterns
   - Demand visualization

3. Layout Views:
   - 2D/3D layouts
   - Interactive floor plans
   - Zone visualization
   - Bin status tracking

4. Performance Analytics:
   - Metric tracking
   - Trend analysis
   - Performance distribution
   - Anomaly detection

Usage Example:
------------
from warehouse_optimization.visualization import (
    WarehouseDashboard,
    WarehouseHeatmap,
    WarehouseLayout,
    PerformanceVisualizer
)

# Create visualization components
dashboard = WarehouseDashboard()
heatmap = WarehouseHeatmap()
layout = WarehouseLayout()
performance = PerformanceVisualizer()

# Create comprehensive visualization
dashboard_fig = dashboard.create_dashboard(warehouse_state, metrics)
heatmap_fig = heatmap.create_picking_heatmap(warehouse_state, picking_data)
layout_fig = layout.create_layout_view(warehouse_state)
perf_fig = performance.create_performance_dashboard(current_metrics)
"""

# List of public objects
__all__ = [
    # Main classes
    'WarehouseDashboard',
    'WarehouseHeatmap',
    'WarehouseLayout',
    'PerformanceVisualizer',
    
    # Configuration classes
    'HeatmapConfig',
    'LayoutConfig',
    'PerformanceConfig',
    
    # Enums
    'ViewMode'
]

# Module metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__status__ = "Development"

# Default configurations
DEFAULT_DASHBOARD_CONFIG = {
    'update_interval': 5000,  # milliseconds
    'dark_mode': False,
    'enable_3d': True
}

DEFAULT_HEATMAP_CONFIG = {
    'colorscale': 'Viridis',
    'show_labels': True,
    'label_format': '.2f',
    'opacity': 0.8,
    'enable_hover': True
}

DEFAULT_LAYOUT_CONFIG = {
    'view_mode': ViewMode.ISOMETRIC,
    'show_labels': True,
    'show_grid': True,
    'show_dimensions': True,
    'zone_opacity': 0.7,
    'highlight_conflicts': True
}

DEFAULT_PERFORMANCE_CONFIG = {
    'time_window': 30,  # days
    'update_interval': 300,  # seconds
    'moving_average_window': 24,  # hours
    'show_targets': True,
    'show_predictions': True,
    'highlight_anomalies': True
}

def create_visualization_suite(
    custom_configs: dict = None
) -> tuple:
    """
    Create a complete visualization suite with default or custom configurations.
    
    Args:
        custom_configs: Optional custom configurations for components
        
    Returns:
        Tuple of (dashboard, heatmap, layout, performance)
    """
    configs = {
        'dashboard': DEFAULT_DASHBOARD_CONFIG.copy(),
        'heatmap': DEFAULT_HEATMAP_CONFIG.copy(),
        'layout': DEFAULT_LAYOUT_CONFIG.copy(),
        'performance': DEFAULT_PERFORMANCE_CONFIG.copy()
    }
    
    # Update with custom configs if provided
    if custom_configs:
        for component, config in custom_configs.items():
            if component in configs:
                configs[component].update(config)
    
    # Create visualization components
    dashboard = WarehouseDashboard(**configs['dashboard'])
    heatmap = WarehouseHeatmap(HeatmapConfig(**configs['heatmap']))
    layout = WarehouseLayout(LayoutConfig(**configs['layout']))
    performance = PerformanceVisualizer(PerformanceConfig(**configs['performance']))
    
    return dashboard, heatmap, layout, performance

def create_combined_view(
    warehouse_state,
    metrics,
    picking_data=None,
    highlight_bins=None
) -> dict[str, go.Figure]:
    """
    Create all standard visualizations for the current warehouse state.
    
    Args:
        warehouse_state: Current warehouse state
        metrics: Current optimization metrics
        picking_data: Optional picking data for heatmap
        highlight_bins: Optional list of bins to highlight
        
    Returns:
        Dictionary of visualization figures
    """
    suite = create_visualization_suite()
    dashboard, heatmap, layout, performance = suite
    
    return {
        'dashboard': dashboard.create_dashboard(warehouse_state, metrics),
        'heatmap': (
            heatmap.create_picking_heatmap(warehouse_state, picking_data)
            if picking_data is not None else None
        ),
        'layout': layout.create_layout_view(warehouse_state, highlight_bins),
        'performance': performance.create_performance_dashboard(metrics, warehouse_state)
    }

# Utility functions for common visualization tasks
def create_animation(
    figure: go.Figure,
    duration: int = 500,
    transition: dict = None
) -> go.Figure:
    """Add animation capabilities to a figure"""
    transition = transition or {
        'duration': duration,
        'easing': 'cubic-in-out'
    }
    
    figure.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                dict(label='Play',
                     method='animate',
                     args=[None, {'frame': {'duration': duration, 'redraw': True},
                                'fromcurrent': True,
                                'transition': transition}]),
                dict(label='Pause',
                     method='animate',
                     args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate',
                                  'transition': {'duration': 0}}])
            ]
        }]
    )
    return figure

def apply_theme(figure: go.Figure, dark_mode: bool = False) -> go.Figure:
    """Apply consistent theme to visualization"""
    template = 'plotly_dark' if dark_mode else 'plotly_white'
    figure.update_layout(template=template)
    return figure

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def initialize():
    """Initialize the visualization module."""
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing warehouse_optimization.visualization v{__version__}")

# Run initialization when module is imported
initialize()

# This `__init__.py` provides:

# 1. Module Organization:
# - Clear imports
# - Version information
# - Comprehensive documentation
# - Usage examples

# 2. Default Configurations:
# - Dashboard settings
# - Heatmap settings
# - Layout settings
# - Performance settings

# 3. Utility Functions:
# - Suite creation
# - Combined views
# - Animation support
# - Theme application

# 4. Integration Features:
# - Component coordination
# - Configuration management
# - Common visualizations
# - Animation support

# 5. Documentation:
# - Module overview
# - Component descriptions
# - Usage instructions
# - Configuration guidelines

# This initialization file makes it easy to:
# 1. Import all visualization components
# 2. Use default configurations
# 3. Create complete visualization suites
# 4. Apply consistent styling
# 5. Generate combined views
