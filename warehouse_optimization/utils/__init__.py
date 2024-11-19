"""
Utilities module for warehouse optimization system.
Provides data processing, metrics calculation, validation, and common utilities.
"""

from .data_processing import (
    DataProcessor,
    DataPreprocessConfig
)
from .metrics import (
    WarehouseMetrics,
    MetricsConfig
)
from .validation import (
    DataValidator,
    ValidationConfig
)

# Version information
__version__ = "0.1.0"

# Module level docstring
__doc__ = """
Warehouse Optimization Utilities Module
====================================

This module provides utility functions and tools:

1. Data Processing:
   - Data preprocessing
   - Feature engineering
   - Data transformation
   - Data cleaning

2. Metrics:
   - Performance metrics
   - Optimization metrics
   - Utilization metrics
   - Statistical analysis

3. Validation:
   - Data validation
   - Constraint checking
   - Input verification
   - State validation

Usage Example:
------------
from warehouse_optimization.utils import (
    DataProcessor,
    WarehouseMetrics,
    DataValidator
)

# Initialize utilities
processor = DataProcessor()
metrics = WarehouseMetrics()
validator = DataValidator()

# Process data
processed_data = processor.preprocess_order_data(order_data)

# Calculate metrics
performance_metrics = metrics.calculate_optimization_metrics(
    warehouse_state,
    picking_routes,
    current_time
)

# Validate data
is_valid, violations = validator.validate_warehouse_state(warehouse_state)
"""

# List of public objects
__all__ = [
    # Main classes
    'DataProcessor',
    'WarehouseMetrics',
    'DataValidator',
    
    # Configuration classes
    'DataPreprocessConfig',
    'MetricsConfig',
    'ValidationConfig'
]

# Module metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__status__ = "Development"

# Default configurations
DEFAULT_PREPROCESS_CONFIG = {
    'remove_outliers': True,
    'outlier_threshold': 3.0,
    'fill_missing': True,
    'min_data_points': 30,
    'time_aggregation': '1D',
    'normalize_features': True,
    'encode_categorical': True
}

DEFAULT_METRICS_CONFIG = {
    'time_window': 30,  # days
    'moving_average_window': 24,  # hours
    'distance_unit': 'meters',
    'time_unit': 'seconds',
    'include_historical': True,
    'confidence_level': 0.95
}

DEFAULT_VALIDATION_CONFIG = {
    'max_weight_per_shelf': 500.0,  # kg
    'max_items_per_bin': 50,
    'min_gap_between_items': 0.02,  # meters
    'max_stack_height': 2.0,  # meters
}

def create_utils_suite(
    custom_configs: dict = None
) -> tuple:
    """
    Create a complete utilities suite with default or custom configurations.
    
    Args:
        custom_configs: Optional custom configurations for components
        
    Returns:
        Tuple of (data_processor, metrics, validator)
    """
    configs = {
        'preprocess': DEFAULT_PREPROCESS_CONFIG.copy(),
        'metrics': DEFAULT_METRICS_CONFIG.copy(),
        'validation': DEFAULT_VALIDATION_CONFIG.copy()
    }
    
    # Update with custom configs if provided
    if custom_configs:
        for component, config in custom_configs.items():
            if component in configs:
                configs[component].update(config)
    
    # Create utility components
    processor = DataProcessor(
        DataPreprocessConfig(**configs['preprocess'])
    )
    metrics = WarehouseMetrics(
        MetricsConfig(**configs['metrics'])
    )
    validator = DataValidator(
        ValidationConfig(**configs['validation'])
    )
    
    return processor, metrics, validator

# Common utility functions
def validate_numeric_range(
    value: float,
    min_value: float = None,
    max_value: float = None
) -> bool:
    """Validate numeric value within range"""
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True

def calculate_euclidean_distance(
    point1: tuple,
    point2: tuple
) -> float:
    """Calculate Euclidean distance between points"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def format_duration(
    seconds: float,
    include_milliseconds: bool = False
) -> str:
    """Format duration in seconds to human-readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if include_milliseconds:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"

def format_distance(
    meters: float,
    unit: str = 'meters',
    precision: int = 2
) -> str:
    """Format distance in meters to specified unit"""
    conversions = {
        'meters': 1,
        'feet': 3.28084,
        'yards': 1.09361,
        'kilometers': 0.001
    }
    
    if unit not in conversions:
        raise ValueError(f"Unknown unit: {unit}")
        
    converted = meters * conversions[unit]
    return f"{converted:.{precision}f} {unit}"

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def initialize():
    """Initialize the utils module."""
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing warehouse_optimization.utils v{__version__}")

# Run initialization when module is imported
initialize()

# Import common dependencies
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# This `__init__.py` provides:

# 1. Module Organization:
# - Clear imports
# - Version information
# - Comprehensive documentation
# - Usage examples

# 2. Default Configurations:
# - Preprocessing settings
# - Metrics settings
# - Validation settings
# - Easy customization

# 3. Utility Functions:
# - Suite creation
# - Common calculations
# - Formatting helpers
# - Type validation

# 4. Integration Features:
# - Component coordination
# - Configuration management
# - Logging setup
# - Error handling

# 5. Documentation:
# - Module overview
# - Component descriptions
# - Usage instructions
# - Configuration guidelines
