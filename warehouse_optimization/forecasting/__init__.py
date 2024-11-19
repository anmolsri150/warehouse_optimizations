"""
Forecasting module for warehouse optimization system.
Provides demand forecasting, seasonality analysis, and ensemble modeling capabilities.
"""

from .demand_forecaster import DemandForecaster, ForecastConfig
from .ensemble_model import EnsembleModel, ModelWeight
from .seasonality import (
    SeasonalityAnalyzer,
    SeasonalityType,
    SeasonalPattern
)

# Version information
__version__ = "0.1.0"

# Module level docstring
__doc__ = """
Warehouse Optimization Forecasting Module
=======================================

This module provides comprehensive forecasting capabilities:

1. Demand Forecasting:
   - Multi-model ensemble forecasting
   - Advanced seasonality detection
   - Confidence interval estimation
   - Trend analysis

2. Ensemble Modeling:
   - Dynamic model weighting
   - Performance tracking
   - Model combination strategies
   - Automated weight optimization

3. Seasonality Analysis:
   - Multiple seasonality patterns
   - Pattern strength estimation
   - Statistical validation
   - Pattern combination

Usage:
------
from warehouse_optimization.forecasting import (
    DemandForecaster,
    EnsembleModel,
    SeasonalityAnalyzer
)

# Initialize forecaster with configuration
config = ForecastConfig(
    forecast_horizon=7,
    seasonality_mode='multiplicative'
)
forecaster = DemandForecaster(historical_data, config)

# Generate forecast
forecast = forecaster.generate_forecast(product_id)

# Analyze seasonality
analyzer = SeasonalityAnalyzer()
patterns = analyzer.analyze_seasonality(time_series_data)
"""

# List of public objects
__all__ = [
    # Main classes
    'DemandForecaster',
    'EnsembleModel',
    'SeasonalityAnalyzer',
    
    # Supporting classes
    'ForecastConfig',
    'ModelWeight',
    'SeasonalityType',
    'SeasonalPattern'
]

# Module metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__status__ = "Development"

# Default configurations
DEFAULT_FORECAST_CONFIG = {
    'forecast_horizon': 7,
    'seasonality_mode': 'multiplicative',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
    'min_history_days': 30,
    'confidence_interval': 0.95,
    'use_ensemble': True,
    'enable_event_detection': True,
    'refit_frequency': 24
}

DEFAULT_ENSEMBLE_CONFIG = {
    'performance_window': 30,
    'min_weight': 0.1,
    'enable_dynamic_weights': True,
    'validation_split': 0.2
}

DEFAULT_SEASONALITY_CONFIG = {
    'min_pattern_strength': 0.1,
    'confidence_threshold': 0.95,
    'enable_automatic_detection': True,
    'max_patterns': 3
}

def create_forecasting_suite(
    historical_data,
    custom_configs: dict = None
) -> tuple:
    """
    Create a complete forecasting suite with default or custom configurations.
    
    Args:
        historical_data: Historical demand data
        custom_configs: Optional custom configurations
        
    Returns:
        Tuple of (demand_forecaster, ensemble_model, seasonality_analyzer)
    """
    # Initialize configurations
    configs = {
        'forecast': DEFAULT_FORECAST_CONFIG.copy(),
        'ensemble': DEFAULT_ENSEMBLE_CONFIG.copy(),
        'seasonality': DEFAULT_SEASONALITY_CONFIG.copy()
    }
    
    # Update with custom configs if provided
    if custom_configs:
        for config_type, config in custom_configs.items():
            if config_type in configs:
                configs[config_type].update(config)
    
    # Create components
    forecast_config = ForecastConfig(**configs['forecast'])
    forecaster = DemandForecaster(
        historical_data=historical_data,
        config=forecast_config
    )
    
    ensemble = EnsembleModel(**configs['ensemble'])
    
    seasonality = SeasonalityAnalyzer(**configs['seasonality'])
    
    return forecaster, ensemble, seasonality

def validate_forecast_data(data) -> bool:
    """
    Validate input data for forecasting.
    
    Args:
        data: Input data to validate
        
    Returns:
        bool: True if data is valid
    """
    try:
        # Check required columns
        required_columns = {'timestamp', 'product_id', 'quantity'}
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            return False
            
        if not pd.api.types.is_numeric_dtype(data['quantity']):
            return False
            
        # Check for negative quantities
        if (data['quantity'] < 0).any():
            return False
            
        # Check for duplicate entries
        if data.duplicated(['timestamp', 'product_id']).any():
            return False
            
        return True
        
    except Exception:
        return False

def get_forecast_metrics(
    forecaster: DemandForecaster,
    product_id: int
) -> dict:
    """
    Get comprehensive forecast metrics for a product.
    
    Args:
        forecaster: Initialized DemandForecaster
        product_id: Product ID to analyze
        
    Returns:
        dict: Forecast metrics
    """
    metrics = {}
    
    # Generate forecast
    forecast = forecaster.generate_forecast(product_id)
    
    # Calculate metrics
    metrics['mean_demand'] = np.mean(forecast.forecast_values)
    metrics['std_demand'] = np.std(forecast.forecast_values)
    metrics['trend_factor'] = forecast.trend_factor
    metrics['seasonality_factors'] = forecast.seasonality_factors
    
    # Add confidence intervals
    metrics['confidence_intervals'] = {
        'lower': forecast.confidence_intervals[:, 0],
        'upper': forecast.confidence_intervals[:, 1]
    }
    
    return metrics

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def initialize():
    """Initialize the forecasting module."""
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing warehouse_optimization.forecasting v{__version__}")

# Run initialization when module is imported
initialize()

# This `__init__.py` provides:

# 1. Module Organization:
# - Clear imports
# - Version information
# - Comprehensive documentation
# - Usage examples

# 2. Default Configurations:
# - Forecast settings
# - Ensemble parameters
# - Seasonality configurations
# - Easy customization

# 3. Utility Functions:
# - Suite creation
# - Data validation
# - Metrics calculation
# - Logging setup

# 4. Integration Features:
# - Complete forecasting suite
# - Component coordination
# - Configuration management
# - Error handling

# 5. Documentation:
# - Module overview
# - Component descriptions
# - Usage instructions
# - Configuration guidelines

# This initialization file makes it easy to:
# 1. Import all forecasting components
# 2. Use default configurations
# 3. Create complete forecasting suites
# 4. Validate input data
# 5. Access metrics and utilities