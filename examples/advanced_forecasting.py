"""
Advanced example demonstrating complex forecasting capabilities
including seasonality detection, demand prediction, and ensemble modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from warehouse_optimization.forecasting import (
    DemandForecaster,
    EnsembleModel,
    SeasonalityAnalyzer,
    ForecastConfig
)
from warehouse_optimization.utils import (
    DataProcessor,
    DataValidator
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_seasonal_data(days: int = 365) -> pd.DataFrame:
    """
    Generate sample data with multiple seasonal patterns
    
    Args:
        days: Number of days of data to generate
        
    Returns:
        DataFrame with seasonal demand patterns
    """
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='H'
    )
    
    # Generate different seasonal components
    hourly_pattern = np.sin(np.pi * dates.hour / 12) + 1  # Daily cycle
    weekly_pattern = np.sin(np.pi * dates.dayofweek / 3.5) + 1  # Weekly cycle
    monthly_pattern = np.sin(np.pi * dates.day / 15) + 1  # Monthly cycle
    yearly_pattern = np.sin(np.pi * dates.dayofyear / 182.5) + 1  # Yearly cycle
    
    # Combine patterns with different weights
    base_demand = (
        0.4 * hourly_pattern +
        0.3 * weekly_pattern +
        0.2 * monthly_pattern +
        0.1 * yearly_pattern
    )
    
    # Add trend and noise
    trend = np.linspace(0, 2, len(dates))
    noise = np.random.normal(0, 0.1, len(dates))
    
    demand = base_demand + trend + noise
    demand = np.maximum(demand, 0)  # Ensure non-negative
    
    # Create multiple products with different patterns
    products = []
    for product_id in range(1, 6):
        product_demand = demand * np.random.uniform(0.5, 1.5)
        
        # Add random spikes (e.g., promotions)
        spikes = np.random.choice(
            [1, 3],
            size=len(dates),
            p=[0.99, 0.01]
        )
        product_demand = product_demand * spikes
        
        products.append(pd.DataFrame({
            'timestamp': dates,
            'product_id': product_id,
            'quantity': product_demand
        }))
    
    return pd.concat(products, ignore_index=True)

def analyze_seasonality(data: pd.DataFrame) -> dict:
    """
    Analyze seasonal patterns in the data
    
    Args:
        data: Historical demand data
        
    Returns:
        Dictionary of seasonal patterns and strengths
    """
    analyzer = SeasonalityAnalyzer()
    
    patterns = {}
    for product_id in data['product_id'].unique():
        product_data = data[data['product_id'] == product_id]
        ts = pd.Series(
            product_data['quantity'].values,
            index=product_data['timestamp']
        )
        
        patterns[product_id] = analyzer.analyze_seasonality(ts)
    
    return patterns

def create_ensemble_forecast(
    data: pd.DataFrame,
    forecast_horizon: int = 7
) -> tuple:
    """
    Create ensemble forecast using multiple models
    
    Args:
        data: Historical demand data
        forecast_horizon: Number of days to forecast
        
    Returns:
        Tuple of (forecasts, model_weights)
    """
    # Configure forecaster
    config = ForecastConfig(
        forecast_horizon=forecast_horizon,
        seasonality_mode='multiplicative',
        use_ensemble=True
    )
    
    forecaster = DemandForecaster(data, config)
    
    # Generate forecasts for each product
    forecasts = {}
    model_weights = {}
    
    for product_id in data['product_id'].unique():
        forecast = forecaster.generate_forecast(product_id)
        forecasts[product_id] = forecast
        
        # Get model weights if using ensemble
        if config.use_ensemble:
            model_weights[product_id] = forecaster.get_model_weights(product_id)
    
    return forecasts, model_weights

def visualize_results(
    data: pd.DataFrame,
    forecasts: dict,
    seasonality: dict
) -> go.Figure:
    """
    Create visualization of forecasting results
    
    Args:
        data: Historical data
        forecasts: Generated forecasts
        seasonality: Detected seasonal patterns
        
    Returns:
        Plotly figure with visualizations
    """
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            'Historical Demand & Forecast',
            'Seasonal Patterns',
            'Forecast Accuracy',
            'Model Weights',
            'Component Decomposition',
            'Prediction Intervals'
        )
    )
    
    # Plot historical data and forecast
    product_id = 1  # Example with first product
    product_data = data[data['product_id'] == product_id]
    forecast = forecasts[product_id]
    
    fig.add_trace(
        go.Scatter(
            x=product_data['timestamp'],
            y=product_data['quantity'],
            name='Historical',
            mode='lines'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=pd.date_range(
                start=product_data['timestamp'].max(),
                periods=len(forecast.forecast_values),
                freq='D'
            ),
            y=forecast.forecast_values,
            name='Forecast',
            mode='lines',
            line=dict(dash='dash')
        ),
        row=1, col=1
    )
    
    # Plot seasonal patterns
    patterns = seasonality[product_id]
    for pattern_type, pattern in patterns.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pattern.pattern_values))),
                y=pattern.pattern_values,
                name=f'{pattern_type} Pattern',
                mode='lines'
            ),
            row=1, col=2
        )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=pd.date_range(
                start=product_data['timestamp'].max(),
                periods=len(forecast.forecast_values),
                freq='D'
            ),
            y=forecast.confidence_intervals[:, 0],
            fill=None,
            mode='lines',
            line=dict(color='gray'),
            name='Lower CI'
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=pd.date_range(
                start=product_data['timestamp'].max(),
                periods=len(forecast.forecast_values),
                freq='D'
            ),
            y=forecast.confidence_intervals[:, 1],
            fill='tonexty',
            mode='lines',
            line=dict(color='gray'),
            name='Upper CI'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Demand Forecasting Analysis"
    )
    
    return fig

def main():
    """Main execution function"""
    logger.info("Starting advanced forecasting example")
    
    # Generate sample data
    logger.info("Generating seasonal data...")
    data = generate_seasonal_data()
    
    # Initialize processor and validate data
    logger.info("Processing and validating data...")
    processor = DataProcessor()
    validator = DataValidator()
    
    processed_data = processor.preprocess_order_data(data)
    is_valid, violations = validator.validate_order_data(processed_data)
    if not is_valid:
        logger.error(f"Data validation failed: {violations}")
        return
    
    # Analyze seasonality
    logger.info("Analyzing seasonal patterns...")
    seasonality_patterns = analyze_seasonality(processed_data)
    
    # Generate forecasts
    logger.info("Generating ensemble forecasts...")
    forecasts, model_weights = create_ensemble_forecast(processed_data)
    
    # Create visualization
    logger.info("Creating visualizations...")
    fig = visualize_results(processed_data, forecasts, seasonality_patterns)
    
    # Display results
    logger.info("\nForecasting Results:")
    for product_id, forecast in forecasts.items():
        logger.info(f"\nProduct {product_id}:")
        logger.info(f"Mean forecast: {np.mean(forecast.forecast_values):.2f}")
        logger.info(f"Trend factor: {forecast.trend_factor:.2f}")
        logger.info("Seasonal factors:")
        for factor, value in forecast.seasonality_factors.items():
            logger.info(f"  {factor}: {value:.2f}")
    
    logger.info("\nVisualization created - you can display the fig using your preferred plotting library")
    
    # Example of displaying with plotly
    # fig.show()

if __name__ == "__main__":
    main()

# This advanced forecasting example demonstrates:

# 1. Complex Data Generation:
# - Multiple seasonal patterns
# - Trend components
# - Random events/spikes
# - Product-specific patterns

# 2. Advanced Analysis:
# - Seasonality detection
# - Pattern strength analysis
# - Ensemble forecasting
# - Confidence intervals

# 3. Visualization:
# - Historical data and forecasts
# - Seasonal patterns
# - Model performance
# - Prediction intervals

# 4. Key Features:
# - Multiple forecasting models
# - Pattern decomposition
# - Model weight optimization
# - Confidence bounds
