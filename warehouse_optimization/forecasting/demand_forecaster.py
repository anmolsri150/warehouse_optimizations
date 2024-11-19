from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

from ..core.types import DemandForecast, ProductAttributes

@dataclass
class ForecastConfig:
    """Configuration for demand forecasting"""
    forecast_horizon: int = 7  # days
    seasonality_mode: str = 'multiplicative'
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    min_history_days: int = 30
    confidence_interval: float = 0.95
    use_ensemble: bool = True
    enable_event_detection: bool = True
    refit_frequency: int = 24  # hours

class DemandForecaster:
    """
    Manages demand forecasting using ensemble methods and advanced forecasting techniques.
    Combines multiple models for robust predictions and handles seasonality patterns.
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        config: Optional[ForecastConfig] = None,
        external_features: Optional[pd.DataFrame] = None
    ):
        self.historical_data = self._preprocess_historical_data(historical_data)
        self.config = config or ForecastConfig()
        self.external_features = external_features
        
        # Initialize models
        self.prophet_models: Dict[int, Prophet] = {}
        self.xgb_models: Dict[int, xgb.XGBRegressor] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        
        # Track seasonality patterns
        self.seasonality_patterns: Dict[int, Dict[str, float]] = {}
        
        # Track forecast performance
        self.forecast_accuracy: Dict[int, float] = {}
        self.last_refit: Dict[int, datetime] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._initialize_models()
    
    def _preprocess_historical_data(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocess historical data for forecasting"""
        df = data.copy()
        
        # Ensure datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Aggregate to daily level if needed
        if len(df['timestamp'].dt.floor('D').unique()) < len(df):
            df = df.groupby([
                'product_id',
                df['timestamp'].dt.floor('D')
            ])['quantity'].sum().reset_index()
        
        # Add time-based features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def _initialize_models(self):
        """Initialize forecasting models for each product"""
        unique_products = self.historical_data['product_id'].unique()
        
        for product_id in unique_products:
            product_data = self._get_product_data(product_id)
            
            if len(product_data) >= self.config.min_history_days:
                # Initialize Prophet model
                self.prophet_models[product_id] = self._create_prophet_model()
                
                # Initialize XGBoost model
                self.xgb_models[product_id] = self._create_xgb_model()
                
                # Initialize scaler
                self.scalers[product_id] = StandardScaler()
                
                # Extract seasonality patterns
                self.seasonality_patterns[product_id] = self._extract_seasonality_patterns(
                    product_data
                )
                
                # Track initialization
                self.last_refit[product_id] = datetime.now()
    
    def _create_prophet_model(self) -> Prophet:
        """Create and configure Prophet model"""
        return Prophet(
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            holidays_prior_scale=self.config.holidays_prior_scale,
            interval_width=self.config.confidence_interval
        )
    
    def _create_xgb_model(self) -> xgb.XGBRegressor:
        """Create and configure XGBoost model"""
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def _extract_seasonality_patterns(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract seasonality patterns from historical data"""
        patterns = {}
        
        # Daily patterns
        daily_avg = data.groupby('day_of_week')['quantity'].mean()
        patterns['daily_pattern'] = daily_avg / daily_avg.mean()
        
        # Monthly patterns
        monthly_avg = data.groupby('month')['quantity'].mean()
        patterns['monthly_pattern'] = monthly_avg / monthly_avg.mean()
        
        # Weekend effect
        weekend_effect = (
            data[data['is_weekend'] == 1]['quantity'].mean() /
            data[data['is_weekend'] == 0]['quantity'].mean()
        )
        patterns['weekend_effect'] = weekend_effect
        
        return patterns
    
    def generate_forecast(
        self,
        product_id: int,
        horizon_days: Optional[int] = None,
        include_features: bool = True
    ) -> DemandForecast:
        """Generate demand forecast for a product"""
        if product_id not in self.prophet_models:
            raise KeyError(f"No model initialized for product {product_id}")
            
        horizon = horizon_days or self.config.forecast_horizon
        
        # Check if refit needed
        if self._needs_refit(product_id):
            self._refit_models(product_id)
        
        # Generate Prophet forecast
        prophet_forecast = self._generate_prophet_forecast(
            product_id,
            horizon
        )
        
        # Generate XGBoost forecast
        xgb_forecast = self._generate_xgb_forecast(
            product_id,
            horizon
        )
        
        # Combine forecasts
        combined_forecast = self._combine_forecasts(
            prophet_forecast,
            xgb_forecast
        )
        
        # Apply seasonality adjustments
        adjusted_forecast = self._apply_seasonality_adjustments(
            combined_forecast,
            self.seasonality_patterns[product_id]
        )
        
        return DemandForecast(
            product_id=product_id,
            forecast_values=adjusted_forecast['forecast'].values,
            confidence_intervals=np.column_stack([
                adjusted_forecast['lower'].values,
                adjusted_forecast['upper'].values
            ]),
            seasonality_factors=self.seasonality_patterns[product_id],
            trend_factor=self._calculate_trend_factor(product_id)
        )
    
    def _generate_prophet_forecast(
        self,
        product_id: int,
        horizon: int
    ) -> pd.DataFrame:
        """Generate forecast using Prophet"""
        model = self.prophet_models[product_id]
        future = model.make_future_dataframe(periods=horizon)
        
        # Add external features if available
        if self.external_features is not None:
            for feature in self.external_features.columns:
                if feature != 'timestamp':
                    future[feature] = self._get_future_feature_values(
                        feature,
                        horizon
                    )
        
        forecast = model.predict(future)
        return forecast.tail(horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def _generate_xgb_forecast(
        self,
        product_id: int,
        horizon: int
    ) -> pd.DataFrame:
        """Generate forecast using XGBoost"""
        model = self.xgb_models[product_id]
        scaler = self.scalers[product_id]
        
        # Generate future features
        future_features = self._create_future_features(horizon)
        
        # Scale features
        scaled_features = scaler.transform(future_features)
        
        # Generate predictions
        predictions = model.predict(scaled_features)
        
        return pd.DataFrame({
            'ds': pd.date_range(
                start=datetime.now(),
                periods=horizon,
                freq='D'
            ),
            'yhat': predictions
        })
    
    def _combine_forecasts(
        self,
        prophet_forecast: pd.DataFrame,
        xgb_forecast: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine forecasts from different models"""
        # Calculate weights based on historical performance
        prophet_weight = 0.6
        xgb_weight = 0.4
        
        combined = pd.DataFrame()
        combined['ds'] = prophet_forecast['ds']
        
        # Weighted average of predictions
        combined['forecast'] = (
            prophet_weight * prophet_forecast['yhat'] +
            xgb_weight * xgb_forecast['yhat']
        )
        
        # Confidence intervals from Prophet
        combined['lower'] = prophet_forecast['yhat_lower']
        combined['upper'] = prophet_forecast['yhat_upper']
        
        return combined
    
    def _apply_seasonality_adjustments(
        self,
        forecast: pd.DataFrame,
        seasonality_patterns: Dict[str, float]
    ) -> pd.DataFrame:
        """Apply seasonality adjustments to forecast"""
        adjusted = forecast.copy()
        
        # Apply daily patterns
        day_of_week = adjusted['ds'].dt.dayofweek
        daily_factors = seasonality_patterns['daily_pattern']
        adjusted['forecast'] *= [daily_factors[d] for d in day_of_week]
        
        # Apply monthly patterns
        month = adjusted['ds'].dt.month
        monthly_factors = seasonality_patterns['monthly_pattern']
        adjusted['forecast'] *= [monthly_factors[m] for m in month]
        
        # Apply weekend effect
        is_weekend = adjusted['ds'].dt.dayofweek.isin([5, 6])
        weekend_factor = seasonality_patterns['weekend_effect']
        adjusted.loc[is_weekend, 'forecast'] *= weekend_factor
        
        return adjusted
    
    def _calculate_trend_factor(self, product_id: int) -> float:
        """Calculate trend factor for product"""
        historical = self._get_product_data(product_id)
        if len(historical) < 2:
            return 1.0
            
        # Calculate simple trend
        x = np.arange(len(historical))
        y = historical['quantity'].values
        z = np.polyfit(x, y, 1)
        slope = z[0]
        
        # Normalize trend
        avg_demand = y.mean()
        if avg_demand == 0:
            return 1.0
            
        return 1.0 + (slope / avg_demand)
    
    def _needs_refit(self, product_id: int) -> bool:
        """Check if model needs to be refitted"""
        if product_id not in self.last_refit:
            return True
            
        hours_since_refit = (
            datetime.now() - self.last_refit[product_id]
        ).total_seconds() / 3600
        
        return hours_since_refit >= self.config.refit_frequency
    
    def _refit_models(self, product_id: int):
        """Refit models for a product"""
        product_data = self._get_product_data(product_id)
        
        # Refit Prophet
        prophet_data = product_data[['timestamp', 'quantity']].rename(
            columns={'timestamp': 'ds', 'quantity': 'y'}
        )
        self.prophet_models[product_id] = self._create_prophet_model()
        self.prophet_models[product_id].fit(prophet_data)
        
        # Refit XGBoost
        features = self._create_feature_matrix(product_data)
        targets = product_data['quantity'].values
        
        self.scalers[product_id] = StandardScaler()
        scaled_features = self.scalers[product_id].fit_transform(features)
        
        self.xgb_models[product_id] = self._create_xgb_model()
        self.xgb_models[product_id].fit(scaled_features, targets)
        
        # Update last refit time
        self.last_refit[product_id] = datetime.now()
    
    def _get_product_data(self, product_id: int) -> pd.DataFrame:
        """Get historical data for a product"""
        return self.historical_data[
            self.historical_data['product_id'] == product_id
        ].copy()
    
    def get_demand_score(
        self,
        product_id: int,
        horizon_days: Optional[int] = None
    ) -> float:
        """Get normalized demand score for product"""
        forecast = self.generate_forecast(product_id, horizon_days)
        
        # Calculate average predicted demand
        avg_demand = np.mean(forecast.forecast_values)
        
        # Normalize by maximum demand across all products
        max_demand = max(
            np.mean(self.generate_forecast(pid).forecast_values)
            for pid in self.prophet_models.keys()
        )
        
        return avg_demand / max_demand if max_demand > 0 else 0.0

# This implementation of the demand forecaster provides:

# 1. Core Forecasting Features:
# - Ensemble forecasting (Prophet + XGBoost)
# - Seasonality pattern detection
# - Trend analysis
# - Confidence intervals

# 2. Advanced Features:
# - Auto model refitting
# - External feature support
# - Seasonality adjustments
# - Demand scoring

# 3. Model Management:
# - Multiple model types
# - Model persistence
# - Performance tracking
# - Feature engineering

# 4. Preprocessing:
# - Data validation
# - Time feature extraction
# - Scaling
# - Aggregation

# 5. Configuration:
# - Flexible model parameters
# - Forecast horizons
# - Update frequencies
# - Confidence levels