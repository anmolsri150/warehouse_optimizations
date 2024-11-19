from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet
from lightgbm import LGBMRegressor
import logging

@dataclass
class ModelWeight:
    """Stores model weights and performance metrics"""
    weight: float
    mape: float
    rmse: float
    last_updated: datetime

class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    Advanced ensemble model combining multiple forecasting algorithms
    with dynamic weight adjustment and performance tracking.
    """
    
    def __init__(
        self,
        models: Optional[Dict[str, BaseEstimator]] = None,
        weights: Optional[Dict[str, float]] = None,
        performance_window: int = 30,
        min_weight: float = 0.1,
        enable_dynamic_weights: bool = True,
        validation_split: float = 0.2
    ):
        self.models = models or self._get_default_models()
        self.performance_window = performance_window
        self.min_weight = min_weight
        self.enable_dynamic_weights = enable_dynamic_weights
        self.validation_split = validation_split
        
        # Initialize weights and performance tracking
        self.model_weights: Dict[str, ModelWeight] = {}
        self._initialize_weights(weights)
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict]] = {
            model_name: [] for model_name in self.models.keys()
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Feature importance tracking
        self.feature_importance: Dict[str, Dict[str, float]] = {}
    
    def _get_default_models(self) -> Dict[str, BaseEstimator]:
        """Initialize default set of models"""
        return {
            'prophet': Prophet(
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            ),
            'xgboost': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            ),
            'lightgbm': LGBMRegressor(
                objective='regression',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
        }
    
    def _initialize_weights(self, weights: Optional[Dict[str, float]] = None):
        """Initialize model weights"""
        if weights:
            # Validate and normalize provided weights
            total = sum(weights.values())
            normalized_weights = {
                k: max(v / total, self.min_weight)
                for k, v in weights.items()
            }
        else:
            # Equal weights initially
            n_models = len(self.models)
            normalized_weights = {
                name: 1.0 / n_models
                for name in self.models.keys()
            }
        
        # Initialize ModelWeight objects
        for model_name, weight in normalized_weights.items():
            self.model_weights[model_name] = ModelWeight(
                weight=weight,
                mape=float('inf'),
                rmse=float('inf'),
                last_updated=datetime.now()
            )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """Fit all models in ensemble"""
        # Split data for validation
        val_size = int(len(X) * self.validation_split)
        train_X, val_X = X[:-val_size], X[-val_size:]
        train_y, val_y = y[:-val_size], y[-val_size:]
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Fitting model: {model_name}")
                
                if model_name == 'prophet':
                    # Special handling for Prophet
                    prophet_df = self._prepare_prophet_data(train_X, train_y)
                    model.fit(prophet_df)
                else:
                    # Standard sklearn-like API
                    model.fit(train_X, train_y)
                
                # Validate and update weights
                if self.enable_dynamic_weights:
                    self._update_model_weights(
                        model_name,
                        model,
                        val_X,
                        val_y
                    )
                
                # Track feature importance if available
                if hasattr(model, 'feature_importances_') and feature_names:
                    self._update_feature_importance(
                        model_name,
                        model,
                        feature_names
                    )
                    
            except Exception as e:
                self.logger.error(f"Error fitting {model_name}: {str(e)}")
                continue
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Generate ensemble predictions"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'prophet':
                    prophet_X = self._prepare_prophet_data(X)
                    pred = model.predict(prophet_X)['yhat'].values
                else:
                    pred = model.predict(X)
                
                predictions[model_name] = pred
                
            except Exception as e:
                self.logger.error(f"Error predicting with {model_name}: {str(e)}")
                predictions[model_name] = np.zeros(len(X))
        
        # Combine predictions using weights
        weighted_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.model_weights[model_name].weight
            weighted_pred += weight * pred
        
        if return_components:
            return weighted_pred, predictions
        return weighted_pred
    
    def _update_model_weights(
        self,
        model_name: str,
        model: BaseEstimator,
        val_X: pd.DataFrame,
        val_y: np.ndarray
    ):
        """Update model weights based on validation performance"""
        try:
            # Get predictions
            if model_name == 'prophet':
                prophet_X = self._prepare_prophet_data(val_X)
                pred = model.predict(prophet_X)['yhat'].values
            else:
                pred = model.predict(val_X)
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(val_y, pred)
            rmse = np.sqrt(mean_squared_error(val_y, pred))
            
            # Update performance history
            self.performance_history[model_name].append({
                'timestamp': datetime.now(),
                'mape': mape,
                'rmse': rmse
            })
            
            # Update model weight
            self.model_weights[model_name] = ModelWeight(
                weight=self._calculate_new_weight(model_name, mape, rmse),
                mape=mape,
                rmse=rmse,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating weights for {model_name}: {str(e)}")
    
    def _calculate_new_weight(
        self,
        model_name: str,
        mape: float,
        rmse: float
    ) -> float:
        """Calculate new weight based on performance metrics"""
        # Get current performance for all models
        performances = {
            name: weight.mape
            for name, weight in self.model_weights.items()
        }
        
        # Update current model's performance
        performances[model_name] = mape
        
        # Calculate inverse error (higher for better performing models)
        inverse_errors = {
            name: 1 / (error + 1e-10)
            for name, error in performances.items()
        }
        
        # Normalize to get weights
        total = sum(inverse_errors.values())
        weights = {
            name: max(err / total, self.min_weight)
            for name, err in inverse_errors.items()
        }
        
        # Renormalize after applying minimum weight
        total = sum(weights.values())
        return weights[model_name] / total
    
    def _prepare_prophet_data(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        df = pd.DataFrame({'ds': X.index})
        if y is not None:
            df['y'] = y
        return df
    
    def _update_feature_importance(
        self,
        model_name: str,
        model: BaseEstimator,
        feature_names: List[str]
    ):
        """Update feature importance tracking"""
        if not hasattr(model, 'feature_importances_'):
            return
            
        importances = model.feature_importances_
        self.feature_importance[model_name] = {
            feature: importance
            for feature, importance in zip(feature_names, importances)
        }
    
    def get_model_diagnostics(self) -> Dict[str, Dict]:
        """Get diagnostic information for all models"""
        diagnostics = {}
        
        for model_name in self.models.keys():
            model_diag = {
                'weight': self.model_weights[model_name].weight,
                'mape': self.model_weights[model_name].mape,
                'rmse': self.model_weights[model_name].rmse,
                'last_updated': self.model_weights[model_name].last_updated,
                'performance_history': self.performance_history[model_name]
            }
            
            if model_name in self.feature_importance:
                model_diag['feature_importance'] = self.feature_importance[model_name]
                
            diagnostics[model_name] = model_diag
        
        return diagnostics
    
    def get_best_model(self) -> Tuple[str, BaseEstimator]:
        """Get the best performing model"""
        best_model = min(
            self.model_weights.items(),
            key=lambda x: x[1].mape
        )
        return best_model[0], self.models[best_model[0]]
    
    def add_model(
        self,
        name: str,
        model: BaseEstimator,
        weight: Optional[float] = None
    ):
        """Add a new model to the ensemble"""
        if name in self.models:
            raise ValueError(f"Model {name} already exists in ensemble")
            
        self.models[name] = model
        
        # Initialize weight
        if weight is None:
            weight = self.min_weight
            # Redistribute remaining weight
            remaining = 1.0 - weight
            for existing_weight in self.model_weights.values():
                existing_weight.weight *= remaining
        
        self.model_weights[name] = ModelWeight(
            weight=weight,
            mape=float('inf'),
            rmse=float('inf'),
            last_updated=datetime.now()
        )
        
        self.performance_history[name] = []
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found in ensemble")
            
        # Remove model
        self.models.pop(name)
        
        # Redistribute weight
        removed_weight = self.model_weights[name].weight
        self.model_weights.pop(name)
        
        # Redistribute weight to remaining models
        if self.model_weights:
            weight_per_model = removed_weight / len(self.model_weights)
            for weight in self.model_weights.values():
                weight.weight += weight_per_model
        
        # Remove from performance history
        self.performance_history.pop(name)

# This implementation provides:

# 1. Core Ensemble Features:
# - Multiple model support
# - Dynamic weight adjustment
# - Performance tracking
# - Model diagnostics

# 2. Advanced Features:
# - Feature importance tracking
# - Model addition/removal
# - Performance history
# - Validation splitting

# 3. Model Management:
# - Individual model tracking
# - Weight optimization
# - Error handling
# - Diagnostics

# 4. Flexibility:
# - Support for different model types
# - Custom weight initialization
# - Performance window configuration
# - Minimum weight thresholds

# 5. Monitoring:
# - Performance metrics
# - Feature importance
# - Weight evolution
# - Model diagnostics