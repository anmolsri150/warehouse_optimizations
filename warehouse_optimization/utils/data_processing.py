from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
from dataclasses import dataclass

@dataclass
class DataPreprocessConfig:
    """Configuration for data preprocessing"""
    remove_outliers: bool = True
    outlier_threshold: float = 3.0  # standard deviations
    fill_missing: bool = True
    min_data_points: int = 30
    time_aggregation: str = '1D'
    normalize_features: bool = True
    encode_categorical: bool = True

class DataProcessor:
    """
    Handles data preprocessing, transformations, and utility functions
    for warehouse optimization data.
    """
    
    def __init__(
        self,
        config: Optional[DataPreprocessConfig] = None
    ):
        self.config = config or DataPreprocessConfig()
        self.logger = logging.getLogger(__name__)
        
        # Store preprocessing states
        self.categorical_mappings: Dict[str, Dict] = {}
        self.scaling_factors: Dict[str, Dict[str, float]] = {}
        self.feature_statistics: Dict[str, Dict[str, float]] = {}
    
    def preprocess_order_data(
        self,
        order_data: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Preprocess order data
        
        Args:
            order_data: Raw order data
            required_columns: Optional list of required columns
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            df = order_data.copy()
            
            # Validate required columns
            required = required_columns or [
                'order_id', 'timestamp', 'product_id', 'quantity'
            ]
            self._validate_columns(df, required)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Remove outliers if configured
            if self.config.remove_outliers:
                df = self._remove_outliers(df, ['quantity'])
            
            # Fill missing values if configured
            if self.config.fill_missing:
                df = self._fill_missing_values(df)
            
            # Add derived features
            df = self._add_time_features(df)
            df = self._add_order_features(df)
            
            # Normalize numerical features if configured
            if self.config.normalize_features:
                numerical_columns = df.select_dtypes(include=[np.number]).columns
                df = self._normalize_features(df, numerical_columns)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing order data: {str(e)}")
            raise
    
    def preprocess_inventory_data(
        self,
        inventory_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocess inventory data"""
        try:
            df = inventory_data.copy()
            
            # Validate required columns
            required = ['product_id', 'bin_id', 'quantity', 'date_added']
            self._validate_columns(df, required)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['date_added'])
            
            # Add inventory features
            df = self._add_inventory_features(df)
            
            # Normalize if configured
            if self.config.normalize_features:
                numerical_columns = df.select_dtypes(include=[np.number]).columns
                df = self._normalize_features(df, numerical_columns)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing inventory data: {str(e)}")
            raise
    
    def preprocess_product_data(
        self,
        product_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocess product data"""
        try:
            df = product_data.copy()
            
            # Validate required columns
            required = ['product_id', 'category', 'length', 'width', 'height', 'weight']
            self._validate_columns(df, required)
            
            # Calculate derived features
            df['volume'] = df['length'] * df['width'] * df['height']
            df['density'] = df['weight'] / df['volume']
            
            # Encode categorical features if configured
            if self.config.encode_categorical:
                categorical_columns = ['category']
                df = self._encode_categorical_features(df, categorical_columns)
            
            # Normalize numerical features
            if self.config.normalize_features:
                numerical_columns = ['length', 'width', 'height', 'weight', 'volume', 'density']
                df = self._normalize_features(df, numerical_columns)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing product data: {str(e)}")
            raise
    
    def preprocess_location_data(
        self,
        location_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocess storage location data"""
        try:
            df = location_data.copy()
            
            # Validate required columns
            required = ['bin_id', 'zone_type', 'x_coord', 'y_coord', 'z_coord']
            self._validate_columns(df, required)
            
            # Calculate distances
            df['distance_to_pickup'] = self._calculate_distances(
                df[['x_coord', 'y_coord', 'z_coord']]
            )
            
            # Encode categorical features
            if self.config.encode_categorical:
                categorical_columns = ['zone_type']
                df = self._encode_categorical_features(df, categorical_columns)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing location data: {str(e)}")
            raise
    
    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str]):
        """Validate required columns exist in DataFrame"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Remove outliers from specified columns"""
        for column in columns:
            z_scores = stats.zscore(df[column])
            df = df[abs(z_scores) < self.config.outlier_threshold]
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using appropriate strategies"""
        # Numeric columns: fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns: fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def _add_order_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add order-specific features"""
        # Calculate order size
        order_sizes = df.groupby('order_id')['quantity'].sum()
        df['order_size'] = df['order_id'].map(order_sizes)
        
        # Calculate product frequency
        product_freq = df.groupby('product_id')['order_id'].count()
        df['product_frequency'] = df['product_id'].map(product_freq)
        
        return df
    
    def _add_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add inventory-specific features"""
        # Calculate turnover rate
        df['turnover_rate'] = df.groupby('product_id')['quantity'].transform(
            lambda x: x.diff() / x.shift(1)
        )
        
        # Calculate days of supply
        daily_demand = df.groupby('product_id')['quantity'].transform(
            lambda x: x.mean()
        )
        df['days_of_supply'] = df['quantity'] / daily_demand
        
        return df
    
    def _normalize_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Normalize numerical features"""
        for column in columns:
            if column not in self.scaling_factors:
                mean = df[column].mean()
                std = df[column].std()
                self.scaling_factors[column] = {'mean': mean, 'std': std}
            
            mean = self.scaling_factors[column]['mean']
            std = self.scaling_factors[column]['std']
            
            if std > 0:
                df[column] = (df[column] - mean) / std
            
        return df
    
    def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Encode categorical features"""
        for column in columns:
            if column not in self.categorical_mappings:
                unique_values = df[column].unique()
                self.categorical_mappings[column] = {
                    value: i for i, value in enumerate(unique_values)
                }
            
            df[column] = df[column].map(self.categorical_mappings[column])
        
        return df
    
    def _calculate_distances(self, coordinates: pd.DataFrame) -> pd.Series:
        """Calculate Euclidean distances from pickup point"""
        pickup_point = np.array([0, 0, 0])  # Assuming pickup point is at origin
        return np.sqrt(
            ((coordinates - pickup_point) ** 2).sum(axis=1)
        )
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate and store feature statistics"""
        stats_dict = {}
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            stats_dict[column] = {
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'median': df[column].median()
            }
        
        return stats_dict
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Inverse transform normalized features"""
        result = df.copy()
        columns = columns or self.scaling_factors.keys()
        
        for column in columns:
            if column in self.scaling_factors:
                mean = self.scaling_factors[column]['mean']
                std = self.scaling_factors[column]['std']
                result[column] = (result[column] * std) + mean
        
        return result
    
    def decode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Decode categorical features"""
        result = df.copy()
        columns = columns or self.categorical_mappings.keys()
        
        for column in columns:
            if column in self.categorical_mappings:
                reverse_mapping = {
                    v: k for k, v in self.categorical_mappings[column].items()
                }
                result[column] = result[column].map(reverse_mapping)
        
        return result

# This implementation provides:

# 1. Core Data Processing:
# - Order data preprocessing
# - Inventory data preprocessing
# - Product data preprocessing
# - Location data preprocessing

# 2. Feature Engineering:
# - Time-based features
# - Order features
# - Inventory features
# - Distance calculations

# 3. Data Cleaning:
# - Outlier removal
# - Missing value handling
# - Data validation
# - Format conversion

# 4. Feature Transformation:
# - Normalization
# - Categorical encoding
# - Inverse transformations
# - Feature scaling

# 5. Statistics and Analysis:
# - Feature statistics
# - Scaling factors
# - Categorical mappings
# - Data validation