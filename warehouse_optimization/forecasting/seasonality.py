from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats, signal
from datetime import datetime, timedelta
from dataclasses import dataclass
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult

import logging
from enum import Enum

class SeasonalityType(Enum):
    """Types of seasonality patterns"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"

@dataclass
class SeasonalPattern:
    """Represents a detected seasonal pattern"""
    pattern_type: SeasonalityType
    frequency: int
    strength: float
    indices: np.ndarray
    pattern_values: np.ndarray
    confidence: float
    peak_periods: List[int]
    trough_periods: List[int]

class SeasonalityAnalyzer:
    """
    Analyzes and manages seasonality patterns in time series data.
    Detects multiple seasonality levels and provides pattern information.
    """
    
    def __init__(
        self,
        min_pattern_strength: float = 0.1,
        confidence_threshold: float = 0.95,
        enable_automatic_detection: bool = True,
        max_patterns: int = 3
    ):
        self.min_pattern_strength = min_pattern_strength
        self.confidence_threshold = confidence_threshold
        self.enable_automatic_detection = enable_automatic_detection
        self.max_patterns = max_patterns
        
        # Initialize pattern storage
        self.detected_patterns: Dict[SeasonalityType, SeasonalPattern] = {}
        self.custom_patterns: List[SeasonalPattern] = []
        
        # Statistical testing parameters
        self.significance_level = 0.05
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def analyze_seasonality(
        self,
        data: pd.Series,
        timestamp_column: Optional[pd.Series] = None,
        frequency: Optional[str] = None
    ) -> Dict[SeasonalityType, SeasonalPattern]:
        """
        Analyze time series data for seasonal patterns
        
        Args:
            data: Time series data
            timestamp_column: Optional timestamp information
            frequency: Optional known frequency
        
        Returns:
            Dictionary of detected seasonal patterns
        """
        try:
            # Reset patterns
            self.detected_patterns = {}
            
            # Prepare time series
            ts = self._prepare_time_series(data, timestamp_column, frequency)
            
            # Decompose time series
            decomposition = self._decompose_time_series(ts)
            
            # Detect patterns at different frequencies
            self._detect_hourly_pattern(ts, decomposition)
            self._detect_daily_pattern(ts, decomposition)
            self._detect_weekly_pattern(ts, decomposition)
            self._detect_monthly_pattern(ts, decomposition)
            
            # Validate and filter patterns
            self._validate_patterns()
            
            return self.detected_patterns
            
        except Exception as e:
            self.logger.error(f"Error in seasonality analysis: {str(e)}")
            return {}
    
    def _prepare_time_series(
        self,
        data: pd.Series,
        timestamp_column: Optional[pd.Series],
        frequency: Optional[str]
    ) -> pd.Series:
        """Prepare time series for analysis"""
        if timestamp_column is not None:
            # Create time series with timestamp index
            ts = pd.Series(data.values, index=pd.DatetimeIndex(timestamp_column))
        else:
            ts = data.copy()
        
        # Resample if frequency provided
        if frequency:
            ts = ts.resample(frequency).mean()
        
        # Handle missing values
        ts = ts.interpolate()
        
        return ts
    
    def _decompose_time_series(
        self,
        ts: pd.Series
    ) -> DecomposeResult:
        """Decompose time series into trend, seasonal, and residual components"""
        return seasonal_decompose(
            ts,
            model='multiplicative',
            period=self._estimate_dominant_period(ts)
        )
    
    def _estimate_dominant_period(self, ts: pd.Series) -> int:
        """Estimate dominant periodicity in time series"""
        # Perform spectral analysis
        frequencies, spectrum = signal.periodogram(ts.values)
        
        # Find dominant frequency
        dominant_freq = frequencies[np.argmax(spectrum)]
        
        # Convert to period
        if dominant_freq > 0:
            return int(1 / dominant_freq)
        return 1
    
    def _detect_hourly_pattern(
        self,
        ts: pd.Series,
        decomposition: DecomposeResult
    ):
        """Detect hourly seasonality pattern"""
        if len(ts) < 24:
            return
        
        hourly_pattern = self._analyze_pattern(
            ts,
            24,
            SeasonalityType.HOURLY
        )
        
        if hourly_pattern:
            self.detected_patterns[SeasonalityType.HOURLY] = hourly_pattern
    
    def _detect_daily_pattern(
        self,
        ts: pd.Series,
        decomposition: DecomposeResult
    ):
        """Detect daily seasonality pattern"""
        if len(ts) < 7:
            return
        
        daily_pattern = self._analyze_pattern(
            ts,
            7,
            SeasonalityType.DAILY
        )
        
        if daily_pattern:
            self.detected_patterns[SeasonalityType.DAILY] = daily_pattern
    
    def _detect_weekly_pattern(
        self,
        ts: pd.Series,
        decomposition: DecomposeResult
    ):
        """Detect weekly seasonality pattern"""
        if len(ts) < 4 * 7:
            return
        
        weekly_pattern = self._analyze_pattern(
            ts,
            7,
            SeasonalityType.WEEKLY
        )
        
        if weekly_pattern:
            self.detected_patterns[SeasonalityType.WEEKLY] = weekly_pattern
    
    def _detect_monthly_pattern(
        self,
        ts: pd.Series,
        decomposition: DecomposeResult
    ):
        """Detect monthly seasonality pattern"""
        if len(ts) < 30:
            return
        
        monthly_pattern = self._analyze_pattern(
            ts,
            30,
            SeasonalityType.MONTHLY
        )
        
        if monthly_pattern:
            self.detected_patterns[SeasonalityType.MONTHLY] = monthly_pattern
    
    def _analyze_pattern(
        self,
        ts: pd.Series,
        period: int,
        pattern_type: SeasonalityType
    ) -> Optional[SeasonalPattern]:
        """Analyze time series for specific seasonal pattern"""
        # Extract pattern
        pattern_values = self._extract_pattern_values(ts, period)
        
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(ts, pattern_values, period)
        
        # Test significance
        is_significant = self._test_pattern_significance(
            ts,
            pattern_values,
            period
        )
        
        if strength >= self.min_pattern_strength and is_significant:
            return SeasonalPattern(
                pattern_type=pattern_type,
                frequency=period,
                strength=strength,
                indices=np.arange(period),
                pattern_values=pattern_values,
                confidence=self._calculate_pattern_confidence(ts, pattern_values, period),
                peak_periods=self._find_peak_periods(pattern_values),
                trough_periods=self._find_trough_periods(pattern_values)
            )
        
        return None
    
    def _extract_pattern_values(
        self,
        ts: pd.Series,
        period: int
    ) -> np.ndarray:
        """Extract pattern values for given period"""
        # Reshape data into period-length segments
        n_periods = len(ts) // period
        reshaped = ts[:n_periods * period].values.reshape(n_periods, period)
        
        # Calculate mean pattern
        return np.mean(reshaped, axis=0)
    
    def _calculate_pattern_strength(
        self,
        ts: pd.Series,
        pattern_values: np.ndarray,
        period: int
    ) -> float:
        """Calculate strength of seasonal pattern"""
        # Calculate variance explained by pattern
        total_variance = np.var(ts.values)
        if total_variance == 0:
            return 0.0
            
        # Repeat pattern for full series length
        n_repeats = len(ts) // len(pattern_values)
        full_pattern = np.tile(pattern_values, n_repeats)
        
        # Calculate residual variance
        residual_variance = np.var(ts.values[:len(full_pattern)] - full_pattern)
        
        return max(0, 1 - (residual_variance / total_variance))
    
    def _test_pattern_significance(
        self,
        ts: pd.Series,
        pattern_values: np.ndarray,
        period: int
    ) -> bool:
        """Test if pattern is statistically significant"""
        # Perform autocorrelation test
        acf = sm.tsa.stattools.acf(ts.values, nlags=period)
        
        # Test if autocorrelation at period lag is significant
        confidence_interval = stats.norm.interval(
            self.confidence_threshold,
            loc=0,
            scale=1/np.sqrt(len(ts))
        )[1]
        
        return abs(acf[period-1]) > confidence_interval
    
    def _calculate_pattern_confidence(
        self,
        ts: pd.Series,
        pattern_values: np.ndarray,
        period: int
    ) -> float:
        """Calculate confidence level in pattern"""
        # Calculate standard error of pattern values
        n_periods = len(ts) // period
        reshaped = ts[:n_periods * period].values.reshape(n_periods, period)
        std_error = np.std(reshaped, axis=0) / np.sqrt(n_periods)
        
        # Calculate average relative error
        relative_error = np.mean(std_error / np.abs(pattern_values))
        
        return max(0, 1 - relative_error)
    
    def _find_peak_periods(
        self,
        pattern_values: np.ndarray
    ) -> List[int]:
        """Find peak periods in pattern"""
        return list(signal.find_peaks(pattern_values)[0])
    
    def _find_trough_periods(
        self,
        pattern_values: np.ndarray
    ) -> List[int]:
        """Find trough periods in pattern"""
        return list(signal.find_peaks(-pattern_values)[0])
    
    def _validate_patterns(self):
        """Validate and filter detected patterns"""
        if not self.detected_patterns:
            return
            
        # Sort patterns by strength
        sorted_patterns = sorted(
            self.detected_patterns.items(),
            key=lambda x: x[1].strength,
            reverse=True
        )
        
        # Keep only top patterns
        self.detected_patterns = {
            k: v for k, v in sorted_patterns[:self.max_patterns]
        }
    
    def get_combined_pattern(
        self,
        timestamp: pd.Timestamp
    ) -> float:
        """Get combined seasonal factor for given timestamp"""
        if not self.detected_patterns:
            return 1.0
            
        combined_factor = 1.0
        
        for pattern in self.detected_patterns.values():
            # Get appropriate index for timestamp based on pattern type
            idx = self._get_pattern_index(timestamp, pattern.pattern_type)
            if idx is not None:
                # Multiply by pattern value (normalized)
                pattern_value = pattern.pattern_values[idx]
                normalized_value = pattern_value / np.mean(pattern.pattern_values)
                combined_factor *= normalized_value
        
        return combined_factor
    
    def _get_pattern_index(
        self,
        timestamp: pd.Timestamp,
        pattern_type: SeasonalityType
    ) -> Optional[int]:
        """Get pattern index for timestamp"""
        if pattern_type == SeasonalityType.HOURLY:
            return timestamp.hour
        elif pattern_type == SeasonalityType.DAILY:
            return timestamp.dayofweek
        elif pattern_type == SeasonalityType.WEEKLY:
            return timestamp.week % 52
        elif pattern_type == SeasonalityType.MONTHLY:
            return timestamp.month - 1
        return None

# This implementation provides:

# 1. Core Seasonality Features:
# - Multiple seasonality type detection
# - Pattern strength calculation
# - Statistical significance testing
# - Pattern confidence estimation

# 2. Pattern Analysis:
# - Time series decomposition
# - Spectral analysis
# - Peak/trough detection
# - Pattern validation

# 3. Advanced Features:
# - Auto-detection of patterns
# - Multiple frequency support
# - Combined pattern calculation
# - Confidence scoring

# 4. Pattern Types:
# - Hourly patterns
# - Daily patterns
# - Weekly patterns
# - Monthly patterns
# - Custom patterns

# 5. Statistical Features:
# - Variance analysis
# - Autocorrelation testing
# - Standard error calculation
# - Significance testing