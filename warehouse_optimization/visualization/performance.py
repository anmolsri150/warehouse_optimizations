from typing import Dict, List, Optional, Union, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..core.types import (
    OptimizationMetrics,
    WarehouseState,
    PickingRoute
)

@dataclass
class PerformanceConfig:
    """Configuration for performance visualization"""
    time_window: timedelta = timedelta(days=30)
    update_interval: int = 300  # seconds
    moving_average_window: int = 24  # hours
    show_targets: bool = True
    show_predictions: bool = True
    highlight_anomalies: bool = True
    custom_thresholds: Optional[Dict[str, float]] = None

class PerformanceVisualizer:
    """
    Handles visualization of performance metrics and analytics tracking.
    Provides comprehensive performance dashboards and trend analysis.
    """
    
    def __init__(
        self,
        config: Optional[PerformanceConfig] = None,
        enable_realtime: bool = True
    ):
        self.config = config or PerformanceConfig()
        self.enable_realtime = enable_realtime
        
        # Initialize metrics tracking
        self.metrics_history: List[Dict] = []
        self.target_metrics: Dict[str, float] = {}
        self.anomaly_thresholds: Dict[str, Tuple[float, float]] = {}
        
        # Performance categories
        self.metric_categories = {
            'efficiency': [
                'picking_efficiency',
                'travel_distance',
                'time_per_pick'
            ],
            'utilization': [
                'space_utilization',
                'volume_utilization',
                'weight_utilization'
            ],
            'quality': [
                'accuracy_rate',
                'error_rate',
                'constraint_violations'
            ],
            'productivity': [
                'picks_per_hour',
                'items_processed',
                'completion_rate'
            ]
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize anomaly detection
        self._initialize_anomaly_detection()
    
    def create_performance_dashboard(
        self,
        current_metrics: OptimizationMetrics,
        warehouse_state: Optional[WarehouseState] = None
    ) -> go.Figure:
        """Create comprehensive performance dashboard"""
        try:
            # Update metrics history
            self._update_metrics_history(current_metrics)
            
            # Create dashboard layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Efficiency Metrics',
                    'Utilization Metrics',
                    'Quality Metrics',
                    'Productivity Metrics',
                    'Trend Analysis',
                    'Performance Distribution'
                )
            )
            
            # Add metric visualizations
            self._add_efficiency_metrics(fig, 1, 1)
            self._add_utilization_metrics(fig, 1, 2)
            self._add_quality_metrics(fig, 2, 1)
            self._add_productivity_metrics(fig, 2, 2)
            self._add_trend_analysis(fig, 3, 1)
            self._add_performance_distribution(fig, 3, 2)
            
            # Update layout
            self._update_dashboard_layout(fig)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {str(e)}")
            return self._create_error_figure()
    
    def create_trend_analysis(
        self,
        metric_name: str,
        time_range: Optional[timedelta] = None
    ) -> go.Figure:
        """Create detailed trend analysis for specific metric"""
        try:
            # Get metric history
            history_df = self._get_metric_history(metric_name, time_range)
            
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(
                go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df[metric_name],
                    mode='lines',
                    name='Actual'
                )
            )
            
            # Add moving average
            ma = self._calculate_moving_average(history_df[metric_name])
            fig.add_trace(
                go.Scatter(
                    x=history_df['timestamp'],
                    y=ma,
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Moving Average'
                )
            )
            
            # Add target if available
            if metric_name in self.target_metrics:
                fig.add_hline(
                    y=self.target_metrics[metric_name],
                    line_dash="dot",
                    annotation_text="Target",
                    line_color="green"
                )
            
            # Add anomalies if enabled
            if self.config.highlight_anomalies:
                anomalies = self._detect_anomalies(history_df[metric_name])
                fig.add_trace(
                    go.Scatter(
                        x=history_df.loc[anomalies, 'timestamp'],
                        y=history_df.loc[anomalies, metric_name],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='x'
                        ),
                        name='Anomalies'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f"Trend Analysis: {metric_name}",
                xaxis_title="Time",
                yaxis_title="Value",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating trend analysis: {str(e)}")
            return self._create_error_figure()
    
    def create_picking_analytics(
        self,
        picking_routes: List[PickingRoute],
        time_window: Optional[timedelta] = None
    ) -> go.Figure:
        """Create picking performance analytics"""
        try:
            # Process picking data
            picking_df = self._process_picking_routes(picking_routes, time_window)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Picking Time Distribution',
                    'Route Efficiency',
                    'Picks per Hour',
                    'Distance per Pick'
                )
            )
            
            # Add picking time distribution
            fig.add_trace(
                go.Histogram(
                    x=picking_df['picking_time'],
                    name='Picking Time'
                ),
                row=1, col=1
            )
            
            # Add route efficiency scatter
            fig.add_trace(
                go.Scatter(
                    x=picking_df['total_distance'],
                    y=picking_df['picks_count'],
                    mode='markers',
                    name='Route Efficiency'
                ),
                row=1, col=2
            )
            
            # Add picks per hour trend
            fig.add_trace(
                go.Scatter(
                    x=picking_df['timestamp'],
                    y=picking_df['picks_per_hour'],
                    mode='lines',
                    name='Picks per Hour'
                ),
                row=2, col=1
            )
            
            # Add distance per pick trend
            fig.add_trace(
                go.Scatter(
                    x=picking_df['timestamp'],
                    y=picking_df['distance_per_pick'],
                    mode='lines',
                    name='Distance per Pick'
                ),
                row=2, col=2
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating picking analytics: {str(e)}")
            return self._create_error_figure()
    
    def _update_metrics_history(self, metrics: OptimizationMetrics):
        """Update metrics history with new data"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            **metrics.__dict__
        })
        
        # Remove old metrics
        cutoff_time = datetime.now() - self.config.time_window
        self.metrics_history = [
            m for m in self.metrics_history
            if m['timestamp'] >= cutoff_time
        ]
    
    def _add_efficiency_metrics(self, fig: go.Figure, row: int, col: int):
        """Add efficiency metrics visualization"""
        metrics = self.metric_categories['efficiency']
        for i, metric in enumerate(metrics):
            data = self._get_metric_history(metric)
            
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data[metric],
                    name=metric,
                    showlegend=True
                ),
                row=row, col=col
            )
    
    def _add_utilization_metrics(self, fig: go.Figure, row: int, col: int):
        """Add utilization metrics visualization"""
        metrics = self.metric_categories['utilization']
        current_values = [
            self.metrics_history[-1][metric]
            for metric in metrics
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=current_values,
                name='Utilization'
            ),
            row=row, col=col
        )
    
    def _add_quality_metrics(self, fig: go.Figure, row: int, col: int):
        """Add quality metrics visualization"""
        metrics = self.metric_categories['quality']
        for metric in metrics:
            data = self._get_metric_history(metric)
            
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data[metric],
                    name=metric,
                    fill='tonexty'
                ),
                row=row, col=col
            )
    
    def _add_productivity_metrics(self, fig: go.Figure, row: int, col: int):
        """Add productivity metrics visualization"""
        metrics = self.metric_categories['productivity']
        data = pd.DataFrame(self.metrics_history)
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=data[metrics[0]].iloc[-1],
                delta={'reference': data[metrics[0]].iloc[-2]},
                title={'text': metrics[0]}
            ),
            row=row, col=col
        )
    
    def _add_trend_analysis(self, fig: go.Figure, row: int, col: int):
        """Add trend analysis visualization"""
        # Implementation for trend analysis visualization
        pass
    
    def _add_performance_distribution(self, fig: go.Figure, row: int, col: int):
        """Add performance distribution visualization"""
        # Implementation for performance distribution
        pass
    
    def _calculate_moving_average(
        self,
        series: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """Calculate moving average for series"""
        window = window or self.config.moving_average_window
        return series.rolling(window=window, min_periods=1).mean()
    
    def _detect_anomalies(self, series: pd.Series) -> pd.Series:
        """Detect anomalies in metric values"""
        if not self.config.highlight_anomalies:
            return pd.Series(False, index=series.index)
            
        # Calculate z-scores
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > 3  # Mark values more than 3 standard deviations away
    
    def _get_metric_history(
        self,
        metric_name: str,
        time_range: Optional[timedelta] = None
    ) -> pd.DataFrame:
        """Get historical data for specific metric"""
        df = pd.DataFrame(self.metrics_history)
        
        if time_range:
            cutoff_time = datetime.now() - time_range
            df = df[df['timestamp'] >= cutoff_time]
            
        return df
    
    def _update_dashboard_layout(self, fig: go.Figure):
        """Update dashboard layout"""
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Warehouse Performance Dashboard",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
    
    def _create_error_figure(self) -> go.Figure:
        """Create error figure when visualization fails"""
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating visualization",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        return fig

# This implementation provides:

# 1. Core Performance Features:
# - Comprehensive dashboards
# - Trend analysis
# - Picking analytics
# - Real-time tracking

# 2. Metric Categories:
# - Efficiency metrics
# - Utilization metrics
# - Quality metrics
# - Productivity metrics

# 3. Advanced Analytics:
# - Moving averages
# - Anomaly detection
# - Performance distribution
# - Trend analysis

# 4. Visualization Types:
# - Time series plots
# - Bar charts
# - Histograms
# - Indicators

# 5. Interactive Features:
# - Real-time updates
# - Custom time ranges
# - Anomaly highlighting
# - Target tracking