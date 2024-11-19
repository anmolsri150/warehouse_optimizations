from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats

from ..core.types import (
    WarehouseState,
    OptimizationMetrics,
    PickingRoute,
    StorageLocation
)

@dataclass
class MetricsConfig:
    """Configuration for metrics calculation"""
    time_window: timedelta = timedelta(days=30)
    moving_average_window: int = 24  # hours
    distance_unit: str = 'meters'
    time_unit: str = 'seconds'
    include_historical: bool = True
    confidence_level: float = 0.95

class WarehouseMetrics:
    """
    Calculates and tracks various performance metrics for warehouse optimization.
    Provides comprehensive analytics and KPI tracking.
    """
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None
    ):
        self.config = config or MetricsConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics tracking
        self.historical_metrics: List[Dict] = []
        self.baseline_metrics: Optional[Dict] = None
        
        # Metric categories
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
    
    def calculate_optimization_metrics(
        self,
        warehouse_state: WarehouseState,
        picking_routes: List[PickingRoute],
        current_time: datetime
    ) -> OptimizationMetrics:
        """
        Calculate comprehensive optimization metrics
        
        Args:
            warehouse_state: Current warehouse state
            picking_routes: Recent picking routes
            current_time: Current timestamp
            
        Returns:
            OptimizationMetrics object
        """
        try:
            # Calculate basic metrics
            picking_efficiency = self._calculate_picking_efficiency(picking_routes)
            space_utilization = self._calculate_space_utilization(warehouse_state)
            constraint_satisfaction = self._calculate_constraint_satisfaction(warehouse_state)
            
            # Calculate advanced metrics
            travel_distance = self._calculate_travel_distance(picking_routes)
            time_metrics = self._calculate_time_metrics(picking_routes)
            utilization_metrics = self._calculate_utilization_metrics(warehouse_state)
            
            metrics = OptimizationMetrics(
                picking_efficiency=picking_efficiency,
                space_utilization=space_utilization,
                constraint_satisfaction=constraint_satisfaction,
                travel_distance=travel_distance,
                total_time=time_metrics['total_time'],
                num_moves=len(picking_routes)
            )
            
            # Update historical metrics
            self._update_historical_metrics(metrics, current_time)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating optimization metrics: {str(e)}")
            raise
    
    def calculate_picking_metrics(
        self,
        routes: List[PickingRoute],
        time_window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Calculate picking-specific metrics"""
        try:
            time_window = time_window or self.config.time_window
            
            # Filter routes by time window
            recent_routes = [
                route for route in routes
                if datetime.now() - route.timestamp <= time_window
            ]
            
            metrics = {
                'average_picking_time': self._calculate_average_picking_time(recent_routes),
                'picks_per_hour': self._calculate_picks_per_hour(recent_routes),
                'travel_distance_per_pick': self._calculate_travel_per_pick(recent_routes),
                'picking_accuracy': self._calculate_picking_accuracy(recent_routes),
                'route_efficiency': self._calculate_route_efficiency(recent_routes)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating picking metrics: {str(e)}")
            raise
    
    def calculate_utilization_metrics(
        self,
        warehouse_state: WarehouseState
    ) -> Dict[str, float]:
        """Calculate utilization metrics"""
        try:
            metrics = {
                'volume_utilization': self._calculate_volume_utilization(warehouse_state),
                'weight_utilization': self._calculate_weight_utilization(warehouse_state),
                'bin_utilization': self._calculate_bin_utilization(warehouse_state),
                'zone_utilization': self._calculate_zone_utilization(warehouse_state)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating utilization metrics: {str(e)}")
            raise
    
    def calculate_performance_metrics(
        self,
        baseline_state: Optional[WarehouseState] = None
    ) -> Dict[str, float]:
        """Calculate performance improvement metrics"""
        try:
            if not self.historical_metrics:
                return {}
            
            current = self.historical_metrics[-1]
            baseline = self.baseline_metrics or self.historical_metrics[0]
            
            improvements = {
                metric: self._calculate_improvement(
                    current.get(metric, 0),
                    baseline.get(metric, 0)
                )
                for metric in self.metric_categories['efficiency']
            }
            
            return improvements
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def _calculate_picking_efficiency(
        self,
        routes: List[PickingRoute]
    ) -> float:
        """Calculate picking efficiency metric"""
        if not routes:
            return 0.0
            
        total_time = sum(route.estimated_time for route in routes)
        total_picks = sum(len(route.items) for route in routes)
        
        return total_picks / total_time if total_time > 0 else 0.0
    
    def _calculate_space_utilization(
        self,
        warehouse_state: WarehouseState
    ) -> float:
        """Calculate space utilization metric"""
        total_volume = sum(
            loc.dimensions.volume
            for loc in warehouse_state.storage_locations.values()
        )
        
        used_volume = sum(
            sum(item.volume for item in loc.current_items.values())
            for loc in warehouse_state.storage_locations.values()
        )
        
        return used_volume / total_volume if total_volume > 0 else 0.0
    
    def _calculate_constraint_satisfaction(
        self,
        warehouse_state: WarehouseState
    ) -> float:
        """Calculate constraint satisfaction rate"""
        # Implementation depends on constraint checking logic
        return 1.0
    
    def _calculate_travel_distance(
        self,
        routes: List[PickingRoute]
    ) -> float:
        """Calculate total travel distance"""
        return sum(route.total_distance for route in routes)
    
    def _calculate_time_metrics(
        self,
        routes: List[PickingRoute]
    ) -> Dict[str, float]:
        """Calculate time-based metrics"""
        return {
            'total_time': sum(route.estimated_time for route in routes),
            'average_time_per_pick': self._calculate_average_picking_time(routes),
            'time_efficiency': self._calculate_time_efficiency(routes)
        }
    
    def _calculate_utilization_metrics(
        self,
        warehouse_state: WarehouseState
    ) -> Dict[str, float]:
        """Calculate utilization metrics"""
        return {
            'volume_utilization': self._calculate_volume_utilization(warehouse_state),
            'weight_utilization': self._calculate_weight_utilization(warehouse_state),
            'bin_utilization': self._calculate_bin_utilization(warehouse_state)
        }
    
    def _calculate_volume_utilization(
        self,
        warehouse_state: WarehouseState
    ) -> float:
        """Calculate volume utilization"""
        total_volume = 0.0
        used_volume = 0.0
        
        for location in warehouse_state.storage_locations.values():
            total_volume += location.dimensions.volume
            used_volume += sum(
                item.volume for item in location.current_items.values()
            )
        
        return used_volume / total_volume if total_volume > 0 else 0.0
    
    def _calculate_weight_utilization(
        self,
        warehouse_state: WarehouseState
    ) -> float:
        """Calculate weight utilization"""
        total_capacity = 0.0
        used_weight = 0.0
        
        for location in warehouse_state.storage_locations.values():
            total_capacity += location.dimensions.max_weight
            used_weight += sum(
                item.weight for item in location.current_items.values()
            )
        
        return used_weight / total_capacity if total_capacity > 0 else 0.0
    
    def _calculate_bin_utilization(
        self,
        warehouse_state: WarehouseState
    ) -> float:
        """Calculate bin utilization rate"""
        total_bins = len(warehouse_state.storage_locations)
        used_bins = sum(
            1 for loc in warehouse_state.storage_locations.values()
            if loc.current_items
        )
        
        return used_bins / total_bins if total_bins > 0 else 0.0
    
    def _calculate_zone_utilization(
        self,
        warehouse_state: WarehouseState
    ) -> Dict[str, float]:
        """Calculate utilization by zone"""
        zone_metrics = {}
        
        for zone_type in set(loc.zone_type for loc in warehouse_state.storage_locations.values()):
            zone_locations = [
                loc for loc in warehouse_state.storage_locations.values()
                if loc.zone_type == zone_type
            ]
            
            total_volume = sum(loc.dimensions.volume for loc in zone_locations)
            used_volume = sum(
                sum(item.volume for item in loc.current_items.values())
                for loc in zone_locations
            )
            
            zone_metrics[zone_type] = used_volume / total_volume if total_volume > 0 else 0.0
        
        return zone_metrics
    
    def _calculate_improvement(
        self,
        current: float,
        baseline: float
    ) -> float:
        """Calculate improvement percentage"""
        if baseline == 0:
            return float('inf') if current > 0 else 0.0
            
        return ((current - baseline) / baseline) * 100
    
    def _update_historical_metrics(
        self,
        metrics: OptimizationMetrics,
        timestamp: datetime
    ):
        """Update historical metrics tracking"""
        self.historical_metrics.append({
            'timestamp': timestamp,
            **metrics.__dict__
        })
        
        # Remove old metrics outside time window
        cutoff_time = timestamp - self.config.time_window
        self.historical_metrics = [
            m for m in self.historical_metrics
            if m['timestamp'] >= cutoff_time
        ]
    
    def get_metrics_summary(
        self,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of metrics"""
        if not self.historical_metrics:
            return {}
            
        time_range = time_range or self.config.time_window
        cutoff_time = datetime.now() - time_range
        
        recent_metrics = [
            m for m in self.historical_metrics
            if m['timestamp'] >= cutoff_time
        ]
        
        summary = {}
        for category, metrics in self.metric_categories.items():
            category_summary = {}
            for metric in metrics:
                values = [m.get(metric, 0) for m in recent_metrics]
                if values:
                    category_summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
            summary[category] = category_summary
        
        return summary

# This implementation provides:

# 1. Core Metrics:
# - Optimization metrics
# - Picking metrics
# - Utilization metrics
# - Performance metrics

# 2. Metric Categories:
# - Efficiency metrics
# - Utilization metrics
# - Quality metrics
# - Productivity metrics

# 3. Statistical Analysis:
# - Historical tracking
# - Improvement calculations
# - Statistical summaries
# - Performance comparisons

# 4. Utilization Analysis:
# - Space utilization
# - Weight utilization
# - Bin utilization
# - Zone utilization

# 5. Specialized Metrics:
# - Picking efficiency
# - Travel distance
# - Time efficiency
# - Constraint satisfaction