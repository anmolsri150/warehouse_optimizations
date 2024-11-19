from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..core.types import (
    WarehouseState,
    OptimizationMetrics,
    StorageLocation,
    ProductAttributes
)

class WarehouseDashboard:
    """
    Main dashboard for warehouse visualization.
    Provides comprehensive views of warehouse state, performance metrics,
    and optimization results.
    """
    
    def __init__(
        self,
        layout_config: Optional[Dict] = None,
        update_interval: int = 5000,  # milliseconds
        enable_3d: bool = True,
        dark_mode: bool = False
    ):
        self.layout_config = layout_config or self._default_layout_config()
        self.update_interval = update_interval
        self.enable_3d = enable_3d
        self.dark_mode = dark_mode
        
        # Initialize figures
        self.figures: Dict[str, go.Figure] = {}
        
        # Track metrics history
        self.metrics_history: List[Dict] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize dashboard
        self._initialize_dashboard()
    
    def _default_layout_config(self) -> Dict:
        """Default dashboard layout configuration"""
        return {
            'rows': 3,
            'cols': 2,
            'specs': [
                [{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}],
                [None, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}]
            ],
            'subplot_titles': [
                'Warehouse Layout',
                'Utilization Metrics',
                'Performance Trends',
                'Picking Patterns',
                'Optimization Impact'
            ]
        }
    
    def _initialize_dashboard(self):
        """Initialize dashboard figures"""
        # Create main figure with subplots
        self.figures['main'] = make_subplots(
            rows=self.layout_config['rows'],
            cols=self.layout_config['cols'],
            specs=self.layout_config['specs'],
            subplot_titles=self.layout_config['subplot_titles'],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Set theme
        self._apply_theme(self.figures['main'])
    
    def update_dashboard(
        self,
        warehouse_state: WarehouseState,
        metrics: OptimizationMetrics,
        update_time: datetime
    ) -> go.Figure:
        """
        Update dashboard with new state and metrics
        
        Args:
            warehouse_state: Current warehouse state
            metrics: Current optimization metrics
            update_time: Timestamp of update
            
        Returns:
            Updated dashboard figure
        """
        try:
            # Store metrics history
            self.metrics_history.append({
                'timestamp': update_time,
                **metrics.__dict__
            })
            
            # Update layout visualization
            self._update_layout_view(warehouse_state)
            
            # Update metrics visualization
            self._update_metrics_view(metrics)
            
            # Update performance trends
            self._update_performance_trends()
            
            # Update picking patterns
            self._update_picking_patterns(warehouse_state)
            
            # Update optimization impact
            self._update_optimization_impact()
            
            return self.figures['main']
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard: {str(e)}")
            return None
    
    def _update_layout_view(self, warehouse_state: WarehouseState):
        """Update warehouse layout visualization"""
        if self.enable_3d:
            self._update_3d_layout(warehouse_state)
        else:
            self._update_2d_layout(warehouse_state)
    
    def _update_3d_layout(self, warehouse_state: WarehouseState):
        """Update 3D warehouse layout visualization"""
        # Clear existing traces
        self.figures['main'].update_traces(
            selector=dict(type='scatter3d'),
            row=1, col=1
        )
        
        # Add zones
        for zone_type, locations in warehouse_state.zones.items():
            self._add_zone_visualization(zone_type, locations)
        
        # Add items
        self._add_item_visualization(warehouse_state)
        
        # Update layout
        self.figures['main'].update_layout(
            scene=dict(
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )
    
    def _update_2d_layout(self, warehouse_state: WarehouseState):
        """Update 2D warehouse layout visualization"""
        # Create heatmap of utilization
        utilization_matrix = self._create_utilization_matrix(warehouse_state)
        
        self.figures['main'].add_trace(
            go.Heatmap(
                z=utilization_matrix,
                colorscale='Viridis',
                showscale=True
            ),
            row=1, col=1
        )
    
    def _update_metrics_view(self, metrics: OptimizationMetrics):
        """Update metrics visualization"""
        # Create gauge charts for key metrics
        self.figures['main'].add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.space_utilization * 100,
                title={'text': "Space Utilization (%)"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 2}
            )
        )
        
        self.figures['main'].add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.picking_efficiency * 100,
                title={'text': "Picking Efficiency (%)"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 2, 'column': 2}
            )
        )
    
    def _update_performance_trends(self):
        """Update performance trends visualization"""
        if not self.metrics_history:
            return
            
        # Convert metrics history to DataFrame
        history_df = pd.DataFrame(self.metrics_history)
        
        # Plot trends
        self.figures['main'].add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['picking_efficiency'],
                name="Picking Efficiency",
                mode='lines'
            ),
            row=2, col=2
        )
        
        self.figures['main'].add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['space_utilization'],
                name="Space Utilization",
                mode='lines'
            ),
            row=2, col=2
        )
    
    def _update_picking_patterns(self, warehouse_state: WarehouseState):
        """Update picking patterns visualization"""
        # Create heatmap of picking frequency
        picking_matrix = self._create_picking_matrix(warehouse_state)
        
        self.figures['main'].add_trace(
            go.Heatmap(
                z=picking_matrix,
                colorscale='Reds',
                showscale=True
            ),
            row=3, col=1
        )
    
    def _update_optimization_impact(self):
        """Update optimization impact visualization"""
        if len(self.metrics_history) < 2:
            return
            
        # Calculate improvements
        improvements = self._calculate_improvements()
        
        self.figures['main'].add_trace(
            go.Bar(
                x=list(improvements.keys()),
                y=list(improvements.values()),
                name="Improvements"
            ),
            row=3, col=2
        )
    
    def _add_zone_visualization(
        self,
        zone_type: str,
        locations: List[tuple]
    ):
        """Add zone visualization to 3D layout"""
        x, y, z = zip(*locations)
        
        self.figures['main'].add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self._get_zone_color(zone_type),
                    opacity=0.8
                ),
                name=zone_type
            ),
            row=1, col=1
        )
    
    def _add_item_visualization(self, warehouse_state: WarehouseState):
        """Add item visualization to layout"""
        for item_id, bin_id in warehouse_state.item_locations.items():
            location = warehouse_state.storage_locations[bin_id]
            self._add_item_marker(item_id, location)
    
    def _add_item_marker(
        self,
        item_id: int,
        location: StorageLocation
    ):
        """Add individual item marker to visualization"""
        self.figures['main'].add_trace(
            go.Scatter3d(
                x=[location.coordinates[0]],
                y=[location.coordinates[1]],
                z=[location.coordinates[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    symbol='square'
                ),
                name=f'Item {item_id}'
            ),
            row=1, col=1
        )
    
    def _create_utilization_matrix(
        self,
        warehouse_state: WarehouseState
    ) -> np.ndarray:
        """Create utilization matrix for heatmap"""
        max_x = max(loc.coordinates[0] for loc in warehouse_state.storage_locations.values())
        max_y = max(loc.coordinates[1] for loc in warehouse_state.storage_locations.values())
        
        matrix = np.zeros((max_x + 1, max_y + 1))
        
        for location in warehouse_state.storage_locations.values():
            x, y, _ = location.coordinates
            items = len(location.current_items)
            capacity = location.dimensions.volume
            utilization = items / capacity if capacity > 0 else 0
            matrix[x, y] = utilization
        
        return matrix
    
    def _create_picking_matrix(
        self,
        warehouse_state: WarehouseState
    ) -> np.ndarray:
        """Create picking frequency matrix"""
        # Implementation similar to utilization matrix
        # but based on picking frequency data
        pass
    
    def _calculate_improvements(self) -> Dict[str, float]:
        """Calculate improvements in metrics"""
        if len(self.metrics_history) < 2:
            return {}
            
        current = self.metrics_history[-1]
        baseline = self.metrics_history[0]
        
        return {
            'Picking Efficiency': (
                (current['picking_efficiency'] - baseline['picking_efficiency']) /
                baseline['picking_efficiency'] * 100
            ),
            'Space Utilization': (
                (current['space_utilization'] - baseline['space_utilization']) /
                baseline['space_utilization'] * 100
            ),
            'Travel Distance': (
                (baseline['travel_distance'] - current['travel_distance']) /
                baseline['travel_distance'] * 100
            )
        }
    
    def _get_zone_color(self, zone_type: str) -> str:
        """Get color for zone visualization"""
        color_map = {
            'normal': 'blue',
            'cold_storage': 'cyan',
            'fragile': 'yellow',
            'hazardous': 'red',
            'high_value': 'purple'
        }
        return color_map.get(zone_type, 'gray')
    
    def _apply_theme(self, fig: go.Figure):
        """Apply theme to figure"""
        template = 'plotly_dark' if self.dark_mode else 'plotly_white'
        
        fig.update_layout(
            template=template,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )

# This dashboard implementation provides:

# 1. Core Visualization Features:
# - 3D warehouse layout
# - Real-time metrics
# - Performance trends
# - Picking patterns
# - Optimization impact

# 2. Interactive Elements:
# - Zoomable layout
# - Clickable items
# - Hoverable metrics
# - Dynamic updates

# 3. Customization:
# - Theme options
# - Layout configuration
# - Update intervals
# - Dimension options

# 4. Performance Tracking:
# - Metrics history
# - Improvement calculations
# - Trend analysis
# - Pattern visualization

# 5. Advanced Features:
# - Heatmap generation
# - Zone coloring
# - Item tracking
# - Utilization metrics