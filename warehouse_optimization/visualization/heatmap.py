from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..core.types import (
    WarehouseState,
    StorageLocation,
    ProductAttributes
)

@dataclass
class HeatmapConfig:
    """Configuration for heatmap visualization"""
    colorscale: str = 'Viridis'
    show_labels: bool = True
    label_format: str = '.2f'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    opacity: float = 0.8
    enable_hover: bool = True
    custom_colormap: Optional[Dict[str, str]] = None

class WarehouseHeatmap:
    """
    Specialized heatmap visualizations for warehouse analytics.
    Provides various heatmap views for different metrics and patterns.
    """
    
    def __init__(
        self,
        config: Optional[HeatmapConfig] = None,
        enable_animations: bool = True
    ):
        self.config = config or HeatmapConfig()
        self.enable_animations = enable_animations
        
        # Initialize tracking
        self.history: Dict[str, List[np.ndarray]] = {
            'picking_frequency': [],
            'utilization': [],
            'demand_patterns': [],
            'traffic_density': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def create_picking_heatmap(
        self,
        warehouse_state: WarehouseState,
        picking_data: pd.DataFrame,
        time_window: Optional[timedelta] = None
    ) -> go.Figure:
        """
        Create heatmap of picking frequency
        
        Args:
            warehouse_state: Current warehouse state
            picking_data: Historical picking data
            time_window: Optional time window for analysis
            
        Returns:
            Plotly figure with picking frequency heatmap
        """
        try:
            # Process picking data
            picking_matrix = self._calculate_picking_frequency(
                warehouse_state,
                picking_data,
                time_window
            )
            
            # Create figure
            fig = self._create_heatmap_figure(
                picking_matrix,
                "Picking Frequency Heatmap",
                "Number of Picks"
            )
            
            # Store in history
            self.history['picking_frequency'].append(picking_matrix)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating picking heatmap: {str(e)}")
            return None
    
    def create_utilization_heatmap(
        self,
        warehouse_state: WarehouseState,
        include_reserved: bool = True
    ) -> go.Figure:
        """Create heatmap of space utilization"""
        try:
            # Calculate utilization
            utilization_matrix = self._calculate_utilization(
                warehouse_state,
                include_reserved
            )
            
            # Create figure
            fig = self._create_heatmap_figure(
                utilization_matrix,
                "Space Utilization Heatmap",
                "Utilization %"
            )
            
            # Store in history
            self.history['utilization'].append(utilization_matrix)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating utilization heatmap: {str(e)}")
            return None
    
    def create_demand_heatmap(
        self,
        warehouse_state: WarehouseState,
        demand_forecasts: Dict[int, np.ndarray]
    ) -> go.Figure:
        """Create heatmap of predicted demand patterns"""
        try:
            # Calculate demand patterns
            demand_matrix = self._calculate_demand_patterns(
                warehouse_state,
                demand_forecasts
            )
            
            # Create figure
            fig = self._create_heatmap_figure(
                demand_matrix,
                "Predicted Demand Patterns",
                "Predicted Demand"
            )
            
            # Store in history
            self.history['demand_patterns'].append(demand_matrix)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating demand heatmap: {str(e)}")
            return None
    
    def create_traffic_heatmap(
        self,
        warehouse_state: WarehouseState,
        traffic_data: pd.DataFrame,
        time_window: Optional[timedelta] = None
    ) -> go.Figure:
        """Create heatmap of picking traffic density"""
        try:
            # Calculate traffic density
            traffic_matrix = self._calculate_traffic_density(
                warehouse_state,
                traffic_data,
                time_window
            )
            
            # Create figure
            fig = self._create_heatmap_figure(
                traffic_matrix,
                "Traffic Density Heatmap",
                "Traffic Density"
            )
            
            # Store in history
            self.history['traffic_density'].append(traffic_matrix)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating traffic heatmap: {str(e)}")
            return None
    
    def _calculate_picking_frequency(
        self,
        warehouse_state: WarehouseState,
        picking_data: pd.DataFrame,
        time_window: Optional[timedelta]
    ) -> np.ndarray:
        """Calculate picking frequency matrix"""
        max_x = max(loc.coordinates[0] for loc in warehouse_state.storage_locations.values())
        max_y = max(loc.coordinates[1] for loc in warehouse_state.storage_locations.values())
        
        frequency_matrix = np.zeros((max_x + 1, max_y + 1))
        
        # Filter by time window if provided
        if time_window:
            cutoff_time = datetime.now() - time_window
            picking_data = picking_data[picking_data['timestamp'] >= cutoff_time]
        
        # Calculate frequencies
        for bin_id, group in picking_data.groupby('bin_id'):
            if bin_id in warehouse_state.storage_locations:
                location = warehouse_state.storage_locations[bin_id]
                x, y, _ = location.coordinates
                frequency_matrix[x, y] = len(group)
        
        return frequency_matrix
    
    def _calculate_utilization(
        self,
        warehouse_state: WarehouseState,
        include_reserved: bool
    ) -> np.ndarray:
        """Calculate space utilization matrix"""
        max_x = max(loc.coordinates[0] for loc in warehouse_state.storage_locations.values())
        max_y = max(loc.coordinates[1] for loc in warehouse_state.storage_locations.values())
        
        utilization_matrix = np.zeros((max_x + 1, max_y + 1))
        
        for location in warehouse_state.storage_locations.values():
            x, y, _ = location.coordinates
            
            # Calculate utilization
            total_volume = location.dimensions.volume
            used_volume = sum(
                item.volume for item in location.current_items.values()
            )
            
            if include_reserved:
                # Add reserved space
                reserved_volume = sum(
                    item.volume for item in location.current_items.values()
                    if hasattr(item, 'is_reserved') and item.is_reserved
                )
                used_volume += reserved_volume
            
            utilization_matrix[x, y] = (used_volume / total_volume * 100) if total_volume > 0 else 0
        
        return utilization_matrix
    
    def _calculate_demand_patterns(
        self,
        warehouse_state: WarehouseState,
        demand_forecasts: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Calculate demand pattern matrix"""
        max_x = max(loc.coordinates[0] for loc in warehouse_state.storage_locations.values())
        max_y = max(loc.coordinates[1] for loc in warehouse_state.storage_locations.values())
        
        demand_matrix = np.zeros((max_x + 1, max_y + 1))
        
        for item_id, forecast in demand_forecasts.items():
            if item_id in warehouse_state.item_locations:
                bin_id = warehouse_state.item_locations[item_id]
                location = warehouse_state.storage_locations[bin_id]
                x, y, _ = location.coordinates
                demand_matrix[x, y] += np.mean(forecast)
        
        return demand_matrix
    
    def _calculate_traffic_density(
        self,
        warehouse_state: WarehouseState,
        traffic_data: pd.DataFrame,
        time_window: Optional[timedelta]
    ) -> np.ndarray:
        """Calculate traffic density matrix"""
        max_x = max(loc.coordinates[0] for loc in warehouse_state.storage_locations.values())
        max_y = max(loc.coordinates[1] for loc in warehouse_state.storage_locations.values())
        
        traffic_matrix = np.zeros((max_x + 1, max_y + 1))
        
        # Filter by time window if provided
        if time_window:
            cutoff_time = datetime.now() - time_window
            traffic_data = traffic_data[traffic_data['timestamp'] >= cutoff_time]
        
        # Calculate traffic density
        for _, row in traffic_data.iterrows():
            path = self._decode_path(row['path'])
            for x, y in path:
                if 0 <= x <= max_x and 0 <= y <= max_y:
                    traffic_matrix[x, y] += 1
        
        return traffic_matrix
    
    def _create_heatmap_figure(
        self,
        matrix: np.ndarray,
        title: str,
        colorbar_title: str
    ) -> go.Figure:
        """Create Plotly heatmap figure"""
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale=self.config.colorscale,
            showscale=True,
            text=matrix if self.config.show_labels else None,
            texttemplate=f'%{{z:{self.config.label_format}}}' if self.config.show_labels else None,
            textfont={"size": 10},
            hoverongaps=self.config.enable_hover,
            hovertemplate=(
                f'X: %{{x}}<br>'
                f'Y: %{{y}}<br>'
                f'{colorbar_title}: %{{z:{self.config.label_format}}}<extra></extra>'
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            coloraxis_colorbar_title=colorbar_title
        )
        
        return fig
    
    def _decode_path(self, path_string: str) -> List[Tuple[int, int]]:
        """Decode path string to list of coordinates"""
        try:
            return [
                tuple(map(int, coord.split(',')))
                for coord in path_string.split(';')
            ]
        except Exception:
            return []
    
    def get_history_animation(
        self,
        metric_type: str,
        frame_duration: int = 500
    ) -> Optional[go.Figure]:
        """Create animation of historical heatmaps"""
        if not self.enable_animations or not self.history.get(metric_type):
            return None
            
        frames = []
        for i, matrix in enumerate(self.history[metric_type]):
            frames.append(
                go.Frame(
                    data=[go.Heatmap(z=matrix)],
                    name=f'frame{i}'
                )
            )
        
        fig = go.Figure(
            data=[go.Heatmap(z=self.history[metric_type][0])],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    dict(label='Play',
                         method='animate',
                         args=[None, {'frame': {'duration': frame_duration}}]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0}}])
                ]
            }]
        )
        
        return fig

# This implementation provides:

# 1. Core Heatmap Types:
# - Picking frequency
# - Space utilization
# - Demand patterns
# - Traffic density

# 2. Advanced Features:
# - Customizable configurations
# - Animation support
# - History tracking
# - Multiple metrics

# 3. Visualization Options:
# - Label customization
# - Color schemes
# - Hover information
# - Opacity control

# 4. Analysis Features:
# - Time window filtering
# - Reserved space tracking
# - Path analysis
# - Pattern detection

# 5. Interactive Elements:
# - Animated history
# - Playback controls
# - Hover details
# - Dynamic updates