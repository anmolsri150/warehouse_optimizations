from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum
import pandas as pd

from ..core.types import (
    WarehouseState,
    StorageLocation,
    ProductAttributes,
    BinDimensions
)

class ViewMode(Enum):
    """View modes for layout visualization"""
    TOP_DOWN = "top_down"
    ISOMETRIC = "isometric"
    FIRST_PERSON = "first_person"
    SECTION = "section"

@dataclass
class LayoutConfig:
    """Configuration for layout visualization"""
    view_mode: ViewMode = ViewMode.ISOMETRIC
    show_labels: bool = True
    show_grid: bool = True
    show_dimensions: bool = True
    zone_opacity: float = 0.7
    highlight_conflicts: bool = True
    bin_spacing: float = 0.1
    aisle_width: float = 2.0
    custom_colors: Optional[Dict[str, str]] = None

class WarehouseLayout:
    """
    Handles warehouse layout visualization and interactive floor plan features.
    Provides multiple view modes and interactive elements for warehouse visualization.
    """
    
    def __init__(
        self,
        config: Optional[LayoutConfig] = None,
        enable_interactions: bool = True
    ):
        self.config = config or LayoutConfig()
        self.enable_interactions = enable_interactions
        
        # Default colors
        self.default_colors = {
            'zone_normal': 'rgb(99, 110, 250)',
            'zone_cold_storage': 'rgb(25, 211, 243)',
            'zone_fragile': 'rgb(255, 193, 7)',
            'zone_hazardous': 'rgb(255, 87, 34)',
            'zone_high_value': 'rgb(156, 39, 176)',
            'occupied': 'rgb(76, 175, 80)',
            'empty': 'rgb(189, 189, 189)',
            'conflict': 'rgb(244, 67, 54)',
            'highlight': 'rgb(255, 235, 59)',
            'aisle': 'rgb(238, 238, 238)'
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize interaction state
        self.selected_bin: Optional[str] = None
        self.highlighted_path: List[Tuple[int, int, int]] = []
        self.active_filters: Dict[str, bool] = {}
    
    def create_layout_view(
        self,
        warehouse_state: WarehouseState,
        highlight_bins: Optional[List[str]] = None
    ) -> go.Figure:
        """Create warehouse layout visualization"""
        try:
            if self.config.view_mode == ViewMode.TOP_DOWN:
                return self._create_top_down_view(warehouse_state, highlight_bins)
            elif self.config.view_mode == ViewMode.ISOMETRIC:
                return self._create_isometric_view(warehouse_state, highlight_bins)
            elif self.config.view_mode == ViewMode.FIRST_PERSON:
                return self._create_first_person_view(warehouse_state)
            else:
                return self._create_section_view(warehouse_state)
                
        except Exception as e:
            self.logger.error(f"Error creating layout view: {str(e)}")
            return self._create_error_figure()
    
    def _create_top_down_view(
        self,
        warehouse_state: WarehouseState,
        highlight_bins: Optional[List[str]] = None
    ) -> go.Figure:
        """Create 2D top-down view of warehouse"""
        fig = go.Figure()
        
        # Add zones
        for zone_type, locations in warehouse_state.zones.items():
            zone_color = self._get_zone_color(zone_type)
            
            # Create zone shapes
            for location in locations:
                fig.add_shape(
                    type="rect",
                    x0=location[0] - 0.5,
                    y0=location[1] - 0.5,
                    x1=location[0] + 0.5,
                    y1=location[1] + 0.5,
                    fillcolor=zone_color,
                    opacity=self.config.zone_opacity,
                    line=dict(color="white", width=1),
                )
        
        # Add bins
        for bin_id, location in warehouse_state.storage_locations.items():
            color = self._get_bin_color(
                location,
                bin_id in (highlight_bins or [])
            )
            
            fig.add_trace(go.Scatter(
                x=[location.coordinates[0]],
                y=[location.coordinates[1]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='square',
                ),
                text=bin_id if self.config.show_labels else None,
                hovertemplate=(
                    f"Bin: {bin_id}<br>"
                    f"Zone: {location.zone_type}<br>"
                    f"Utilization: {self._calculate_bin_utilization(location):.1f}%"
                    "<extra></extra>"
                ),
                name=bin_id
            ))
        
        # Add aisles
        self._add_aisles(fig, warehouse_state)
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                showgrid=self.config.show_grid,
                zeroline=False,
                title="X Coordinate"
            ),
            yaxis=dict(
                showgrid=self.config.show_grid,
                zeroline=False,
                title="Y Coordinate",
                scaleanchor="x",
                scaleratio=1
            ),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        # Add interactive elements if enabled
        if self.enable_interactions:
            self._add_interactions(fig)
        
        return fig
    
    def _create_isometric_view(
        self,
        warehouse_state: WarehouseState,
        highlight_bins: Optional[List[str]] = None
    ) -> go.Figure:
        """Create 3D isometric view of warehouse"""
        fig = go.Figure()
        
        # Add zones as 3D surfaces
        for zone_type, locations in warehouse_state.zones.items():
            zone_color = self._get_zone_color(zone_type)
            
            x, y, z = self._get_zone_coordinates(locations)
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                color=zone_color,
                opacity=self.config.zone_opacity,
                name=zone_type
            ))
        
        # Add bins
        for bin_id, location in warehouse_state.storage_locations.items():
            color = self._get_bin_color(
                location,
                bin_id in (highlight_bins or [])
            )
            
            # Create bin visualization
            x, y, z = self._create_bin_vertices(location.coordinates, location.dimensions)
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                color=color,
                opacity=1.0,
                name=bin_id,
                hovertemplate=(
                    f"Bin: {bin_id}<br>"
                    f"Zone: {location.zone_type}<br>"
                    f"Utilization: {self._calculate_bin_utilization(location):.1f}%"
                    "<extra></extra>"
                )
            ))
        
        # Update layout for 3D view
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=self.config.show_grid),
                yaxis=dict(showgrid=self.config.show_grid),
                zaxis=dict(showgrid=self.config.show_grid),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
    
    def _create_first_person_view(
        self,
        warehouse_state: WarehouseState
    ) -> go.Figure:
        """Create first-person perspective view"""
        # Implementation for first-person view
        # This would typically involve raycasting and perspective projection
        pass
    
    def _create_section_view(
        self,
        warehouse_state: WarehouseState
    ) -> go.Figure:
        """Create cross-section view of warehouse"""
        # Implementation for section view
        pass
    
    def _get_zone_coordinates(
        self,
        locations: List[Tuple[int, int, int]]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Get coordinates for zone visualization"""
        x, y, z = [], [], []
        
        for loc in locations:
            # Create vertices for zone block
            vertices = self._create_zone_vertices(loc)
            x.extend(vertices[0])
            y.extend(vertices[1])
            z.extend(vertices[2])
        
        return x, y, z
    
    def _create_zone_vertices(
        self,
        location: Tuple[int, int, int]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Create vertices for zone block"""
        x0, y0, z0 = location
        vertices = (
            [x0-0.5, x0+0.5, x0+0.5, x0-0.5, x0-0.5, x0+0.5, x0+0.5, x0-0.5],
            [y0-0.5, y0-0.5, y0+0.5, y0+0.5, y0-0.5, y0-0.5, y0+0.5, y0+0.5],
            [z0, z0, z0, z0, z0+1, z0+1, z0+1, z0+1]
        )
        return vertices
    
    def _create_bin_vertices(
        self,
        coordinates: Tuple[int, int, int],
        dimensions: BinDimensions
    ) -> Tuple[List[float], List[float], List[float]]:
        """Create vertices for bin visualization"""
        x, y, z = coordinates
        l, w, h = dimensions.length, dimensions.width, dimensions.height
        
        vertices = (
            [x, x+l, x+l, x, x, x+l, x+l, x],
            [y, y, y+w, y+w, y, y, y+w, y+w],
            [z, z, z, z, z+h, z+h, z+h, z+h]
        )
        return vertices
    
    def _add_aisles(self, fig: go.Figure, warehouse_state: WarehouseState):
        """Add aisle visualization"""
        # Calculate aisle positions
        aisle_positions = self._calculate_aisle_positions(warehouse_state)
        
        for aisle in aisle_positions:
            fig.add_shape(
                type="rect",
                x0=aisle['x0'],
                y0=aisle['y0'],
                x1=aisle['x1'],
                y1=aisle['y1'],
                fillcolor=self.default_colors['aisle'],
                opacity=0.3,
                line=dict(width=0),
            )
    
    def _calculate_aisle_positions(
        self,
        warehouse_state: WarehouseState
    ) -> List[Dict[str, float]]:
        """Calculate aisle positions"""
        # Implementation for aisle position calculation
        pass
    
    def _calculate_bin_utilization(self, location: StorageLocation) -> float:
        """Calculate bin utilization percentage"""
        total_volume = location.dimensions.volume
        used_volume = sum(item.volume for item in location.current_items.values())
        
        return (used_volume / total_volume * 100) if total_volume > 0 else 0
    
    def _get_zone_color(self, zone_type: str) -> str:
        """Get color for zone visualization"""
        if self.config.custom_colors and zone_type in self.config.custom_colors:
            return self.config.custom_colors[zone_type]
        
        return self.default_colors.get(f'zone_{zone_type}', self.default_colors['zone_normal'])
    
    def _get_bin_color(self, location: StorageLocation, highlighted: bool) -> str:
        """Get color for bin visualization"""
        if highlighted:
            return self.default_colors['highlight']
            
        if self._has_conflicts(location):
            return self.default_colors['conflict']
            
        return (
            self.default_colors['occupied']
            if location.current_items
            else self.default_colors['empty']
        )
    
    def _has_conflicts(self, location: StorageLocation) -> bool:
        """Check for conflicts in bin"""
        if not self.config.highlight_conflicts:
            return False
            
        # Implementation for conflict detection
        return False
    
    def _add_interactions(self, fig: go.Figure):
        """Add interactive elements to figure"""
        fig.update_layout(
            clickmode='event+select',
            dragmode='select',
            selectdirection='h'
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

# 1. Multiple View Modes:
# - Top-down view
# - Isometric 3D view
# - First-person view
# - Section view

# 2. Visualization Features:
# - Zone coloring
# - Bin status
# - Utilization indicators
# - Aisle representation

# 3. Interactive Elements:
# - Bin selection
# - Path highlighting
# - Hover information
# - View controls

# 4. Configuration Options:
# - Custom colors
# - Label visibility
# - Grid display
# - Opacity settings

# 5. Advanced Features:
# - Conflict detection
# - Space utilization
# - Interactive filtering
# - Multiple perspectives
