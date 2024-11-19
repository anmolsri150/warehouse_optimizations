import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import pandas as pd
from collections import defaultdict

from ..core.types import (
    WarehouseState,
    StorageLocation,
    ProductAttributes,
    BinState,
    ItemDimensions,
    OptimizationState
)

@dataclass
class ZoneState:
    """Tracks state of a warehouse zone"""
    zone_type: str
    locations: Dict[str, StorageLocation]
    capacity: Dict[str, float]  # Different capacity types (volume, weight, etc.)
    utilization: Dict[str, float]
    environmental_conditions: Dict[str, float]
    last_updated: datetime

@dataclass
class WarehouseMetrics:
    """Tracks warehouse performance metrics"""
    total_volume_utilization: float
    total_weight_utilization: float
    zone_utilization: Dict[str, float]
    picking_efficiency: float
    constraint_satisfaction: float
    last_updated: datetime

class StateManager:
    """
    Manages the overall state of the warehouse including zones,
    locations, and transitions. Tracks metrics and provides state
    information for optimization.
    """
    
    def __init__(
        self,
        layout_grid: np.ndarray,
        zones: Dict[str, List[Tuple[int, int, int]]],
        storage_locations: Dict[str, StorageLocation],
        update_frequency: int = 300  # seconds
    ):
        self.layout_grid = layout_grid
        self.zones = zones
        self.storage_locations = storage_locations
        self.update_frequency = update_frequency
        
        # Initialize state tracking
        self.zone_states: Dict[str, ZoneState] = {}
        self.item_locations: Dict[int, str] = {}  # product_id -> bin_id
        self.current_metrics = WarehouseMetrics(
            total_volume_utilization=0.0,
            total_weight_utilization=0.0,
            zone_utilization={},
            picking_efficiency=0.0,
            constraint_satisfaction=0.0,
            last_updated=datetime.now()
        )
        
        # Initialize change tracking
        self.state_history: List[Dict] = []
        self.pending_changes: List[Dict] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize zones and states
        self._initialize_zones()
    
    def _initialize_zones(self):
        """Initialize zone states and metrics"""
        for zone_type, locations in self.zones.items():
            zone_locations = {
                loc_id: self.storage_locations[loc_id]
                for loc_id in self._get_zone_locations(zone_type)
            }
            
            self.zone_states[zone_type] = ZoneState(
                zone_type=zone_type,
                locations=zone_locations,
                capacity=self._calculate_zone_capacity(zone_locations),
                utilization=self._calculate_zone_utilization(zone_locations),
                environmental_conditions=self._get_zone_conditions(zone_type),
                last_updated=datetime.now()
            )
    
    def get_current_state(self) -> WarehouseState:
        """Get current warehouse state"""
        self._update_if_needed()
        
        return WarehouseState(
            layout_grid=self.layout_grid,
            zones=self.zones,
            storage_locations=self.storage_locations,
            item_locations=self.item_locations,
            picking_area_location=self._get_picking_area_location()
        )
    
    def update_item_location(
        self,
        product_id: int,
        bin_id: str,
        validate: bool = True
    ) -> bool:
        """Update item location with optional validation"""
        if validate and not self._validate_location_update(product_id, bin_id):
            return False
        
        # Record previous state
        previous_bin = self.item_locations.get(product_id)
        
        # Update location
        self.item_locations[product_id] = bin_id
        
        # Record change
        self._record_state_change({
            'timestamp': datetime.now(),
            'action': 'move_item',
            'product_id': product_id,
            'from_bin': previous_bin,
            'to_bin': bin_id
        })
        
        # Update affected zones
        self._update_affected_zones(product_id, previous_bin, bin_id)
        
        return True
    
    def get_zone_state(self, zone_type: str) -> Optional[ZoneState]:
        """Get current state of a specific zone"""
        return self.zone_states.get(zone_type)
    
    def get_bin_state(self, bin_id: str) -> Optional[BinState]:
        """Get current state of a specific bin"""
        location = self.storage_locations.get(bin_id)
        if not location:
            return None
            
        return BinState(
            dimensions=location.dimensions,
            current_items=location.current_items,
            current_weight=sum(
                item.weight for item in location.current_items.values()
            )
        )
    
    def get_optimization_state(self) -> OptimizationState:
        """Get current optimization state"""
        return OptimizationState(
            current_layout=self._get_layout_state(),
            demand_forecasts={},  # To be filled by forecasting module
            constraint_violations=self._get_current_violations(),
            performance_metrics=self._get_performance_metrics()
        )
    
    def get_utilization_metrics(self) -> Dict[str, float]:
        """Get current utilization metrics"""
        self._update_if_needed()
        
        return {
            'total_volume': self.current_metrics.total_volume_utilization,
            'total_weight': self.current_metrics.total_weight_utilization,
            'zones': self.current_metrics.zone_utilization,
            'picking_efficiency': self.current_metrics.picking_efficiency
        }
    
    def simulate_item_placement(
        self,
        product_id: int,
        bin_id: str
    ) -> Tuple[bool, Dict[str, float]]:
        """Simulate placing an item without actually updating state"""
        # Create temporary state
        temp_state = self._create_temporary_state()
        
        # Try placement in temporary state
        success = self._simulate_placement(
            temp_state,
            product_id,
            bin_id
        )
        
        if success:
            # Calculate metrics for simulated state
            metrics = self._calculate_state_metrics(temp_state)
            return True, metrics
        
        return False, {}
    
    def get_available_locations(
        self,
        product_attributes: ProductAttributes
    ) -> List[str]:
        """Get list of available locations for a product"""
        available_locations = []
        
        for bin_id, location in self.storage_locations.items():
            if self._is_location_suitable(location, product_attributes):
                available_locations.append(bin_id)
        
        return available_locations
    
    def _update_if_needed(self):
        """Update state if update frequency has passed"""
        if (datetime.now() - self.current_metrics.last_updated).total_seconds() > self.update_frequency:
            self._update_metrics()
    
    def _update_metrics(self):
        """Update all warehouse metrics"""
        total_volume = 0.0
        total_weight = 0.0
        zone_utils = {}
        
        for zone_type, zone_state in self.zone_states.items():
            # Update zone metrics
            zone_metrics = self._calculate_zone_metrics(zone_state)
            zone_utils[zone_type] = zone_metrics['utilization']
            
            total_volume += zone_metrics['volume_used']
            total_weight += zone_metrics['weight_used']
        
        self.current_metrics = WarehouseMetrics(
            total_volume_utilization=total_volume / self._get_total_volume(),
            total_weight_utilization=total_weight / self._get_total_weight_capacity(),
            zone_utilization=zone_utils,
            picking_efficiency=self._calculate_picking_efficiency(),
            constraint_satisfaction=self._calculate_constraint_satisfaction(),
            last_updated=datetime.now()
        )
    
    def _calculate_zone_metrics(self, zone_state: ZoneState) -> Dict[str, float]:
        """Calculate metrics for a specific zone"""
        total_volume = 0.0
        used_volume = 0.0
        total_weight = 0.0
        used_weight = 0.0
        
        for location in zone_state.locations.values():
            total_volume += location.dimensions.volume
            total_weight += location.dimensions.max_weight
            
            for item in location.current_items.values():
                used_volume += item.volume
                used_weight += item.weight
        
        return {
            'volume_used': used_volume,
            'volume_total': total_volume,
            'weight_used': used_weight,
            'weight_total': total_weight,
            'utilization': (used_volume / total_volume) if total_volume > 0 else 0.0
        }
    
    def _calculate_picking_efficiency(self) -> float:
        """Calculate current picking efficiency"""
        # Implementation depends on specific efficiency metrics
        return 0.0
    
    def _calculate_constraint_satisfaction(self) -> float:
        """Calculate constraint satisfaction rate"""
        violations = self._get_current_violations()
        total_items = len(self.item_locations)
        
        if total_items == 0:
            return 1.0
            
        return 1.0 - (len(violations) / total_items)
    
    def _record_state_change(self, change: Dict):
        """Record a state change"""
        self.state_history.append(change)
        
        # Keep history within reasonable size
        if len(self.state_history) > 10000:
            self.state_history = self.state_history[-10000:]
    
    def _get_zone_locations(self, zone_type: str) -> List[str]:
        """Get all location IDs in a zone"""
        return [
            loc_id
            for loc_id, location in self.storage_locations.items()
            if location.zone_type == zone_type
        ]
    
    def _validate_location_update(
        self,
        product_id: int,
        bin_id: str
    ) -> bool:
        """Validate a location update"""
        if bin_id not in self.storage_locations:
            return False
            
        # Additional validation logic
        return True
    
    def _update_affected_zones(
        self,
        product_id: int,
        previous_bin: Optional[str],
        new_bin: str
    ):
        """Update states of affected zones"""
        affected_zones = set()
        
        if previous_bin:
            affected_zones.add(
                self.storage_locations[previous_bin].zone_type
            )
            
        affected_zones.add(
            self.storage_locations[new_bin].zone_type
        )
        
        for zone_type in affected_zones:
            self._update_zone_state(zone_type)
    
    def _update_zone_state(self, zone_type: str):
        """Update state of a specific zone"""
        zone_state = self.zone_states[zone_type]
        
        zone_state.utilization = self._calculate_zone_utilization(
            zone_state.locations
        )
        zone_state.environmental_conditions = self._get_zone_conditions(
            zone_type
        )
        zone_state.last_updated = datetime.now()
    
    def _get_layout_state(self) -> Dict:
        """Get current layout state"""
        return {
            'grid': self.layout_grid.copy(),
            'occupied_locations': self._get_occupied_locations(),
            'zone_boundaries': self._get_zone_boundaries()
        }
    
    def _get_occupied_locations(self) -> Dict[str, List[str]]:
        """Get currently occupied locations by zone"""
        occupied = defaultdict(list)
        
        for product_id, bin_id in self.item_locations.items():
            zone_type = self.storage_locations[bin_id].zone_type
            occupied[zone_type].append(bin_id)
            
        return dict(occupied)
    
    def _get_zone_boundaries(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """Get zone boundaries"""
        return self.zones.copy()
    
    def get_state_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get state change history within time range"""
        if not start_time and not end_time:
            return self.state_history
            
        filtered_history = []
        for change in self.state_history:
            timestamp = change['timestamp']
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered_history.append(change)
            
        return filtered_history

# This `state_manager.py` provides:

# 1. Core State Management:
# - Warehouse layout tracking
# - Zone state management
# - Location state tracking
# - Item placement management

# 2. Metrics Tracking:
# - Utilization metrics
# - Performance metrics
# - Zone-specific metrics
# - Picking efficiency

# 3. State Operations:
# - State updates
# - State validation
# - Change tracking
# - History management

# 4. Zone Management:
# - Zone initialization
# - Zone state updates
# - Zone metrics
# - Environmental conditions

# 5. Advanced Features:
# - State simulation
# - Placement validation
# - Metric calculations
# - History filtering

# The StateManager acts as the central source of truth for the warehouse state and provides comprehensive functionality for state management and tracking.
