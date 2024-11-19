from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from datetime import datetime
import numpy as np

@dataclass
class ItemDimensions:
    """Physical dimensions and properties of an item"""
    length: float
    width: float
    height: float
    weight: float
    volume: float = field(init=False)
    
    def __post_init__(self):
        self.volume = self.length * self.width * self.height

@dataclass
class BinDimensions:
    """Physical dimensions and properties of a storage bin"""
    length: float
    width: float
    height: float
    max_weight: float
    volume: float = field(init=False)
    
    def __post_init__(self):
        self.volume = self.length * self.width * self.height

@dataclass
class ProductAttributes:
    """Product-specific attributes and constraints"""
    product_id: int
    category: str
    is_fragile: bool
    requires_cold_storage: bool
    is_hazardous: bool
    max_stack: int
    shelf_life_days: Optional[int]
    price: float
    dimensions: ItemDimensions
    compatible_zones: Set[str] = field(default_factory=set)
    incompatible_products: Set[int] = field(default_factory=set)

@dataclass
class StorageLocation:
    """Storage location information"""
    bin_id: str
    aisle: str
    shelf: str
    bin: str
    zone_type: str
    dimensions: BinDimensions
    coordinates: Tuple[int, int, int]  # x, y, z coordinates
    distance_to_pickup: float
    requires_ladder: bool
    current_weight: float = 0.0
    current_items: Dict[int, ItemDimensions] = field(default_factory=dict)

@dataclass
class WarehouseState:
    """Current state of the warehouse"""
    layout_grid: np.ndarray
    zones: Dict[str, List[Tuple[int, int, int]]]
    storage_locations: Dict[str, StorageLocation]
    item_locations: Dict[int, str]  # product_id -> bin_id
    picking_area_location: Tuple[int, int, int]

@dataclass
class OptimizationState:
    """State for optimization decisions"""
    current_layout: Dict[str, Any]
    demand_forecasts: Dict[int, np.ndarray]
    constraint_violations: List[str]
    performance_metrics: Dict[str, float]

@dataclass
class OrderInfo:
    """Order information"""
    order_id: int
    timestamp: datetime
    items: List[Tuple[int, int]]  # List of (product_id, quantity)
    status: str

@dataclass
class PlacementConstraints:
    """Constraints for item placement"""
    zone_rules: Dict[str, Dict[str, Any]]
    product_rules: Dict[str, Dict[str, Any]]
    bin_rules: Dict[str, Dict[str, Any]]
    physical_constraints: Dict[str, Dict[str, float]]
    affinity_rules: Dict[str, float]
    demand_rules: Dict[str, float]

@dataclass
class DemandForecast:
    """Demand forecast information"""
    product_id: int
    forecast_values: np.ndarray
    confidence_intervals: np.ndarray
    seasonality_factors: Dict[str, float]
    trend_factor: float

@dataclass
class PickingRoute:
    """Picking route information"""
    order_id: int
    items: List[Tuple[int, str]]  # List of (product_id, bin_id)
    route: List[Tuple[int, int, int]]  # List of coordinates
    estimated_time: float
    total_distance: float

@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance"""
    picking_efficiency: float
    space_utilization: float
    constraint_satisfaction: float
    demand_satisfaction: float
    travel_distance: float
    total_time: float
    num_moves: int

@dataclass
class BinState:
    """Current state of a bin"""
    dimensions: BinDimensions
    current_items: Dict[int, ItemDimensions]
    current_weight: float
    max_items: int = field(init=False)
    available_volume: float = field(init=False)
    utilization: float = field(init=False)
    
    def __post_init__(self):
        self.max_items = self._calculate_max_items()
        self.available_volume = self._calculate_available_volume()
        self.utilization = self._calculate_utilization()
    
    def _calculate_max_items(self) -> int:
        """Calculate maximum items based on bin dimensions"""
        avg_small_item_volume = 0.001  # m³
        theoretical_max = int(self.dimensions.volume / avg_small_item_volume)
        return min(theoretical_max, 10)  # Cap at 10 for practical purposes
    
    def _calculate_available_volume(self) -> float:
        """Calculate available volume in the bin"""
        used_volume = sum(item.volume for item in self.current_items.values())
        return self.dimensions.volume - used_volume
    
    def _calculate_utilization(self) -> float:
        """Calculate current bin utilization"""
        return 1 - (self.available_volume / self.dimensions.volume)

@dataclass
class ActionSpace:
    """Action space definition for the RL environment"""
    placement_locations: List[Tuple[int, int, int]]
    valid_orientations: List[str]
    valid_zones: List[str]
    
    def get_action_size(self) -> int:
        """Get total size of action space"""
        return (
            len(self.placement_locations) *
            len(self.valid_orientations) *
            len(self.valid_zones)
        )

@dataclass
class ObservationSpace:
    """Observation space definition for the RL environment"""
    layout_shape: Tuple[int, int, int]
    num_products: int
    num_zones: int
    num_features: int
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """Get shape of observation space"""
        return (
            *self.layout_shape,
            self.num_products + self.num_zones + self.num_features
        )

# Type aliases for common uses
LocationCoord = Tuple[int, int, int]
BinID = str
ProductID = int
OrderID = int
ZoneType = str
Quantity = int
Distance = float
Weight = float
Volume = float
TimeStamp = datetime

# ```

# This `types.py` file provides:

# 1. Core Data Structures:
# - ItemDimensions & BinDimensions for physical properties
# - ProductAttributes for product-specific information
# - StorageLocation for bin/location information
# - WarehouseState for overall warehouse state
# - OptimizationState for optimization process

# 2. Optimization Types:
# - PlacementConstraints for rules and constraints
# - DemandForecast for demand prediction data
# - PickingRoute for order picking information
# - OptimizationMetrics for performance tracking

# 3. Bin Management:
# - BinState for detailed bin status
# - Automatic calculations for utilization and capacity

# 4. RL-Specific Types:
# - ActionSpace for defining possible actions
# - ObservationSpace for state representation

# 5. Utility Types:
# - Type aliases for common data types
# - Comprehensive type hints
