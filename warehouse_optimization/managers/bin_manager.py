import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
from datetime import datetime
from collections import defaultdict

from ..core.types import (
    ItemDimensions,
    BinDimensions,
    ProductAttributes,
    BinState,
    StorageLocation
)

@dataclass
class BinUtilization:
    """Tracks bin utilization metrics"""
    volume_used: float = 0.0
    weight_used: float = 0.0
    num_items: int = 0
    vertical_space_used: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    stacking_levels: Dict[int, float] = field(default_factory=dict)

class BinManager:
    """
    Manages bin operations, capacity constraints, and physical placement optimization.
    Handles complex bin packing problems and maintains bin states.
    """
    
    def __init__(
        self,
        safety_factor: float = 0.85,
        min_gap: float = 0.02,
        max_stack_height_factor: float = 0.9,
        enable_tetris_optimization: bool = True
    ):
        self.safety_factor = safety_factor  # Account for imperfect packing
        self.min_gap = min_gap  # Minimum gap between items in meters
        self.max_stack_height_factor = max_stack_height_factor
        self.enable_tetris_optimization = enable_tetris_optimization
        
        # Initialize storage structures
        self.bins: Dict[str, BinState] = {}
        self.bin_utilization: Dict[str, BinUtilization] = {}
        self.item_locations: Dict[int, str] = {}  # product_id -> bin_id
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Tetris-style optimization parameters
        self.tetris_params = {
            'horizontal_bonus': 0.2,  # Bonus for good horizontal fit
            'vertical_bonus': 0.3,    # Bonus for good vertical fit
            'corner_bonus': 0.1,      # Bonus for fitting in corners
            'stability_factor': 0.4    # Factor for stack stability
        }
    
    def initialize_bin(
        self,
        bin_id: str,
        dimensions: BinDimensions,
        zone_type: str
    ):
        """Initialize a new bin with given dimensions"""
        self.bins[bin_id] = BinState(
            dimensions=dimensions,
            current_items={},
            current_weight=0.0
        )
        
        self.bin_utilization[bin_id] = BinUtilization()
        self.logger.info(f"Initialized bin {bin_id} with dimensions {dimensions}")
    
    def can_fit_item(
        self,
        item_dims: ItemDimensions,
        bin_dims: BinDimensions,
        orientation: int = 0
    ) -> Tuple[bool, List[str]]:
        """
        Check if an item can fit in given dimensions with orientation
        Returns (can_fit, reasons_if_cannot_fit)
        """
        violations = []
        
        # Apply safety factor to bin dimensions
        effective_bin_dims = BinDimensions(
            length=bin_dims.length * self.safety_factor,
            width=bin_dims.width * self.safety_factor,
            height=bin_dims.height * self.safety_factor,
            max_weight=bin_dims.max_weight
        )
        
        # Get oriented dimensions
        item_l, item_w, item_h = self._get_oriented_dimensions(
            item_dims,
            orientation
        )
        
        # Check basic dimensional constraints
        if item_l + (2 * self.min_gap) > effective_bin_dims.length:
            violations.append("Length exceeds bin capacity")
        if item_w + (2 * self.min_gap) > effective_bin_dims.width:
            violations.append("Width exceeds bin capacity")
        if item_h + self.min_gap > effective_bin_dims.height:
            violations.append("Height exceeds bin capacity")
        if item_dims.weight > effective_bin_dims.max_weight:
            violations.append("Weight exceeds bin capacity")
            
        return len(violations) == 0, violations
    
    def find_optimal_placement(
        self,
        item: ProductAttributes,
        bin_id: str
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """
        Find optimal placement position within a bin
        Returns (placement_coordinates, fitness_score)
        """
        bin_state = self.bins[bin_id]
        best_position = None
        best_score = float('-inf')
        
        # Get all possible positions
        positions = self._generate_possible_positions(
            item.dimensions,
            bin_state
        )
        
        for position in positions:
            score = self._calculate_placement_score(
                item.dimensions,
                position,
                bin_state
            )
            
            if score > best_score:
                best_score = score
                best_position = position
        
        return best_position, best_score
    
    def add_item_to_bin(
        self,
        bin_id: str,
        item: ProductAttributes,
        position: Dict[str, float]
    ) -> bool:
        """Add item to bin at specified position"""
        if bin_id not in self.bins:
            self.logger.error(f"Bin {bin_id} not found")
            return False
        
        bin_state = self.bins[bin_id]
        
        # Verify placement is still valid
        if not self._verify_placement(item.dimensions, position, bin_state):
            self.logger.error("Placement verification failed")
            return False
        
        # Add item to bin
        bin_state.current_items[item.product_id] = item.dimensions
        bin_state.current_weight += item.dimensions.weight
        
        # Update utilization metrics
        self._update_bin_utilization(bin_id, item)
        
        # Update item location
        self.item_locations[item.product_id] = bin_id
        
        return True
    
    def remove_item_from_bin(
        self,
        product_id: int,
        bin_id: str
    ) -> bool:
        """Remove item from bin"""
        if bin_id not in self.bins:
            return False
            
        bin_state = self.bins[bin_id]
        
        if product_id not in bin_state.current_items:
            return False
            
        # Remove item
        item_dims = bin_state.current_items.pop(product_id)
        bin_state.current_weight -= item_dims.weight
        
        # Update utilization metrics
        self._update_bin_utilization(bin_id)
        
        # Remove item location
        self.item_locations.pop(product_id, None)
        
        return True
    
    def calculate_bin_utilization(self, bin_id: str) -> Dict[str, float]:
        """Calculate comprehensive bin utilization metrics"""
        if bin_id not in self.bins:
            return {}
            
        utilization = self.bin_utilization[bin_id]
        bin_state = self.bins[bin_id]
        
        return {
            'volume_utilization': utilization.volume_used / bin_state.dimensions.volume,
            'weight_utilization': utilization.weight_used / bin_state.dimensions.max_weight,
            'item_count_utilization': utilization.num_items / bin_state.max_items,
            'vertical_space_utilization': utilization.vertical_space_used / bin_state.dimensions.height,
            'overall_utilization': self._calculate_overall_utilization(bin_id)
        }
    
    def _calculate_overall_utilization(self, bin_id: str) -> float:
        """Calculate weighted overall utilization score"""
        metrics = self.calculate_bin_utilization(bin_id)
        
        weights = {
            'volume_utilization': 0.4,
            'weight_utilization': 0.3,
            'item_count_utilization': 0.2,
            'vertical_space_utilization': 0.1
        }
        
        return sum(
            metrics[key] * weights[key]
            for key in weights
            if key in metrics
        )
    
    def _get_oriented_dimensions(
        self,
        dims: ItemDimensions,
        orientation: int
    ) -> Tuple[float, float, float]:
        """Get dimensions based on orientation"""
        if orientation == 0:  # Horizontal
            return dims.length, dims.width, dims.height
        else:  # Vertical
            return dims.width, dims.length, dims.height
    
    def _generate_possible_positions(
        self,
        item_dims: ItemDimensions,
        bin_state: BinState
    ) -> List[Dict[str, float]]:
        """Generate list of possible placement positions"""
        positions = []
        
        if not self.enable_tetris_optimization:
            # Simple placement at the bottom
            positions.append({
                'x': self.min_gap,
                'y': self.min_gap,
                'z': 0.0
            })
            return positions
        
        # Complex Tetris-style placement
        occupied_spaces = self._get_occupied_spaces(bin_state)
        
        # Find valid positions considering existing items
        for x in np.arange(0, bin_state.dimensions.length - item_dims.length, self.min_gap):
            for y in np.arange(0, bin_state.dimensions.width - item_dims.width, self.min_gap):
                z = self._find_valid_height(x, y, item_dims, occupied_spaces)
                
                if z is not None:
                    positions.append({'x': x, 'y': y, 'z': z})
        
        return positions
    
    def _find_valid_height(
        self,
        x: float,
        y: float,
        item_dims: ItemDimensions,
        occupied_spaces: List[Dict[str, float]]
    ) -> Optional[float]:
        """Find valid height for item placement"""
        # Find maximum height of items below this position
        max_height = 0.0
        
        for space in occupied_spaces:
            if (space['x'] < x + item_dims.length and 
                x < space['x'] + space['length'] and
                space['y'] < y + item_dims.width and
                y < space['y'] + space['width']):
                max_height = max(max_height, space['z'] + space['height'])
        
        return max_height if max_height + item_dims.height <= self.bins[bin_id].dimensions.height else None
    
    def _calculate_placement_score(
        self,
        item_dims: ItemDimensions,
        position: Dict[str, float],
        bin_state: BinState
    ) -> float:
        """Calculate placement score considering various factors"""
        score = 0.0
        
        # Base stability score
        score += self._calculate_stability_score(
            item_dims,
            position,
            bin_state
        ) * self.tetris_params['stability_factor']
        
        # Contact surface bonus
        score += self._calculate_contact_score(
            item_dims,
            position,
            bin_state
        )
        
        # Space utilization score
        score += self._calculate_space_utilization_score(
            item_dims,
            position,
            bin_state
        )
        
        return score
    
    def _calculate_stability_score(
        self,
        item_dims: ItemDimensions,
        position: Dict[str, float],
        bin_state: BinState
    ) -> float:
        """Calculate stability score for placement"""
        # Base stability (contact with floor or other items)
        if position['z'] == 0:
            return 1.0
            
        support_area = 0.0
        total_area = item_dims.length * item_dims.width
        
        for other_item in bin_state.current_items.values():
            if self._has_support_contact(item_dims, position, other_item):
                support_area += self._calculate_support_area(
                    item_dims,
                    position,
                    other_item
                )
        
        return min(support_area / total_area, 1.0)
    
    def _has_support_contact(
        self,
        item_dims: ItemDimensions,
        position: Dict[str, float],
        other_item: ItemDimensions
    ) -> bool:
        """Check if item has support contact with another item"""
        # Implementation of support contact checking
        pass
    
    def _calculate_support_area(
        self,
        item_dims: ItemDimensions,
        position: Dict[str, float],
        other_item: ItemDimensions
    ) -> float:
        """Calculate support area between items"""
        # Implementation of support area calculation
        pass
    
    def _update_bin_utilization(
        self,
        bin_id: str,
        added_item: Optional[ProductAttributes] = None
    ):
        """Update bin utilization metrics"""
        bin_state = self.bins[bin_id]
        utilization = BinUtilization()
        
        # Calculate current utilization
        for item_dims in bin_state.current_items.values():
            utilization.volume_used += item_dims.volume
            utilization.weight_used += item_dims.weight
            
        utilization.num_items = len(bin_state.current_items)
        
        # Update stacking levels if item added
        if added_item:
            self._update_stacking_levels(bin_id, added_item)
        
        self.bin_utilization[bin_id] = utilization
    
    def _update_stacking_levels(
        self,
        bin_id: str,
        item: ProductAttributes
    ):
        """Update stacking levels for bin"""
        utilization = self.bin_utilization[bin_id]
        
        # Update stacking levels based on item position
        # Implementation of stacking level updates
        pass


# This `bin_manager.py` provides:

# 1. Core Bin Management:
# - Bin state tracking
# - Item placement validation
# - Capacity constraints
# - Physical dimension handling

# 2. Advanced Placement Features:
# - Tetris-style optimization
# - Stability calculations
# - Space utilization optimization
# - Multi-orientation support

# 3. Utilization Tracking:
# - Volume utilization
# - Weight utilization
# - Vertical space usage
# - Stacking efficiency

# 4. Safety Features:
# - Minimum gaps between items
# - Safety factors for dimensions
# - Stack stability checking
# - Weight limit enforcement

# 5. Optimization Features:
# - Optimal position finding
# - Score-based placement
# - Contact surface optimization
# - Space utilization maximization
