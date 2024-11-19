import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from gymnasium import spaces
from dataclasses import asdict

from .types import (
    WarehouseState, 
    OptimizationState, 
    ProductAttributes, 
    StorageLocation,
    BinState, 
    ActionSpace, 
    ObservationSpace,
    LocationCoord,
    ItemDimensions
)

from ..managers.bin_manager import BinManager
from ..managers.constraint_manager import ConstraintManager
from ..managers.affinity_manager import AffinityManager
from ..managers.state_manager import StateManager
from ..forecasting.demand_forecaster import DemandForecaster

class WarehouseEnvironment(gym.Env):
    """
    Warehouse optimization environment using OpenAI Gym interface.
    Handles the RL environment for optimizing warehouse item placement.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        bin_manager: BinManager,
        constraint_manager: ConstraintManager,
        affinity_manager: AffinityManager,
        forecaster: DemandForecaster,
        optimization_window: int = 7,
        safety_factor: float = 0.85
    ):
        super().__init__()
        
        self.state_manager = state_manager
        self.bin_manager = bin_manager
        self.constraint_manager = constraint_manager
        self.affinity_manager = affinity_manager
        self.forecaster = forecaster
        self.optimization_window = optimization_window
        self.safety_factor = safety_factor
        
        # Initialize warehouse state
        self.warehouse_state = self.state_manager.get_current_state()
        
        # Set up action and observation spaces
        self._setup_spaces()
        
        # Current episode state
        self.current_item: Optional[ProductAttributes] = None
        self.current_step = 0
        self.max_steps = 100  # Maximum steps per episode
        
    def _setup_spaces(self):
        """Initialize action and observation spaces"""
        # Action space: (bin_id, orientation)
        num_bins = len(self.warehouse_state.storage_locations)
        num_orientations = 2  # horizontal/vertical
        
        self.action_space = spaces.MultiDiscrete([
            num_bins,  # Bin selection
            num_orientations  # Orientation selection
        ])
        
        # Observation space components
        layout_shape = self.warehouse_state.layout_grid.shape
        num_products = len(self.state_manager.get_product_ids())
        num_zones = len(self.warehouse_state.zones)
        num_features = 50  # Base features + forecast features
        
        obs_space = ObservationSpace(
            layout_shape=layout_shape,
            num_products=num_products,
            num_zones=num_zones,
            num_features=num_features
        )
        
        # Create observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_space.get_observation_shape(),
            dtype=np.float32
        )
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        
        # Get new item to place if not provided in options
        if options and 'item' in options:
            self.current_item = options['item']
        else:
            self.current_item = self.state_manager.get_random_item()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'item_id': self.current_item.product_id,
            'valid_locations': self._get_valid_locations(),
            'current_utilization': self.state_manager.get_utilization_metrics()
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute environment step.
        Args:
            action: [bin_id_index, orientation]
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Convert action to bin_id and orientation
        bin_idx, orientation = action
        bin_id = list(self.warehouse_state.storage_locations.keys())[bin_idx]
        
        # Get bin and validate placement
        bin_state = self.warehouse_state.storage_locations[bin_id]
        can_place, violations = self._validate_placement(
            self.current_item,
            bin_state,
            orientation
        )
        
        # Calculate reward and update state if placement is valid
        if can_place:
            # Update state
            self._update_state(bin_id, self.current_item, orientation)
            
            # Calculate comprehensive reward
            reward = self._calculate_reward(bin_id, self.current_item)
        else:
            # Penalty for invalid placement
            reward = -100.0
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode should end
        terminated = not can_place or self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'violations': violations if not can_place else [],
            'placement_bin': bin_id if can_place else None,
            'reward_components': self._get_reward_components(bin_id, self.current_item),
            'utilization': self.state_manager.get_utilization_metrics()
        }
        
        return observation, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation array"""
        # Get warehouse state features
        layout_features = self.state_manager.get_layout_features()
        
        # Get item features
        item_features = self._get_item_features(self.current_item)
        
        # Get forecast features
        forecast_features = self._get_forecast_features(self.current_item)
        
        # Combine all features
        observation = np.concatenate([
            layout_features.flatten(),
            item_features,
            forecast_features
        ])
        
        return observation.astype(np.float32)
    
    def _validate_placement(
        self,
        item: ProductAttributes,
        bin_state: StorageLocation,
        orientation: int
    ) -> Tuple[bool, List[str]]:
        """Validate if item can be placed in the given bin"""
        # Check physical constraints
        can_fit, violations = self.bin_manager.can_fit_item(
            item.dimensions,
            bin_state.dimensions,
            orientation
        )
        
        if not can_fit:
            return False, violations
        
        # Check zone compatibility
        if not self.constraint_manager.check_zone_compatibility(
            item,
            bin_state.zone_type
        ):
            violations.append("Zone incompatibility")
            return False, violations
        
        # Check item compatibility with existing items
        if not self.constraint_manager.check_item_compatibility(
            item,
            bin_state.current_items
        ):
            violations.append("Item incompatibility")
            return False, violations
        
        return True, []
    
    def _calculate_reward(
        self,
        bin_id: str,
        item: ProductAttributes
    ) -> float:
        """Calculate comprehensive reward for placement"""
        reward_components = self._get_reward_components(bin_id, item)
        
        # Combine components with weights
        weights = {
            'distance': 0.3,
            'affinity': 0.2,
            'demand': 0.2,
            'utilization': 0.15,
            'picking_efficiency': 0.15
        }
        
        total_reward = sum(
            component * weights[name]
            for name, component in reward_components.items()
        )
        
        return total_reward
    
    def _get_reward_components(
        self,
        bin_id: str,
        item: ProductAttributes
    ) -> Dict[str, float]:
        """Calculate individual reward components"""
        bin_state = self.warehouse_state.storage_locations[bin_id]
        
        # Distance-based component
        distance_score = -bin_state.distance_to_pickup
        
        # Affinity-based component
        affinity_score = self.affinity_manager.calculate_affinity_score(
            item.product_id,
            bin_id
        )
        
        # Demand-based component
        demand_score = self.forecaster.get_demand_score(
            item.product_id,
            self.optimization_window
        )
        
        # Utilization component
        utilization_score = self.bin_manager.calculate_utilization_score(bin_id)
        
        # Picking efficiency component
        picking_score = self._calculate_picking_efficiency(bin_id, item)
        
        return {
            'distance': distance_score,
            'affinity': affinity_score,
            'demand': demand_score,
            'utilization': utilization_score,
            'picking_efficiency': picking_score
        }
    
    def _calculate_picking_efficiency(
        self,
        bin_id: str,
        item: ProductAttributes
    ) -> float:
        """Calculate picking efficiency score"""
        bin_state = self.warehouse_state.storage_locations[bin_id]
        
        # Base efficiency based on distance
        base_efficiency = 1.0 / (1.0 + bin_state.distance_to_pickup)
        
        # Penalty for requiring ladder
        if bin_state.requires_ladder:
            base_efficiency *= 0.7
        
        # Penalty for heavy items on high shelves
        if (
            bin_state.requires_ladder and 
            item.dimensions.weight > 5.0
        ):
            base_efficiency *= 0.5
        
        return base_efficiency
    
    def _update_state(
        self,
        bin_id: str,
        item: ProductAttributes,
        orientation: int
    ):
        """Update environment state after successful placement"""
        # Update bin state
        self.bin_manager.add_item_to_bin(bin_id, item, orientation)
        
        # Update warehouse state
        self.state_manager.update_item_location(
            item.product_id,
            bin_id
        )
        
        # Update current state
        self.warehouse_state = self.state_manager.get_current_state()
    
    def _get_valid_locations(self) -> List[str]:
        """Get list of valid bin locations for current item"""
        valid_locations = []
        
        for bin_id, bin_state in self.warehouse_state.storage_locations.items():
            can_place, _ = self._validate_placement(
                self.current_item,
                bin_state,
                0  # Check horizontal orientation
            )
            
            if can_place:
                valid_locations.append(bin_id)
        
        return valid_locations
    
    def render(self):
        """Render the environment (not implemented)"""
        pass

    def close(self):
        """Clean up environment resources"""
        pass

# This `environment.py` file provides:

# 1. Core RL Environment:
# - Implements the OpenAI Gym interface
# - Handles state management and transitions
# - Manages action and observation spaces
# - Calculates rewards

# 2. Key Features:
# - Comprehensive reward function with multiple components
# - Detailed state validation
# - Efficient state updates
# - Flexible observation space

# 3. Validation and Constraints:
# - Physical constraints checking
# - Zone compatibility
# - Item compatibility
# - Bin capacity management

# 4. Reward Components:
# - Distance-based scoring
# - Affinity-based scoring
# - Demand-based scoring
# - Utilization scoring
# - Picking efficiency scoring

# 5. State Management:
# - Current warehouse state tracking
# - Item placement tracking
# - Bin state updates
# - Valid location identification