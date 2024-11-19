"""
Core module for warehouse optimization system.
Contains the main components for RL-based optimization.
"""

from .environment import WarehouseEnvironment
from .optimizer import WarehouseOptimizer
from .types import (
    ItemDimensions,
    BinDimensions,
    ProductAttributes,
    StorageLocation,
    WarehouseState,
    OptimizationState,
    OptimizationMetrics,
    PlacementConstraints,
    DemandForecast,
    PickingRoute,
    BinState,
    ActionSpace,
    ObservationSpace
)

# Version information
__version__ = "0.1.0"

# Module level docstring
__doc__ = """
Warehouse Optimization Core Module
================================

This module provides the core components for warehouse optimization:

1. Environment (WarehouseEnvironment):
   - OpenAI Gym compatible environment
   - Handles state management
   - Implements reward calculation
   - Manages constraints

2. Optimizer (WarehouseOptimizer):
   - Implements PPO-based optimization
   - Handles training and evaluation
   - Provides batch optimization
   - Manages model persistence

3. Types:
   - Comprehensive type system for warehouse entities
   - Support for physical constraints
   - State management types
   - Optimization specific types

Usage:
------
from warehouse_optimization.core import WarehouseEnvironment, WarehouseOptimizer

# Create environment
env = WarehouseEnvironment(
    state_manager=state_manager,
    bin_manager=bin_manager,
    constraint_manager=constraint_manager,
    affinity_manager=affinity_manager,
    forecaster=forecaster
)

# Create optimizer
optimizer = WarehouseOptimizer(
    env=env,
    model_config={...}
)

# Train the model
optimizer.train(total_timesteps=50000)

# Optimize placement
result, metrics = optimizer.optimize_placement(item, current_state)
"""

# List of public objects exported by this module
__all__ = [
    # Main classes
    'WarehouseEnvironment',
    'WarehouseOptimizer',
    
    # Types
    'ItemDimensions',
    'BinDimensions',
    'ProductAttributes',
    'StorageLocation',
    'WarehouseState',
    'OptimizationState',
    'OptimizationMetrics',
    'PlacementConstraints',
    'DemandForecast',
    'PickingRoute',
    'BinState',
    'ActionSpace',
    'ObservationSpace',
]

# Module metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__status__ = "Development"

# Optional: Runtime initialization code
def initialize():
    """Initialize the core module components."""
    import logging
    
    # Setup basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing warehouse_optimization.core v{__version__}")

# Run initialization when module is imported
initialize()