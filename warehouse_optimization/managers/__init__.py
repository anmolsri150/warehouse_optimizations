"""
Warehouse Optimization Managers Module
Provides core management components for warehouse optimization system.
"""

from .affinity_manager import AffinityManager
from .bin_manager import (
    BinManager,
    BinUtilization
)
from .constraint_manager import (
    ConstraintManager,
    ZoneConstraints
)
from .state_manager import (
    StateManager,
    ZoneState,
    WarehouseMetrics
)

# Version information
__version__ = "0.1.0"

# Module level docstring
__doc__ = """
Warehouse Optimization Managers Module
====================================

This module provides the core management components for warehouse optimization:

1. Affinity Manager:
   - Handles product co-occurrence relationships
   - Calculates placement affinities
   - Optimizes product grouping
   - Manages temporal patterns

2. Bin Manager:
   - Manages physical bin constraints
   - Handles item placement optimization
   - Tracks bin utilization
   - Implements space optimization algorithms

3. Constraint Manager:
   - Enforces warehouse rules and constraints
   - Manages zone-specific requirements
   - Handles product compatibility
   - Ensures safety requirements

4. State Manager:
   - Maintains warehouse state
   - Tracks zone and location states
   - Manages state transitions
   - Provides optimization metrics

Usage Example:
-------------
from warehouse_optimization.managers import (
    AffinityManager,
    BinManager,
    ConstraintManager,
    StateManager
)

# Initialize managers
affinity_manager = AffinityManager(
    order_history=order_data,
    storage_locations=storage_locations
)

bin_manager = BinManager(
    safety_factor=0.85,
    min_gap=0.02
)

constraint_manager = ConstraintManager(
    zone_rules=zone_rules,
    product_rules=product_rules,
    safety_rules=safety_rules,
    environmental_rules=env_rules
)

state_manager = StateManager(
    layout_grid=layout,
    zones=zones,
    storage_locations=storage_locations
)

# Use managers in optimization
state = state_manager.get_current_state()
valid_locations = bin_manager.get_valid_locations(item)
affinity_score = affinity_manager.calculate_affinity_score(item_id, bin_id)
constraints_satisfied = constraint_manager.check_placement_constraints(item, location)
"""

# List of public objects exported by this module
__all__ = [
    # Main Classes
    'AffinityManager',
    'BinManager',
    'ConstraintManager',
    'StateManager',
    
    # Supporting Classes
    'BinUtilization',
    'ZoneConstraints',
    'ZoneState',
    'WarehouseMetrics'
]

# Module metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__status__ = "Development"

# Optional: Runtime initialization code
def initialize():
    """Initialize the managers module components."""
    import logging
    
    # Setup basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing warehouse_optimization.managers v{__version__}")

# Run initialization when module is imported
initialize()

# Define common manager configurations
DEFAULT_BIN_MANAGER_CONFIG = {
    'safety_factor': 0.85,
    'min_gap': 0.02,
    'max_stack_height_factor': 0.9,
    'enable_tetris_optimization': True
}

DEFAULT_CONSTRAINT_MANAGER_CONFIG = {
    'zone_rules': {
        'normal': {
            'temp_range': (15, 25),
            'humidity_range': (30, 60),
            'max_weight_per_shelf': 100.0,
            'requires_ventilation': False
        },
        'cold_storage': {
            'temp_range': (-5, 5),
            'humidity_range': (85, 95),
            'max_weight_per_shelf': 80.0,
            'requires_ventilation': True
        },
        'fragile': {
            'max_stack_height': 1,
            'vibration_limit': 0.5,
            'max_weight_per_shelf': 50.0,
            'requires_monitoring': True
        }
    },
    'product_rules': {
        'electronics': {
            'max_stack': 3,
            'requires_stable_temp': True
        },
        'food': {
            'requires_temperature_control': True,
            'shelf_life_tracking': True
        },
        'fragile': {
            'max_stack': 1,
            'requires_careful_handling': True
        }
    },
    'safety_rules': {
        'max_weight_on_fragile': 1.0,
        'min_separation_hazardous': 2.0,
        'max_stack_height': 4
    }
}

DEFAULT_AFFINITY_MANAGER_CONFIG = {
    'affinity_window_days': 90,
    'min_support': 0.01,
    'time_decay_factor': 0.1,
    'cache_refresh_hours': 24
}

DEFAULT_STATE_MANAGER_CONFIG = {
    'update_frequency': 300,  # seconds
    'history_size': 10000,
    'metric_calculation_interval': 60  # seconds
}

def create_manager_suite(
    layout_grid,
    zones,
    storage_locations,
    order_history,
    custom_configs: dict = None
) -> tuple:
    """
    Create a complete suite of managers with default or custom configurations.
    
    Args:
        layout_grid: Warehouse layout grid
        zones: Zone definitions
        storage_locations: Storage location definitions
        order_history: Historical order data
        custom_configs: Optional custom configurations for managers
    
    Returns:
        Tuple of (state_manager, bin_manager, constraint_manager, affinity_manager)
    """
    configs = {
        'bin_manager': DEFAULT_BIN_MANAGER_CONFIG.copy(),
        'constraint_manager': DEFAULT_CONSTRAINT_MANAGER_CONFIG.copy(),
        'affinity_manager': DEFAULT_AFFINITY_MANAGER_CONFIG.copy(),
        'state_manager': DEFAULT_STATE_MANAGER_CONFIG.copy()
    }
    
    if custom_configs:
        for manager_type, config in custom_configs.items():
            if manager_type in configs:
                configs[manager_type].update(config)
    
    # Create managers with configurations
    state_manager = StateManager(
        layout_grid=layout_grid,
        zones=zones,
        storage_locations=storage_locations,
        update_frequency=configs['state_manager']['update_frequency']
    )
    
    bin_manager = BinManager(**configs['bin_manager'])
    
    constraint_manager = ConstraintManager(
        zone_rules=configs['constraint_manager']['zone_rules'],
        product_rules=configs['constraint_manager']['product_rules'],
        safety_rules=configs['constraint_manager']['safety_rules'],
        environmental_rules={}  # Can be customized if needed
    )
    
    affinity_manager = AffinityManager(
        order_history=order_history,
        storage_locations=storage_locations,
        **configs['affinity_manager']
    )
    
    return (state_manager, bin_manager, constraint_manager, affinity_manager)

# This `__init__.py` provides:

# 1. Module Organization:
# - Clear imports
# - Comprehensive documentation
# - Usage examples
# - Version information

# 2. Default Configurations:
# - Bin manager defaults
# - Constraint manager defaults
# - Affinity manager defaults
# - State manager defaults

# 3. Utility Functions:
# - Manager suite creation
# - Module initialization
# - Configuration management

# 4. Documentation:
# - Detailed module description
# - Usage examples
# - Configuration guidelines
# - Manager interactions

# 5. Features:
# - Easy manager instantiation
# - Default configurations
# - Custom configuration support
# - Complete manager suite creation
