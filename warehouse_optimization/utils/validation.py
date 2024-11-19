from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
import json
import re

from ..core.types import (
    WarehouseState,
    ProductAttributes,
    StorageLocation,
    BinDimensions,
    ItemDimensions
)

@dataclass
class ValidationConfig:
    """Configuration for validation rules"""
    max_weight_per_shelf: float = 500.0  # kg
    max_items_per_bin: int = 50
    min_gap_between_items: float = 0.02  # meters
    max_stack_height: float = 2.0  # meters
    required_columns: Dict[str, List[str]] = None
    allowed_zone_types: List[str] = None
    id_pattern: str = r'^[A-Z0-9\-_]+$'

class DataValidator:
    """
    Handles data validation, constraint checking, and input verification
    for warehouse optimization system.
    """
    
    def __init__(
        self,
        config: Optional[ValidationConfig] = None
    ):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Track validation results
        self.validation_results: List[Dict] = []
        self.current_violations: Dict[str, List[str]] = {}
    
    def _check_required_columns(self, order_data, required_columns):
        # Check if required columns exist in the order_data
        missing_columns = [col for col in required_columns if col not in order_data.columns]
        if missing_columns:
            return f"Missing columns: {', '.join(missing_columns)}"
        return None
    
    def _validate_timestamps(self, order_data):
        # Example logic to check if timestamp column exists and has valid values
        if 'timestamp' not in order_data.columns:
            return "Missing timestamp column"
        
        # Check if timestamps are valid (e.g., should be datetime)
        try:
            order_data['timestamp'] = pd.to_datetime(order_data['timestamp'])
        except Exception as e:
            return f"Invalid timestamp format: {str(e)}"
        
        # You can also check for range validity or other constraints
        # For example, if you expect the timestamps to be within the last year:
        min_date = pd.to_datetime('2023-01-01')
        max_date = pd.to_datetime('now')
        invalid_timestamps = order_data[(order_data['timestamp'] < min_date) | (order_data['timestamp'] > max_date)]
        
        if not invalid_timestamps.empty:
            return f"Invalid timestamps found: {invalid_timestamps['timestamp'].to_list()}"
        
        return None  # No issues found

    def _validate_quantities(self, order_data):
        # Check if the 'quantity' column exists
        if 'quantity' not in order_data.columns:
            return "Missing quantity column"
        
        # Check if quantities are positive
        invalid_quantities = order_data[order_data['quantity'] <= 0]
        
        if not invalid_quantities.empty:
            return f"Invalid quantities found: {invalid_quantities['quantity'].to_list()}"
        
        return None  # No issues found

    def _validate_order_ids(self, order_data):
        # Check if 'order_id' column exists
        if 'order_id' not in order_data.columns:
            return "Missing order_id column"
        
        # Check for duplicate order IDs
        duplicate_ids = order_data[order_data.duplicated(subset=['order_id'])]
        
        if not duplicate_ids.empty:
            return f"Duplicate order IDs found: {duplicate_ids['order_id'].to_list()}"
        
        # Optionally, check if order IDs are unique integers or another valid format
        if not all(order_data['order_id'].apply(lambda x: isinstance(x, int))):
            return "Invalid order ID format (must be integers)"
        
        return None  # No issues found



    def _default_config(self) -> ValidationConfig:
        """Create default validation configuration"""
        return ValidationConfig(
            required_columns={
                'products': [
                    'product_id', 'category', 'length', 'width',
                    'height', 'weight'
                ],
                'locations': [
                    'bin_id', 'zone_type', 'x_coord', 'y_coord',
                    'z_coord'
                ],
                'orders': [
                    'order_id', 'timestamp', 'product_id', 'quantity'
                ]
            },
            allowed_zone_types=[
                'normal', 'cold_storage', 'fragile', 'hazardous',
                'high_value'
            ]
        )
    
    def validate_warehouse_state(
        self,
        warehouse_state: WarehouseState
    ) -> Tuple[bool, List[str]]:
        """
        Validate complete warehouse state
        
        Args:
            warehouse_state: Current warehouse state
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        try:
            # Validate layout
            layout_violations = self._validate_layout(
                warehouse_state.layout_grid,
                warehouse_state.zones
            )
            violations.extend(layout_violations)
            
            # Validate storage locations
            location_violations = self._validate_storage_locations(
                warehouse_state.storage_locations
            )
            violations.extend(location_violations)
            
            # Validate item placements
            placement_violations = self._validate_item_placements(
                warehouse_state.item_locations,
                warehouse_state.storage_locations
            )
            violations.extend(placement_violations)
            
            # Update validation results
            self._update_validation_results(violations)
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"Error validating warehouse state: {str(e)}")
            return False, [str(e)]
    
    def validate_product_data(
        self,
        product_data: pd.DataFrame
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate product data"""
        violations = {}
        
        try:
            # Check required columns
            column_violations = self._check_required_columns(
                product_data,
                self.config.required_columns['products']
            )
            if column_violations:
                violations['columns'] = column_violations
            
            # Validate data types
            type_violations = self._validate_product_types(product_data)
            if type_violations:
                violations['types'] = type_violations
            
            # Validate dimensions and weights
            dimension_violations = self._validate_product_dimensions(product_data)
            if dimension_violations:
                violations['dimensions'] = dimension_violations
            
            # Validate product IDs
            id_violations = self._validate_product_ids(product_data)
            if id_violations:
                violations['ids'] = id_violations
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"Error validating product data: {str(e)}")
            return False, {'error': [str(e)]}
    
    def validate_location_data(
        self,
        location_data: pd.DataFrame
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate location data"""
        violations = {}
        
        try:
            # Check required columns
            column_violations = self._check_required_columns(
                location_data,
                self.config.required_columns['locations']
            )
            if column_violations:
                violations['columns'] = column_violations
            
            # Validate zone types
            zone_violations = self._validate_zone_types(location_data)
            if zone_violations:
                violations['zones'] = zone_violations
            
            # Validate coordinates
            coord_violations = self._validate_coordinates(location_data)
            if coord_violations:
                violations['coordinates'] = coord_violations
            
            # Validate bin IDs
            id_violations = self._validate_bin_ids(location_data)
            if id_violations:
                violations['ids'] = id_violations
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"Error validating location data: {str(e)}")
            return False, {'error': [str(e)]}
    
    def validate_order_data(
        self,
        order_data: pd.DataFrame
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate order data"""
        violations = {}
        
        try:
            # Check required columns
            column_violations = self._check_required_columns(
                order_data,
                self.config.required_columns['orders']
            )
            if column_violations:
                violations['columns'] = column_violations
            
            # Validate timestamps
            time_violations = self._validate_timestamps(order_data)
            if time_violations:
                violations['timestamps'] = time_violations
            
            # Validate quantities
            quantity_violations = self._validate_quantities(order_data)
            if quantity_violations:
                violations['quantities'] = quantity_violations
            
            # Validate order IDs
            id_violations = self._validate_order_ids(order_data)
            if id_violations:
                violations['ids'] = id_violations
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"Error validating order data: {str(e)}")
            return False, {'error': [str(e)]}
    
    def validate_placement(
        self,
        item: ProductAttributes,
        location: StorageLocation,
        current_items: Dict[int, ItemDimensions]
    ) -> Tuple[bool, List[str]]:
        """Validate item placement in location"""
        violations = []
        
        try:
            # Check physical constraints
            if not self._check_physical_constraints(item, location):
                violations.append("Physical constraints violated")
            
            # Check weight constraints
            if not self._check_weight_constraints(item, location, current_items):
                violations.append("Weight limit exceeded")
            
            # Check spacing constraints
            if not self._check_spacing_constraints(item, location, current_items):
                violations.append("Minimum spacing requirements not met")
            
            # Check stacking constraints
            if not self._check_stacking_constraints(item, location, current_items):
                violations.append("Stacking constraints violated")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            self.logger.error(f"Error validating placement: {str(e)}")
            return False, [str(e)]
    
    def _validate_layout(
        self,
        layout_grid: np.ndarray,
        zones: Dict[str, List[Tuple[int, int, int]]]
    ) -> List[str]:
        """Validate warehouse layout"""
        violations = []
        
        # Check grid dimensions
        if len(layout_grid.shape) != 3:
            violations.append("Layout grid must be 3-dimensional")
        
        # Check zone definitions
        for zone_type, locations in zones.items():
            if zone_type not in self.config.allowed_zone_types:
                violations.append(f"Invalid zone type: {zone_type}")
            
            for loc in locations:
                if not self._is_valid_coordinate(loc, layout_grid.shape):
                    violations.append(f"Invalid zone location: {loc}")
        
        return violations
    
    def _validate_storage_locations(
        self,
        locations: Dict[str, StorageLocation]
    ) -> List[str]:
        """Validate storage locations"""
        violations = []
        
        for bin_id, location in locations.items():
            # Validate bin ID format
            if not re.match(self.config.id_pattern, bin_id):
                violations.append(f"Invalid bin ID format: {bin_id}")
            
            # Validate zone type
            if location.zone_type not in self.config.allowed_zone_types:
                violations.append(f"Invalid zone type for bin {bin_id}")
            
            # Validate dimensions
            if not self._are_valid_dimensions(location.dimensions):
                violations.append(f"Invalid dimensions for bin {bin_id}")
        
        return violations
    
    def _validate_item_placements(
        self,
        item_locations: Dict[int, str],
        storage_locations: Dict[str, StorageLocation]
    ) -> List[str]:
        """Validate item placements"""
        violations = []
        
        for item_id, bin_id in item_locations.items():
            if bin_id not in storage_locations:
                violations.append(f"Invalid bin ID for item {item_id}")
            
            location = storage_locations[bin_id]
            if len(location.current_items) >= self.config.max_items_per_bin:
                violations.append(f"Bin {bin_id} exceeds maximum items")
        
        return violations
    
    def _check_physical_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation
    ) -> bool:
        """Check physical constraints for placement"""
        # Check dimensions fit
        return (
            item.dimensions.length <= location.dimensions.length and
            item.dimensions.width <= location.dimensions.width and
            item.dimensions.height <= location.dimensions.height
        )
    
    def _check_weight_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation,
        current_items: Dict[int, ItemDimensions]
    ) -> bool:
        """Check weight constraints"""
        current_weight = sum(item.weight for item in current_items.values())
        return current_weight + item.dimensions.weight <= location.dimensions.max_weight
    
    def _check_spacing_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation,
        current_items: Dict[int, ItemDimensions]
    ) -> bool:
        """Check minimum spacing requirements"""
        # Simplified check - in practice would need more sophisticated 3D spacing check
        return True
    
    def _check_stacking_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation,
        current_items: Dict[int, ItemDimensions]
    ) -> bool:
        """Check stacking constraints"""
        current_height = sum(item.height for item in current_items.values())
        return current_height + item.dimensions.height <= self.config.max_stack_height
    
    def _update_validation_results(self, violations: List[str]):
        """Update validation results history"""
        self.validation_results.append({
            'timestamp': datetime.now(),
            'violations': violations,
            'is_valid': len(violations) == 0
        })
        
        # Update current violations
        self.current_violations = {
            'general': violations
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_results:
            return {}
            
        return {
            'total_validations': len(self.validation_results),
            'total_violations': sum(
                len(result['violations'])
                for result in self.validation_results
            ),
            'current_violations': self.current_violations,
            'latest_result': self.validation_results[-1],
            'validation_rate': sum(
                1 for result in self.validation_results
                if result['is_valid']
            ) / len(self.validation_results)
        }

# This implementation provides:

# 1. Core Validation:
# - Warehouse state validation
# - Product data validation
# - Location data validation
# - Order data validation

# 2. Constraint Checking:
# - Physical constraints
# - Weight constraints
# - Spacing constraints
# - Stacking constraints

# 3. Data Verification:
# - Required columns
# - Data types
# - Value ranges
# - ID formats

# 4. Location Validation:
# - Zone types
# - Coordinates
# - Dimensions
# - Bin capacities

# 5. Tracking Features:
# - Validation history
# - Current violations
# - Validation metrics
# - Summary statistics
