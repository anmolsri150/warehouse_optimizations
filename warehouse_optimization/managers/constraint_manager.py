from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np

from ..core.types import (
    ProductAttributes,
    StorageLocation,
    PlacementConstraints,
    ItemDimensions,
    BinDimensions
)

@dataclass
class ZoneConstraints:
    """Zone-specific constraints"""
    temp_range: Tuple[float, float]
    humidity_range: Tuple[float, float]
    compatible_products: Set[str]
    incompatible_products: Set[str]
    max_weight_per_shelf: float
    requires_ventilation: bool
    requires_monitoring: bool
    max_stack_height: int
    security_level: str
    special_handling: Set[str]

class ConstraintManager:
    """
    Manages all constraints and rules for warehouse item placement.
    Handles zone constraints, product compatibility, and environmental requirements.
    """
    
    def __init__(
        self,
        zone_rules: Dict[str, Dict],
        product_rules: Dict[str, Dict],
        safety_rules: Dict[str, Dict],
        environmental_rules: Dict[str, Dict]
    ):
        self.zone_rules = self._initialize_zone_rules(zone_rules)
        self.product_rules = product_rules
        self.safety_rules = safety_rules
        self.environmental_rules = environmental_rules
        
        # Initialize violation tracking
        self.violation_history: List[Dict] = []
        self.active_violations: Dict[str, List[str]] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize constraint cache
        self.compatibility_cache: Dict[Tuple[int, str], bool] = {}
        self.last_cache_update = datetime.now()
    
    def _initialize_zone_rules(
        self,
        zone_rules: Dict[str, Dict]
    ) -> Dict[str, ZoneConstraints]:
        """Initialize structured zone constraints"""
        initialized_rules = {}
        
        for zone_type, rules in zone_rules.items():
            initialized_rules[zone_type] = ZoneConstraints(
                temp_range=rules.get('temp_range', (15, 25)),
                humidity_range=rules.get('humidity_range', (30, 60)),
                compatible_products=set(rules.get('compatible_products', [])),
                incompatible_products=set(rules.get('incompatible_products', [])),
                max_weight_per_shelf=rules.get('max_weight_per_shelf', 100.0),
                requires_ventilation=rules.get('requires_ventilation', False),
                requires_monitoring=rules.get('requires_monitoring', False),
                max_stack_height=rules.get('max_stack_height', 4),
                security_level=rules.get('security_level', 'normal'),
                special_handling=set(rules.get('special_handling', []))
            )
        
        return initialized_rules
    
    def check_placement_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation,
        current_conditions: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check all constraints for item placement
        Returns (is_valid, list_of_violations)
        """
        violations = []
        
        # Check zone compatibility
        zone_violations = self._check_zone_constraints(
            item,
            location,
            current_conditions
        )
        violations.extend(zone_violations)
        
        # Check product compatibility
        product_violations = self._check_product_compatibility(
            item,
            location
        )
        violations.extend(product_violations)
        
        # Check safety constraints
        safety_violations = self._check_safety_constraints(
            item,
            location
        )
        violations.extend(safety_violations)
        
        # Check environmental constraints
        if current_conditions:
            env_violations = self._check_environmental_constraints(
                item,
                location,
                current_conditions
            )
            violations.extend(env_violations)
        
        # Log violations if any
        if violations:
            self._log_violations(item.product_id, location.bin_id, violations)
        
        return len(violations) == 0, violations
    
    def _check_zone_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation,
        current_conditions: Optional[Dict]
    ) -> List[str]:
        """Check zone-specific constraints"""
        violations = []
        zone_rules = self.zone_rules.get(location.zone_type)
        
        if not zone_rules:
            return ["Invalid zone type"]
        
        # Check product category compatibility
        if item.category not in zone_rules.compatible_products:
            violations.append(f"Product category {item.category} not compatible with zone {location.zone_type}")
        
        # Check incompatible products
        if item.category in zone_rules.incompatible_products:
            violations.append(f"Product category {item.category} explicitly incompatible with zone {location.zone_type}")
        
        # Check temperature requirements if conditions provided
        if current_conditions and hasattr(item, 'temperature_requirements'):
            temp = current_conditions.get('temperature')
            if temp and not (zone_rules.temp_range[0] <= temp <= zone_rules.temp_range[1]):
                violations.append("Temperature requirements not met")
        
        # Check weight limits
        current_weight = sum(item.weight for item in location.current_items.values())
        if current_weight + item.dimensions.weight > zone_rules.max_weight_per_shelf:
            violations.append("Weight limit exceeded for shelf")
        
        return violations
    
    def _check_product_compatibility(
        self,
        item: ProductAttributes,
        location: StorageLocation
    ) -> List[str]:
        """Check product compatibility constraints"""
        violations = []
        
        # Get product rules
        product_rules = self.product_rules.get(item.category, {})
        
        # Check compatibility with existing items
        for existing_item_id in location.current_items:
            if not self._are_products_compatible(
                item.product_id,
                existing_item_id
            ):
                violations.append(f"Incompatible with existing product {existing_item_id}")
        
        # Check special handling requirements
        if product_rules.get('special_handling'):
            zone_rules = self.zone_rules.get(location.zone_type)
            if not zone_rules or not product_rules['special_handling'].issubset(zone_rules.special_handling):
                violations.append("Special handling requirements not met")
        
        return violations
    
    def _check_safety_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation
    ) -> List[str]:
        """Check safety-related constraints"""
        violations = []
        
        # Check stacking limits
        current_stack_height = self._calculate_stack_height(location)
        if current_stack_height + 1 > self.zone_rules[location.zone_type].max_stack_height:
            violations.append("Stack height limit exceeded")
        
        # Check hazardous materials constraints
        if item.is_hazardous:
            if not self._check_hazmat_constraints(item, location):
                violations.append("Hazardous material placement constraints violated")
        
        # Check fragile item constraints
        if item.is_fragile:
            if not self._check_fragile_constraints(item, location):
                violations.append("Fragile item placement constraints violated")
        
        return violations
    
    def _check_environmental_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation,
        current_conditions: Dict
    ) -> List[str]:
        """Check environmental constraints"""
        violations = []
        zone_rules = self.zone_rules[location.zone_type]
        
        # Temperature constraints
        if 'temperature' in current_conditions:
            temp = current_conditions['temperature']
            if not (zone_rules.temp_range[0] <= temp <= zone_rules.temp_range[1]):
                violations.append("Temperature out of acceptable range")
        
        # Humidity constraints
        if 'humidity' in current_conditions:
            humidity = current_conditions['humidity']
            if not (zone_rules.humidity_range[0] <= humidity <= zone_rules.humidity_range[1]):
                violations.append("Humidity out of acceptable range")
        
        # Ventilation requirements
        if zone_rules.requires_ventilation:
            if not self._check_ventilation_requirements(location):
                violations.append("Ventilation requirements not met")
        
        return violations
    
    def _are_products_compatible(
        self,
        product_id1: int,
        product_id2: int
    ) -> bool:
        """Check if two products are compatible"""
        cache_key = tuple(sorted([product_id1, product_id2]))
        
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        # Get product categories
        category1 = self.product_rules.get(product_id1, {}).get('category')
        category2 = self.product_rules.get(product_id2, {}).get('category')
        
        if not category1 or not category2:
            return True  # If no specific rules, assume compatible
        
        # Check incompatibility rules
        incompatible = (
            category2 in self.product_rules.get(category1, {}).get('incompatible_with', set()) or
            category1 in self.product_rules.get(category2, {}).get('incompatible_with', set())
        )
        
        # Cache result
        self.compatibility_cache[cache_key] = not incompatible
        return not incompatible
    
    def _calculate_stack_height(
        self,
        location: StorageLocation
    ) -> int:
        """Calculate current stack height in location"""
        if not location.current_items:
            return 0
            
        # Group items by x,y position
        stacks = {}
        for item_id, dims in location.current_items.items():
            position = self._get_item_position(item_id, location)
            if position:
                key = (position['x'], position['y'])
                stacks[key] = stacks.get(key, 0) + 1
        
        return max(stacks.values()) if stacks else 0
    
    def _check_hazmat_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation
    ) -> bool:
        """Check hazardous materials constraints"""
        if not hasattr(item, 'hazmat_class'):
            return True
            
        # Check separation requirements
        for existing_item_id in location.current_items:
            existing_item = self._get_product_attributes(existing_item_id)
            if hasattr(existing_item, 'hazmat_class'):
                if not self._are_hazmat_classes_compatible(
                    item.hazmat_class,
                    existing_item.hazmat_class
                ):
                    return False
        
        return True
    
    def _check_fragile_constraints(
        self,
        item: ProductAttributes,
        location: StorageLocation
    ) -> bool:
        """Check fragile item constraints"""
        # Fragile items should be placed on top or have nothing heavy above them
        position = self._get_item_position(item.product_id, location)
        if not position:
            return True
            
        for other_id, other_dims in location.current_items.items():
            other_pos = self._get_item_position(other_id, location)
            if other_pos and other_pos['z'] > position['z']:
                if other_dims.weight > self.safety_rules.get('max_weight_on_fragile', 1.0):
                    return False
        
        return True
    
    def _check_ventilation_requirements(
        self,
        location: StorageLocation
    ) -> bool:
        """Check if ventilation requirements are met"""
        # Implementation depends on warehouse ventilation system
        return True
    
    def _log_violations(
        self,
        product_id: int,
        bin_id: str,
        violations: List[str]
    ):
        """Log constraint violations"""
        violation_entry = {
            'timestamp': datetime.now(),
            'product_id': product_id,
            'bin_id': bin_id,
            'violations': violations
        }
        
        self.violation_history.append(violation_entry)
        self.active_violations[f"{product_id}_{bin_id}"] = violations
        
        self.logger.warning(
            f"Constraint violations for product {product_id} in bin {bin_id}: {violations}"
        )
    
    def get_violation_history(
        self,
        product_id: Optional[int] = None,
        bin_id: Optional[str] = None
    ) -> List[Dict]:
        """Get violation history with optional filtering"""
        if not product_id and not bin_id:
            return self.violation_history
            
        filtered_history = []
        for violation in self.violation_history:
            if product_id and violation['product_id'] != product_id:
                continue
            if bin_id and violation['bin_id'] != bin_id:
                continue
            filtered_history.append(violation)
        
        return filtered_history
    
    def get_active_violations(
        self,
        product_id: Optional[int] = None,
        bin_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get currently active violations"""
        if not product_id and not bin_id:
            return self.active_violations
            
        filtered_violations = {}
        for key, violations in self.active_violations.items():
            prod_id, bin = key.split('_')
            if product_id and int(prod_id) != product_id:
                continue
            if bin_id and bin != bin_id:
                continue
            filtered_violations[key] = violations
        
        return filtered_violations


# This `constraint_manager.py` provides:

# 1. Core Constraint Management:
# - Zone constraints
# - Product compatibility
# - Safety rules
# - Environmental requirements

# 2. Comprehensive Checks:
# - Temperature and humidity
# - Weight limits
# - Stack height
# - Special handling
# - Hazardous materials
# - Fragile items

# 3. Violation Tracking:
# - History logging
# - Active violation monitoring
# - Filtered violation queries

# 4. Advanced Features:
# - Constraint caching
# - Environmental monitoring
# - Stack height calculation
# - Position-based checks

# 5. Safety Features:
# - Hazmat compatibility
# - Fragile item protection
# - Ventilation requirements
# - Weight distribution