"""
Data generator for warehouse optimization system.
Generates realistic test data with configurable parameters and patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json
import logging
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    """Configuration for data generation"""
    num_products: int = 1000
    num_orders: int = 10000
    num_days: int = 90
    num_aisles: int = 10
    shelves_per_aisle: int = 4
    bins_per_shelf: int = 6
    
    # Product distribution
    product_category_dist: Dict[str, float] = None
    price_ranges: Dict[str, Tuple[float, float]] = None
    dimension_ranges: Dict[str, Dict[str, Tuple[float, float]]] = None
    
    # Order patterns
    orders_per_day_range: Tuple[int, int] = (100, 200)
    items_per_order_range: Tuple[int, int] = (1, 5)
    peak_hours: List[int] = None
    weekend_factor: float = 0.7
    
    def __post_init__(self):
        # Set default distributions if not provided
        if self.product_category_dist is None:
            self.product_category_dist = {
                'normal': 0.5,
                'fragile': 0.15,
                'cold_storage': 0.2,
                'hazardous': 0.1,
                'high_value': 0.05
            }
        
        if self.price_ranges is None:
            self.price_ranges = {
                'normal': (10, 100),
                'fragile': (50, 500),
                'cold_storage': (20, 200),
                'hazardous': (30, 300),
                'high_value': (500, 5000)
            }
        
        if self.dimension_ranges is None:
            self.dimension_ranges = {
                'normal': {
                    'length': (0.2, 0.5),
                    'width': (0.2, 0.4),
                    'height': (0.1, 0.3),
                    'weight': (0.5, 10)
                },
                'fragile': {
                    'length': (0.1, 0.4),
                    'width': (0.1, 0.3),
                    'height': (0.1, 0.2),
                    'weight': (0.1, 5)
                }
            }
        
        if self.peak_hours is None:
            self.peak_hours = [10, 11, 14, 15, 16, 19, 20]

class WarehouseDataGenerator:
    """
    Generates realistic test data for warehouse optimization system.
    """
    
    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize random state
        random.seed(42)
        np.random.seed(42)
        
        # Track generated data
        self.products = None
        self.storage_locations = None
        self.orders = None
        self.inventory = None
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset"""
        self.logger.info("Generating complete dataset...")
        
        # Generate data in sequence
        self.products = self._generate_products()
        self.storage_locations = self._generate_storage_locations()
        self.orders = self._generate_orders()
        self.inventory = self._generate_inventory()
        
        return {
            'products': self.products,
            'storage_locations': self.storage_locations,
            'orders': self.orders,
            'inventory': self.inventory
        }
    
    def _generate_products(self) -> pd.DataFrame:
        """Generate product data"""
        self.logger.info(f"Generating {self.config.num_products} products...")
        
        products = []
        
        for pid in range(1, self.config.num_products + 1):
            # Select category based on distribution
            category = np.random.choice(
                list(self.config.product_category_dist.keys()),
                p=list(self.config.product_category_dist.values())
            )
            
            # Get dimension ranges for category
            dim_ranges = self.config.dimension_ranges.get(
                category,
                self.config.dimension_ranges['normal']
            )
            
            # Generate product
            product = {
                'product_id': pid,
                'category': category,
                'length': random.uniform(*dim_ranges['length']),
                'width': random.uniform(*dim_ranges['width']),
                'height': random.uniform(*dim_ranges['height']),
                'weight': random.uniform(*dim_ranges['weight']),
                'price': random.uniform(*self.config.price_ranges[category]),
                'is_fragile': category in ['fragile', 'high_value'],
                'requires_cold_storage': category == 'cold_storage',
                'is_hazardous': category == 'hazardous',
                'shelf_life_days': random.randint(30, 365) if category == 'cold_storage' else None,
                'stackable': category not in ['fragile', 'high_value'],
                'max_stack': random.randint(1, 3) if category not in ['fragile', 'high_value'] else 1
            }
            
            products.append(product)
        
        return pd.DataFrame(products)
    
    def _generate_storage_locations(self) -> pd.DataFrame:
        """Generate storage location data"""
        self.logger.info("Generating storage locations...")
        
        locations = []
        bin_id = 1
        
        for aisle in range(1, self.config.num_aisles + 1):
            # Assign zone type to aisle
            zone_type = self._assign_zone_type(aisle)
            
            for shelf in range(1, self.config.shelves_per_aisle + 1):
                for bin_num in range(1, self.config.bins_per_shelf + 1):
                    # Calculate coordinates
                    x_coord = aisle
                    y_coord = shelf
                    z_coord = bin_num
                    
                    # Calculate distance to pickup (simplified)
                    distance = abs(x_coord - 1) + abs(y_coord - 1) + abs(z_coord - 1)
                    
                    location = {
                        'bin_id': f"A{aisle:02d}-S{shelf}-B{bin_num:02d}",
                        'aisle': aisle,
                        'shelf': shelf,
                        'bin_num': bin_num,
                        'zone_type': zone_type,
                        'x_coord': x_coord,
                        'y_coord': y_coord,
                        'z_coord': z_coord,
                        'length': 0.6,
                        'width': 0.4,
                        'height': 0.4,
                        'max_weight': 50.0 if zone_type != 'fragile' else 20.0,
                        'distance_to_pickup': distance,
                        'requires_ladder': shelf > 2
                    }
                    
                    locations.append(location)
                    bin_id += 1
        
        return pd.DataFrame(locations)
    
    def _generate_orders(self) -> pd.DataFrame:
        """Generate order data"""
        self.logger.info(f"Generating orders for {self.config.num_days} days...")
        
        orders = []
        order_id = 1
        
        start_date = datetime.now() - timedelta(days=self.config.num_days)
        
        for day in range(self.config.num_days):
            current_date = start_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            
            # Adjust number of orders for weekends
            daily_orders = random.randint(*self.config.orders_per_day_range)
            if is_weekend:
                daily_orders = int(daily_orders * self.config.weekend_factor)
            
            # Generate orders for the day
            for _ in range(daily_orders):
                # Select order hour based on peak hours
                if random.random() < 0.7:  # 70% orders in peak hours
                    hour = random.choice(self.config.peak_hours)
                else:
                    hour = random.randint(8, 22)  # Non-peak hours
                
                order_time = current_date + timedelta(hours=hour, minutes=random.randint(0, 59))
                
                # Generate order items
                num_items = random.randint(*self.config.items_per_order_range)
                products = random.sample(range(1, self.config.num_products + 1), num_items)
                
                for product_id in products:
                    order = {
                        'order_id': order_id,
                        'timestamp': order_time,
                        'product_id': product_id,
                        'quantity': random.randint(1, 3),
                        'is_priority': random.random() < 0.1
                    }
                    orders.append(order)
                
                order_id += 1
        
        return pd.DataFrame(orders)
    
    def _generate_inventory(self) -> pd.DataFrame:
        """Generate inventory data"""
        self.logger.info("Generating inventory data...")
        
        inventory = []
        inventory_id = 1
        
        # Generate initial inventory
        for _, product in self.products.iterrows():
            # Find compatible locations
            compatible_locations = self._get_compatible_locations(product)
            
            # Randomly assign to 1-3 locations
            num_locations = random.randint(1, 3)
            selected_locations = random.sample(compatible_locations, min(num_locations, len(compatible_locations)))
            
            for location in selected_locations:
                inventory.append({
                    'inventory_id': inventory_id,
                    'product_id': product['product_id'],
                    'bin_id': location,
                    'quantity': random.randint(1, 5),
                    'date_added': datetime.now() - timedelta(days=random.randint(0, 30))
                })
                inventory_id += 1
        
        return pd.DataFrame(inventory)
    
    def _assign_zone_type(self, aisle: int) -> str:
        """Assign zone type to aisle"""
        if aisle <= self.config.num_aisles * 0.5:
            return 'normal'
        elif aisle <= self.config.num_aisles * 0.7:
            return 'cold_storage'
        elif aisle <= self.config.num_aisles * 0.85:
            return 'fragile'
        else:
            return 'hazardous'
    
    def _get_compatible_locations(self, product: pd.Series) -> List[str]:
        """Get compatible locations for product"""
        compatible_zones = {
            'normal': ['normal'],
            'cold_storage': ['cold_storage'],
            'fragile': ['fragile'],
            'hazardous': ['hazardous'],
            'high_value': ['fragile']  # High value items go in fragile zone
        }
        
        allowed_zones = compatible_zones.get(product['category'], ['normal'])
        
        return self.storage_locations[
            self.storage_locations['zone_type'].isin(allowed_zones)
        ]['bin_id'].tolist()
    
    def save_data(self, output_dir: str):
        """Save generated data to CSV files"""
        self.logger.info(f"Saving data to {output_dir}...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each dataset
        self.products.to_csv(f"{output_dir}/products.csv", index=False)
        self.storage_locations.to_csv(f"{output_dir}/storage_locations.csv", index=False)
        self.orders.to_csv(f"{output_dir}/orders.csv", index=False)
        self.inventory.to_csv(f"{output_dir}/inventory.csv", index=False)
        
        # Save configuration
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data generation...")
    
    # Create generator with default config
    generator = WarehouseDataGenerator()
    
    # Generate data
    data = generator.generate_all_data()
    
    # Save data
    generator.save_data("sample_data")
    
    # Print summary
    logger.info("\nGenerated Data Summary:")
    for name, df in data.items():
        logger.info(f"{name}: {len(df)} records")

if __name__ == "__main__":
    main()

# This data generator provides:

# 1. Comprehensive Data Generation:
# - Products with realistic attributes
# - Storage locations with zones
# - Orders with temporal patterns
# - Inventory positions

# 2. Configurable Parameters:
# - Number of products/orders
# - Warehouse dimensions
# - Category distributions
# - Time patterns

# 3. Realistic Patterns:
# - Peak hour distribution
# - Weekend patterns
# - Zone assignments
# - Product compatibility

# 4. Key Features:
# - Consistent relationships
# - Constraint adherence
# - Realistic distributions
# - Complete metadata
