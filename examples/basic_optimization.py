"""
Basic example demonstrating core warehouse optimization functionality.
Shows item placement optimization using the RL system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from warehouse_optimization.core import (
    WarehouseEnvironment,
    WarehouseOptimizer
)
from warehouse_optimization.managers import (
    BinManager,
    ConstraintManager,
    AffinityManager,
    StateManager
)
from warehouse_optimization.forecasting import DemandForecaster
from warehouse_optimization.visualization import WarehouseDashboard
from warehouse_optimization.utils import (
    DataProcessor,
    WarehouseMetrics,
    DataValidator
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate sample data for demonstration"""
    # Sample products
    products = pd.DataFrame({
        'product_id': range(1, 101),
        'category': np.random.choice(['electronics', 'fragile', 'normal', 'cold_storage'], 100),
        'length': np.random.uniform(0.1, 0.5, 100),
        'width': np.random.uniform(0.1, 0.5, 100),
        'height': np.random.uniform(0.1, 0.3, 100),
        'weight': np.random.uniform(0.5, 10, 100),
        'price': np.random.uniform(10, 1000, 100)
    })
    
    # Sample warehouse layout
    layout_grid = np.zeros((5, 10, 3))  # 5 aisles, 10 sections, 3 levels
    zones = {
        'normal': [(i, j, k) for i in range(3) for j in range(8) for k in range(3)],
        'cold_storage': [(3, j, k) for j in range(8) for k in range(3)],
        'fragile': [(4, j, k) for j in range(8) for k in range(3)]
    }
    
    # Sample storage locations
    storage_locations = {}
    for zone_type, locations in zones.items():
        for i, loc in enumerate(locations):
            bin_id = f"{zone_type}-{i}"
            storage_locations[bin_id] = {
                'zone_type': zone_type,
                'coordinates': loc,
                'dimensions': {
                    'length': 0.6,
                    'width': 0.4,
                    'height': 0.4,
                    'max_weight': 50.0
                }
            }
    
    # Sample order history
    num_orders = 1000
    order_data = []
    for order_id in range(num_orders):
        num_items = np.random.randint(1, 5)
        for _ in range(num_items):
            order_data.append({
                'order_id': order_id,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'product_id': np.random.randint(1, 101),
                'quantity': np.random.randint(1, 3)
            })
    orders = pd.DataFrame(order_data)
    
    return products, layout_grid, zones, storage_locations, orders

def main():
    """Main execution function"""
    logger.info("Starting basic optimization example")
    
    # Generate sample data
    logger.info("Generating sample data...")
    products, layout_grid, zones, storage_locations, orders = generate_sample_data()
    
    # Initialize data processor and validate data
    logger.info("Initializing data processing...")
    processor = DataProcessor()
    validator = DataValidator()
    
    # Process and validate data
    processed_orders = processor.preprocess_order_data(orders)
    is_valid, violations = validator.validate_order_data(processed_orders)
    if not is_valid:
        logger.error(f"Data validation failed: {violations}")
        return
    
    # Initialize managers
    logger.info("Initializing managers...")
    bin_manager = BinManager(safety_factor=0.85, min_gap=0.02)
    constraint_manager = ConstraintManager()
    affinity_manager = AffinityManager(processed_orders, storage_locations)
    state_manager = StateManager(layout_grid, zones, storage_locations)
    
    # Initialize forecaster
    logger.info("Initializing forecaster...")
    forecaster = DemandForecaster(processed_orders)
    
    # Create environment
    logger.info("Creating optimization environment...")
    env = WarehouseEnvironment(
        state_manager=state_manager,
        bin_manager=bin_manager,
        constraint_manager=constraint_manager,
        affinity_manager=affinity_manager,
        forecaster=forecaster
    )
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = WarehouseOptimizer(env=env)
    
    # Train the model
    logger.info("Training optimizer...")
    optimizer.train(total_timesteps=10000)  # Reduced for example
    
    # Optimize some placements
    logger.info("Testing optimization...")
    test_products = products.sample(5)  # Test with 5 random products
    
    results = []
    for _, product in test_products.iterrows():
        logger.info(f"Optimizing placement for product {product['product_id']}")
        
        placement, metrics = optimizer.optimize_placement(
            product,
            state_manager.get_current_state()
        )
        
        results.append({
            'product_id': product['product_id'],
            'placement': placement,
            'metrics': metrics
        })
    
    # Create visualization
    logger.info("Creating visualization...")
    dashboard = WarehouseDashboard()
    current_metrics = WarehouseMetrics().calculate_optimization_metrics(
        state_manager.get_current_state(),
        [],  # No picking routes in this example
        datetime.now()
    )
    
    dashboard_fig = dashboard.create_dashboard(
        state_manager.get_current_state(),
        current_metrics
    )
    
    # Display results
    logger.info("\nOptimization Results:")
    for result in results:
        logger.info(f"\nProduct {result['product_id']}:")
        logger.info(f"Optimal location: {result['placement']}")
        logger.info(f"Metrics: {result['metrics']}")
    
    logger.info("\nVisualization created - you can display the dashboard_fig using your preferred plotting library")
    
if __name__ == "__main__":
    main()

# This basic example demonstrates:

# 1. Core Functionality:
# - Data generation
# - System initialization
# - Optimization process
# - Result visualization

# 2. Key Components:
# - Environment setup
# - Manager initialization
# - Model training
# - Placement optimization

# 3. Data Flow:
# - Data processing
# - Validation
# - Optimization
# - Visualization

# 4. Error Handling:
# - Data validation
# - Process logging
# - Result tracking
