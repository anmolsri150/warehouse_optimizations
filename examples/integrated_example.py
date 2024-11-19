"""
Integrated example demonstrating complete warehouse optimization system
using generated data and all components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, Tuple

# Import data generator
from data_generator import WarehouseDataGenerator, GeneratorConfig

# Import core components
from warehouse_optimization.core import (
    WarehouseEnvironment,
    WarehouseOptimizer,
    ProductAttributes,
    WarehouseState
)

# Import managers
from warehouse_optimization.managers import (
    BinManager,
    ConstraintManager,
    AffinityManager,
    StateManager
)

# Import forecasting
from warehouse_optimization.forecasting import (
    DemandForecaster,
    SeasonalityAnalyzer,
    ForecastConfig
)

# Import visualization
from warehouse_optimization.visualization import (
    WarehouseDashboard,
    WarehouseHeatmap,
    WarehouseLayout,
    PerformanceVisualizer
)

# Import utilities
from warehouse_optimization.utils import (
    DataProcessor,
    WarehouseMetrics,
    DataValidator
)

class WarehouseOptimizationSystem:
    """
    Integrated warehouse optimization system combining all components.
    """
    
    def __init__(self, output_dir: str = "optimization_results"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.validator = DataValidator()
        self.metrics = WarehouseMetrics()
        
        # Will be initialized with data
        self.environment = None
        self.optimizer = None
        self.forecaster = None
        self.dashboard = None
    
    def initialize_with_data(
        self,
        products: pd.DataFrame,
        storage_locations: pd.DataFrame,
        orders: pd.DataFrame,
        inventory: pd.DataFrame
    ):
        """Initialize system with data"""
        self.logger.info("Initializing system with data...")
        
        # Process and validate data
        self._process_and_validate_data(
            products,
            storage_locations,
            orders,
            inventory
        )
        
        # Initialize managers
        bin_manager = self._initialize_bin_manager(storage_locations)
        constraint_manager = self._initialize_constraint_manager(products)
        affinity_manager = self._initialize_affinity_manager(orders, storage_locations)
        state_manager = self._initialize_state_manager(storage_locations)
        
        # Initialize forecaster
        self.forecaster = self._initialize_forecaster(orders)
        
        # Initialize environment
        self.environment = WarehouseEnvironment(
            state_manager=state_manager,
            bin_manager=bin_manager,
            constraint_manager=constraint_manager,
            affinity_manager=affinity_manager,
            forecaster=self.forecaster
        )
        
        # Initialize optimizer
        self.optimizer = WarehouseOptimizer(self.environment)
        
        # Initialize visualization
        self.dashboard = WarehouseDashboard()
        self.heatmap = WarehouseHeatmap()
        self.layout_viz = WarehouseLayout()
        self.performance_viz = PerformanceVisualizer()
    
    def train_system(self, total_timesteps: int = 50000):
        """Train the optimization system"""
        self.logger.info(f"Training system for {total_timesteps} timesteps...")
        
        try:
            training_results = self.optimizer.train(total_timesteps=total_timesteps)
            
            # Save training metrics
            self._save_training_results(training_results)
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def optimize_placements(
        self,
        new_items: pd.DataFrame
    ) -> Dict[int, Dict]:
        """Optimize placements for new items"""
        self.logger.info(f"Optimizing placements for {len(new_items)} items...")
        
        results = {}
        current_state = self.environment.get_current_state()
        
        for _, item in new_items.iterrows():
            try:
                # Generate forecast for item
                forecast = self.forecaster.generate_forecast(item['product_id'])
                
                # Optimize placement
                placement, metrics = self.optimizer.optimize_placement(
                    item,
                    current_state
                )
                
                results[item['product_id']] = {
                    'placement': placement,
                    'metrics': metrics,
                    'forecast': forecast
                }
                
                # Update current state
                current_state = self.environment.get_current_state()
                
            except Exception as e:
                self.logger.error(f"Error optimizing placement for item {item['product_id']}: {str(e)}")
                continue
        
        return results
    
    def generate_visualizations(
        self,
        optimization_results: Dict[int, Dict]
    ) -> Dict[str, any]:
        """Generate comprehensive visualizations"""
        self.logger.info("Generating visualizations...")
        
        visualizations = {}
        current_state = self.environment.get_current_state()
        current_metrics = self.metrics.calculate_optimization_metrics(
            current_state,
            [],  # No picking routes in this example
            datetime.now()
        )
        
        try:
            # Create dashboard
            visualizations['dashboard'] = self.dashboard.create_dashboard(
                current_state,
                current_metrics
            )
            
            # Create heatmap
            visualizations['heatmap'] = self.heatmap.create_utilization_heatmap(
                current_state
            )
            
            # Create layout visualization
            visualizations['layout'] = self.layout_viz.create_layout_view(
                current_state
            )
            
            # Create performance visualization
            visualizations['performance'] = self.performance_viz.create_performance_dashboard(
                current_metrics,
                current_state
            )
            
            # Save visualizations
            self._save_visualizations(visualizations)
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise
    
    def _process_and_validate_data(
        self,
        products: pd.DataFrame,
        storage_locations: pd.DataFrame,
        orders: pd.DataFrame,
        inventory: pd.DataFrame
    ):
        """Process and validate input data"""
        # Process data
        processed_products = self.data_processor.preprocess_product_data(products)
        processed_locations = self.data_processor.preprocess_location_data(storage_locations)
        processed_orders = self.data_processor.preprocess_order_data(orders)
        processed_inventory = self.data_processor.preprocess_inventory_data(inventory)
        
        # Validate data
        for name, data in [
            ("products", processed_products),
            ("locations", processed_locations),
            ("orders", processed_orders),
            ("inventory", processed_inventory)
        ]:
            is_valid, violations = self.validator.validate_order_data(data)
            if not is_valid:
                raise ValueError(f"Data validation failed for {name}: {violations}")
    
    def _save_training_results(self, results: Dict):
        """Save training results"""
        pd.DataFrame(results).to_csv(
            f"{self.output_dir}/training_results.csv",
            index=False
        )
    
    def _save_visualizations(self, visualizations: Dict):
        """Save visualization figures"""
        for name, fig in visualizations.items():
            fig.write_html(f"{self.output_dir}/{name}.html")

def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Generate sample data
    logger.info("Generating sample data...")
    generator = WarehouseDataGenerator()
    data = generator.generate_all_data()
    
    # Initialize system
    logger.info("Initializing optimization system...")
    system = WarehouseOptimizationSystem()
    system.initialize_with_data(
        products=data['products'],
        storage_locations=data['storage_locations'],
        orders=data['orders'],
        inventory=data['inventory']
    )
    
    # Train system
    logger.info("Training system...")
    training_results = system.train_system(total_timesteps=10000)  # Reduced for example
    
    # Generate new items for optimization
    new_items = data['products'].sample(5)  # Test with 5 random products
    
    # Optimize placements
    logger.info("Optimizing placements...")
    optimization_results = system.optimize_placements(new_items)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizations = system.generate_visualizations(optimization_results)
    
    # Print results
    logger.info("\nOptimization Results:")
    for product_id, result in optimization_results.items():
        logger.info(f"\nProduct {product_id}:")
        logger.info(f"Optimal location: {result['placement']}")
        logger.info(f"Metrics: {result['metrics']}")
        logger.info(f"Forecast mean demand: {np.mean(result['forecast'].forecast_values):.2f}")
    
    logger.info(f"\nResults saved to {system.output_dir}")

if __name__ == "__main__":
    main()

# This integrated example demonstrates:

# 1. System Integration:
# - Data generation
# - Component initialization
# - System training
# - Optimization process

# 2. Complete Workflow:
# - Data processing
# - Validation
# - Training
# - Optimization
# - Visualization

# 3. Key Features:
# - Error handling
# - Result tracking
# - Visualization generation
# - Result storage

# 4. Usage Pattern:
# - System initialization
# - Data integration
# - Optimization
# - Result analysis

# To use this example:

# 1. Run the script directly:
# ```python
# python integrated_example.py
# ```

# 2. Or import and use the system in your code:
# ```python
# from integrated_example import WarehouseOptimizationSystem

# # Initialize system
# system = WarehouseOptimizationSystem()

# # Initialize with your data
# system.initialize_with_data(
#     products=your_products_df,
#     storage_locations=your_locations_df,
#     orders=your_orders_df,
#     inventory=your_inventory_df
# )

# # Train and optimize
# system.train_system()
# results = system.optimize_placements(new_items_df)
# ```
