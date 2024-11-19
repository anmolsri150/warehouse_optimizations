"""
Comprehensive warehouse optimization example incorporating all components:
- Full constraint system
- Advanced forecasting
- Multi-level optimization
- Complete visualization
- Performance tracking
- All managers and handlers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import plotly.graph_objects as go

# Import all components
from warehouse_optimization.core import (
    WarehouseEnvironment,
    WarehouseOptimizer,
    ProductAttributes,
    WarehouseState,
    OptimizationState,
    BinState,
    StorageLocation,
    ItemDimensions,
    BinDimensions,
    PlacementConstraints,
    PickingRoute
)

from warehouse_optimization.managers import (
    BinManager,
    ConstraintManager,
    AffinityManager,
    StateManager,
    ZoneConstraints,
    BinUtilization
)

from warehouse_optimization.forecasting import (
    DemandForecaster,
    EnsembleModel,
    SeasonalityAnalyzer,
    ForecastConfig,
    ModelWeight
)

from warehouse_optimization.visualization import (
    WarehouseDashboard,
    WarehouseHeatmap,
    WarehouseLayout,
    PerformanceVisualizer,
    HeatmapConfig,
    LayoutConfig,
    ViewMode
)

from warehouse_optimization.utils import (
    DataProcessor,
    WarehouseMetrics,
    DataValidator,
    DataPreprocessConfig,
    MetricsConfig,
    ValidationConfig
)

@dataclass
class SystemConfig:
    """Comprehensive system configuration"""
    # Physical constraints
    max_bin_weight: float = 50.0  # kg
    max_bin_volume: float = 0.1  # m³
    min_gap: float = 0.02  # m
    max_stack_height: float = 2.0  # m
    
    # Zone constraints
    zone_rules: Dict[str, Dict] = None
    product_rules: Dict[str, Dict] = None
    
    # Optimization parameters
    optimization_window: int = 7  # days
    reoptimization_threshold: float = 0.2
    max_moves_per_day: int = 50
    
    # Forecasting parameters
    forecast_horizon: int = 14  # days
    seasonality_mode: str = 'multiplicative'
    use_ensemble: bool = True
    
    # Performance thresholds
    min_picking_efficiency: float = 0.7
    max_travel_distance: float = 100  # m
    min_space_utilization: float = 0.6
    
    def __post_init__(self):
        if self.zone_rules is None:
            self.zone_rules = {
                'normal': {
                    'temp_range': (15, 25),
                    'humidity_range': (30, 60),
                    'max_weight_per_shelf': 100.0,
                    'requires_ventilation': False,
                    'compatible_products': {'normal', 'general'},
                    'incompatible_products': set(),
                    'max_stack_height': 4
                },
                'cold_storage': {
                    'temp_range': (-5, 5),
                    'humidity_range': (85, 95),
                    'max_weight_per_shelf': 80.0,
                    'requires_ventilation': True,
                    'compatible_products': {'frozen', 'perishable'},
                    'incompatible_products': {'electronics', 'hazardous'},
                    'max_stack_height': 3
                },
                'fragile': {
                    'temp_range': (15, 25),
                    'humidity_range': (30, 50),
                    'max_weight_per_shelf': 50.0,
                    'requires_ventilation': False,
                    'compatible_products': {'fragile', 'high_value'},
                    'incompatible_products': {'heavy', 'hazardous'},
                    'max_stack_height': 1
                },
                'hazardous': {
                    'temp_range': (10, 30),
                    'humidity_range': (20, 40),
                    'max_weight_per_shelf': 60.0,
                    'requires_ventilation': True,
                    'compatible_products': {'hazardous', 'chemical'},
                    'incompatible_products': {'food', 'perishable', 'fragile'},
                    'max_stack_height': 2,
                    'requires_separation': True,
                    'separation_distance': 2.0  # meters
                },
                'high_value': {
                    'temp_range': (18, 22),
                    'humidity_range': (40, 50),
                    'max_weight_per_shelf': 40.0,
                    'requires_ventilation': False,
                    'compatible_products': {'high_value', 'fragile'},
                    'incompatible_products': {'hazardous', 'heavy'},
                    'max_stack_height': 1,
                    'requires_monitoring': True,
                    'security_level': 'high'
                }
            }

        if self.product_rules is None:
            self.product_rules = {
                'normal': {
                    'max_stack': 4,
                    'temp_sensitivity': 'low',
                    'handling_requirements': 'standard'
                },
                'fragile': {
                    'max_stack': 1,
                    'temp_sensitivity': 'medium',
                    'handling_requirements': 'careful',
                    'requires_padding': True
                },
                'perishable': {
                    'max_stack': 3,
                    'temp_sensitivity': 'high',
                    'handling_requirements': 'temperature_controlled',
                    'shelf_life_tracking': True
                },
                'hazardous': {
                    'max_stack': 2,
                    'temp_sensitivity': 'medium',
                    'handling_requirements': 'specialized',
                    'requires_certification': True,
                    'ventilation_required': True
                },
                'high_value': {
                    'max_stack': 1,
                    'temp_sensitivity': 'medium',
                    'handling_requirements': 'secure',
                    'requires_tracking': True,
                    'insurance_required': True
                }
            }

class ComprehensiveWarehouseSystem:
    """
    Comprehensive warehouse optimization system integrating all components
    and implementing all discussed features.
    """
    
    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        output_dir: str = "warehouse_optimization_results"
    ):
        self.config = config or SystemConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Track system state
        self.current_state: Optional[WarehouseState] = None
        self.optimization_history: List[Dict] = []
        self.performance_metrics: List[Dict] = []
        
    def _initialize_components(self):
        """Initialize all system components"""
        # Initialize utils
        self.data_processor = DataProcessor(
            DataPreprocessConfig(
                remove_outliers=True,
                fill_missing=True,
                normalize_features=True
            )
        )
        
        self.validator = DataValidator(
            ValidationConfig(
                max_weight_per_shelf=self.config.max_bin_weight,
                max_items_per_bin=50,
                min_gap_between_items=self.config.min_gap,
                max_stack_height=self.config.max_stack_height
            )
        )
        
        self.metrics = WarehouseMetrics(
            MetricsConfig(
                time_window=timedelta(days=self.config.optimization_window),
                moving_average_window=24,
                distance_unit='meters',
                time_unit='seconds'
            )
        )
        
        # Initialize visualization components
        self.dashboard = WarehouseDashboard(
            update_interval=5000,
            enable_3d=True
        )
        
        self.heatmap = WarehouseHeatmap(
            HeatmapConfig(
                colorscale='Viridis',
                show_labels=True
            )
        )
        
        self.layout_viz = WarehouseLayout(
            LayoutConfig(
                view_mode=ViewMode.ISOMETRIC,
                show_labels=True,
                show_grid=True
            )
        )
        
        self.performance_viz = PerformanceVisualizer()
        
        # Other components will be initialized with data
        self.environment = None
        self.optimizer = None
        self.forecaster = None
        self.bin_manager = None
        self.constraint_manager = None
        self.affinity_manager = None
        self.state_manager = None

    def initialize_with_data(
        self,
        products: pd.DataFrame,
        storage_locations: pd.DataFrame,
        orders: pd.DataFrame,
        inventory: pd.DataFrame
    ):
        """Initialize system with data and create all required components"""
        self.logger.info("Initializing comprehensive warehouse system...")
        
        try:
            # Process and validate all data
            processed_data = self._process_and_validate_data(
                products, storage_locations, orders, inventory
            )
            
            # Initialize managers
            self._initialize_managers(processed_data)
            
            # Initialize forecasting
            self._initialize_forecasting(processed_data['orders'])
            
            # Initialize environment and optimizer
            self._initialize_optimization(processed_data)
            
            # Set initial state
            self.current_state = self.environment.get_current_state()
            
            self.logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during system initialization: {str(e)}")
            raise
    
    def optimize_warehouse(
        self,
        reoptimize: bool = False,
        max_moves: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete warehouse optimization
        
        Args:
            reoptimize: Whether to reoptimize existing placements
            max_moves: Maximum number of moves allowed
            
        Returns:
            Dictionary containing optimization results and metrics
        """
        self.logger.info("Starting warehouse optimization...")
        
        try:
            # Generate forecasts
            forecasts = self._generate_forecasts()
            
            # Analyze current state
            current_metrics = self._analyze_current_state()
            
            # Determine items to optimize
            items_to_optimize = self._identify_optimization_candidates(
                forecasts,
                current_metrics,
                reoptimize
            )
            
            # Apply move limit if specified
            if max_moves:
                items_to_optimize = items_to_optimize[:max_moves]
            
            # Optimize placements
            optimization_results = self._optimize_placements(
                items_to_optimize,
                forecasts
            )
            
            # Generate visualizations
            visualizations = self._generate_visualizations(
                optimization_results,
                current_metrics
            )
            
            # Save results
            self._save_results(
                optimization_results,
                current_metrics,
                visualizations
            )
            
            return {
                'results': optimization_results,
                'metrics': current_metrics,
                'visualizations': visualizations,
                'forecasts': forecasts
            }
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise
    
    def _process_and_validate_data(
        self,
        products: pd.DataFrame,
        storage_locations: pd.DataFrame,
        orders: pd.DataFrame,
        inventory: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Process and validate all input data"""
        self.logger.info("Processing and validating data...")
        
        # Process data
        processed = {
            'products': self.data_processor.preprocess_product_data(products),
            'storage_locations': self.data_processor.preprocess_location_data(storage_locations),
            'orders': self.data_processor.preprocess_order_data(orders),
            'inventory': self.data_processor.preprocess_inventory_data(inventory)
        }
        
        # Validate each dataset
        for name, data in processed.items():
            is_valid, violations = self.validator.validate_order_data(data)
            if not is_valid:
                raise ValueError(f"Data validation failed for {name}: {violations}")
        
        return processed
    
    def _initialize_managers(self, processed_data: Dict[str, pd.DataFrame]):
        """Initialize all manager components"""
        self.logger.info("Initializing managers...")
        
        # Initialize bin manager
        self.bin_manager = BinManager(
            safety_factor=0.85,
            min_gap=self.config.min_gap,
            max_stack_height_factor=0.9,
            enable_tetris_optimization=True
        )
        
        # Initialize constraint manager with all rules
        self.constraint_manager = ConstraintManager(
            zone_rules=self.config.zone_rules,
            product_rules=self.config.product_rules,
            safety_rules={
                'max_weight_on_fragile': 1.0,
                'min_separation_hazardous': 2.0,
                'max_stack_height': self.config.max_stack_height
            },
            environmental_rules={
                'temperature_monitoring': True,
                'humidity_monitoring': True
            }
        )
        
        # Initialize affinity manager
        self.affinity_manager = AffinityManager(
            order_history=processed_data['orders'],
            storage_locations=processed_data['storage_locations'],
            affinity_window_days=90,
            min_support=0.01,
            time_decay_factor=0.1
        )
        
        # Initialize state manager
        layout_grid = self._create_layout_grid(processed_data['storage_locations'])
        zones = self._create_zones(processed_data['storage_locations'])
        
        self.state_manager = StateManager(
            layout_grid=layout_grid,
            zones=zones,
            storage_locations=processed_data['storage_locations']
        )
    
    def _initialize_forecasting(self, order_data: pd.DataFrame):
        """Initialize forecasting components"""
        self.logger.info("Initializing forecasting...")
        
        # Initialize seasonality analyzer
        self.seasonality_analyzer = SeasonalityAnalyzer(
            min_pattern_strength=0.1,
            confidence_threshold=0.95,
            enable_automatic_detection=True
        )
        
        # Create ensemble model
        self.ensemble_model = EnsembleModel(
            performance_window=30,
            min_weight=0.1,
            enable_dynamic_weights=True
        )
        
        # Initialize demand forecaster
        self.forecaster = DemandForecaster(
            historical_data=order_data,
            config=ForecastConfig(
                forecast_horizon=self.config.forecast_horizon,
                seasonality_mode=self.config.seasonality_mode,
                use_ensemble=self.config.use_ensemble
            )
        )
    
    def _initialize_optimization(self, processed_data: Dict[str, pd.DataFrame]):
        """Initialize optimization components"""
        self.logger.info("Initializing optimization components...")
        
        # Create environment
        self.environment = WarehouseEnvironment(
            state_manager=self.state_manager,
            bin_manager=self.bin_manager,
            constraint_manager=self.constraint_manager,
            affinity_manager=self.affinity_manager,
            forecaster=self.forecaster,
            optimization_window=self.config.optimization_window
        )
        
        # Create optimizer with PPO configuration
        self.optimizer = WarehouseOptimizer(
            env=self.environment,
            model_config={
                "policy_kwargs": dict(
                    net_arch=[256, 256, dict(vf=[128, 64], pi=[128, 64])]
                ),
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99
            }
        )
    
    def _generate_forecasts(self) -> Dict[int, Any]:
        """Generate forecasts for all products"""
        self.logger.info("Generating forecasts...")
        
        forecasts = {}
        for product_id in self.current_state.item_locations.keys():
            forecast = self.forecaster.generate_forecast(product_id)
            seasonal_patterns = self.seasonality_analyzer.analyze_seasonality(
                self.forecaster._get_product_data(product_id)
            )
            
            forecasts[product_id] = {
                'forecast': forecast,
                'seasonal_patterns': seasonal_patterns
            }
        
        return forecasts
    
    def _analyze_current_state(self) -> Dict[str, float]:
        """Analyze current warehouse state"""
        return self.metrics.calculate_optimization_metrics(
            self.current_state,
            [],  # No picking routes in this example
            datetime.now()
        )
    
    def _identify_optimization_candidates(
        self,
        forecasts: Dict[int, Any],
        current_metrics: Dict[str, float],
        reoptimize: bool
    ) -> List[int]:
        """Identify items that need optimization"""
        candidates = []
        
        for product_id, forecast_data in forecasts.items():
            forecast = forecast_data['forecast']
            
            # Check if item needs optimization
            if self._needs_optimization(
                product_id,
                forecast,
                current_metrics,
                reoptimize
            ):
                candidates.append(product_id)
        
        # Sort by priority
        candidates.sort(
            key=lambda x: self._calculate_optimization_priority(
                x,
                forecasts[x]['forecast']
            ),
            reverse=True
        )
        
        return candidates
    
    def _needs_optimization(
        self,
        product_id: int,
        forecast: Any,
        current_metrics: Dict[str, float],
        reoptimize: bool
    ) -> bool:
        """Determine if item needs optimization"""
        if reoptimize:
            return True
            
        current_location = self.current_state.item_locations.get(product_id)
        if not current_location:
            return True
            
        # Check forecast-based criteria
        if np.mean(forecast.forecast_values) > forecast.trend_factor * 1.2:
            return True
            
        # Check performance-based criteria
        if current_metrics['picking_efficiency'] < self.config.min_picking_efficiency:
            return True
            
        return False
    
    def _calculate_optimization_priority(
        self,
        product_id: int,
        forecast: Any
    ) -> float:
        """Calculate optimization priority for item"""
        # Combine multiple factors for priority
        forecast_priority = np.mean(forecast.forecast_values)
        current_location = self.current_state.item_locations.get(product_id)
        
        if current_location:
            location_score = self._calculate_location_score(
                product_id,
                current_location
            )
        else:
            location_score = 0
            
        return forecast_priority * (1 - location_score)
    
    def _optimize_placements(
        self,
        items: List[int],
        forecasts: Dict[int, Any]
    ) -> Dict[int, Dict]:
        """Optimize placements for items"""
        results = {}
        
        for product_id in items:
            placement, metrics = self.optimizer.optimize_placement(
                self._get_product_attributes(product_id),
                self.current_state
            )
            
            results[product_id] = {
                'placement': placement,
                'metrics': metrics,
                'forecast': forecasts[product_id]['forecast']
            }
            
            # Update state if placement is valid
            if placement.get('success', False):
                self.current_state = self.environment.get_current_state()
        
        return results
    
    def _generate_visualizations(
        self,
        optimization_results: Dict[int, Dict],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate all visualizations"""
        return {
            'dashboard': self.dashboard.create_dashboard(
                self.current_state,
                current_metrics
            ),
            'heatmap': self.heatmap.create_utilization_heatmap(
                self.current_state
            ),
            'layout': self.layout_viz.create_layout_view(
                self.current_state,
                highlight_bins=[r['placement']['bin_id'] for r in optimization_results.values()]
            ),
            'performance': self.performance_viz.create_performance_dashboard(
                current_metrics,
                self.current_state
            )
        }
    
    def _save_results(
        self,
        optimization_results: Dict[int, Dict],
        metrics: Dict[str, float],
        visualizations: Dict[str, Any]
    ):
        """Save all results and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save optimization results
        pd.DataFrame(optimization_results).to_csv(
            f"{self.output_dir}/optimization_results_{timestamp}.csv"
        )
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(
            f"{self.output_dir}/metrics_{timestamp}.csv"
        )
        
        # Save visualizations
        for name, fig in visualizations.items():
            fig.write_html(f"{self.output_dir}/{name}_{timestamp}.html")

def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize data generator with specific configuration
    logger.info("Initializing data generator...")
    generator_config = GeneratorConfig(
        num_products=1000,
        num_orders=10000,
        num_days=90,
        num_aisles=10,
        shelves_per_aisle=4,
        bins_per_shelf=6,
        # Product distribution
        product_category_dist={
            'normal': 0.4,
            'fragile': 0.15,
            'cold_storage': 0.2,
            'hazardous': 0.15,
            'high_value': 0.1
        },
        # Price ranges for different categories
        price_ranges={
            'normal': (10, 100),
            'fragile': (50, 500),
            'cold_storage': (20, 200),
            'hazardous': (30, 300),
            'high_value': (500, 5000)
        },
        # Dimension ranges for different categories
        dimension_ranges={
            'normal': {
                'length': (0.2, 0.5),
                'width': (0.2, 0.4),
                'height': (0.1, 0.3),
                'weight': (0.5, 10)
            },
            'fragile': {
                'length': (0.1, 0.3),
                'width': (0.1, 0.3),
                'height': (0.1, 0.2),
                'weight': (0.1, 5)
            },
            'cold_storage': {
                'length': (0.2, 0.4),
                'width': (0.2, 0.4),
                'height': (0.2, 0.4),
                'weight': (0.5, 15)
            },
            'hazardous': {
                'length': (0.1, 0.4),
                'width': (0.1, 0.4),
                'height': (0.1, 0.3),
                'weight': (1, 20)
            },
            'high_value': {
                'length': (0.1, 0.3),
                'width': (0.1, 0.3),
                'height': (0.1, 0.2),
                'weight': (0.1, 3)
            }
        },
        # Order patterns
        orders_per_day_range=(100, 200),
        items_per_order_range=(1, 5),
        peak_hours=[10, 11, 14, 15, 16, 19, 20],
        weekend_factor=0.7
    )
    
    generator = WarehouseDataGenerator(generator_config)
    
    # Generate sample data
    logger.info("Generating sample data...")
    try:
        data = generator.generate_all_data()
        
        # Save generated data for reference
        output_dir = "generated_data"
        os.makedirs(output_dir, exist_ok=True)
        generator.save_data(output_dir)
        logger.info(f"Generated data saved to {output_dir}")
        
        # Print data summary
        logger.info("\nGenerated Data Summary:")
        for name, df in data.items():
            logger.info(f"{name}: {len(df)} records")
            
        # Create system with specific configuration
        system_config = SystemConfig(
            # Physical constraints
            max_bin_weight=50.0,  # kg
            max_bin_volume=0.1,  # m³
            min_gap=0.02,  # m
            max_stack_height=2.0,  # m
            
            # Optimization parameters
            optimization_window=7,  # days
            reoptimization_threshold=0.2,
            max_moves_per_day=50,
            
            # Forecasting parameters
            forecast_horizon=14,  # days
            seasonality_mode='multiplicative',
            use_ensemble=True,
            
            # Performance thresholds
            min_picking_efficiency=0.7,
            max_travel_distance=100,  # m
            min_space_utilization=0.6
        )
        
        logger.info("Initializing warehouse optimization system...")
        system = ComprehensiveWarehouseSystem(config=system_config)
        
        # Initialize system with generated data
        logger.info("Initializing system with generated data...")
        system.initialize_with_data(
            products=data['products'],
            storage_locations=data['storage_locations'],
            orders=data['orders'],
            inventory=data['inventory']
        )
        
        # Run initial optimization
        logger.info("Running initial optimization...")
        initial_results = system.optimize_warehouse(
            reoptimize=True,
            max_moves=50
        )
        
        # Print initial optimization results
        logger.info("\nInitial Optimization Results:")
        logger.info(f"Total items optimized: {len(initial_results['results'])}")
        logger.info(f"Initial picking efficiency: {initial_results['metrics']['picking_efficiency']:.2f}")
        logger.info(f"Initial space utilization: {initial_results['metrics']['space_utilization']:.2f}")
        
        # Simulate some time passing and generate new orders
        logger.info("\nSimulating additional orders...")
        new_orders = generator.generate_additional_orders(
            num_days=7,
            base_orders=data['orders']
        )
        
        # Update system with new orders
        logger.info("Updating system with new orders...")
        system.update_order_history(new_orders)
        
        # Run reoptimization
        logger.info("Running reoptimization...")
        reopt_results = system.optimize_warehouse(
            reoptimize=True,
            max_moves=25  # Limit moves for reoptimization
        )
        
        # Print reoptimization results
        logger.info("\nReoptimization Results:")
        logger.info(f"Total items reoptimized: {len(reopt_results['results'])}")
        logger.info(f"Final picking efficiency: {reopt_results['metrics']['picking_efficiency']:.2f}")
        logger.info(f"Final space utilization: {reopt_results['metrics']['space_utilization']:.2f}")
        
        # Generate and save comprehensive report
        logger.info("\nGenerating comprehensive report...")
        report = {
            'initial_optimization': initial_results,
            'reoptimization': reopt_results,
            'improvement': {
                'picking_efficiency': (
                    reopt_results['metrics']['picking_efficiency'] -
                    initial_results['metrics']['picking_efficiency']
                ),
                'space_utilization': (
                    reopt_results['metrics']['space_utilization'] -
                    initial_results['metrics']['space_utilization']
                )
            }
        }
        
        # Save report
        with open(f"{system.output_dir}/optimization_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nAll results and visualizations saved to: {system.output_dir}")
        logger.info("Optimization complete!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# This comprehensive implementation includes:

# 1. Complete Integration:
# - All managers working together
# - Full constraint system
# - Advanced forecasting
# - Multi-level optimization

# 2. Sophisticated Features:
# - Priority-based optimization
# - Dynamic reoptimization
# - Comprehensive metrics
# - Complete visualization suite

# 3. Advanced Constraints:
# - Zone-specific rules
# - Product compatibility
# - Environmental requirements
# - Safety regulations

# 4. Complex Optimization:
# - Forecast-driven placement
# - Affinity-based grouping
# - Space utilization
# - Picking efficiency

# 5. Complete Visualization:
# - Interactive dashboard
# - Utilization heatmaps
# - 3D layout views
# - Performance metrics

# To use this system, you would:

# 1. Initialize system:
# ```python
# system = ComprehensiveWarehouseSystem()
# ```

# 2. Load your data and initialize:
# ```python
# system.initialize_with_data(
#     products=your_products_df,
#     storage_locations=your_locations_df,
#     orders=your_orders_df,
#     inventory=your_inventory_df
# )
# ```

# 3. Run optimization:
# ```python
# results = system.optimize_warehouse(
#     reoptimize=True,
#     max_moves=50
# )
274