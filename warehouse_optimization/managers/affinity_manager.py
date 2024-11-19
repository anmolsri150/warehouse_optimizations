import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import logging
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import networkx as nx
from datetime import datetime, timedelta

from ..core.types import ProductAttributes, StorageLocation, WarehouseState

class AffinityManager:
    """
    Manages product affinities and co-occurrence relationships.
    Calculates optimal product placements based on order history patterns.
    """
    
    def __init__(
        self,
        order_history: pd.DataFrame,
        storage_locations: Dict[str, StorageLocation],
        affinity_window_days: int = 90,
        min_support: float = 0.01,
        time_decay_factor: float = 0.1,
        cache_refresh_hours: int = 24
    ):
        self.order_history = order_history
        self.storage_locations = storage_locations
        self.affinity_window_days = affinity_window_days
        self.min_support = min_support
        self.time_decay_factor = time_decay_factor
        
        # Initialize affinity matrices
        self.product_affinity_matrix = None
        self.location_affinity_scores = None
        self.frequent_pairs = None
        self.product_groups = None
        
        # Cache management
        self.cache_refresh_hours = cache_refresh_hours
        self.last_cache_update = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize affinity data
        self._initialize_affinity_data()
    
    def _initialize_affinity_data(self):
        """Initialize affinity matrices and related data structures"""
        self.logger.info("Initializing affinity data...")
        
        # Calculate basic affinity matrix
        self.product_affinity_matrix = self._calculate_product_affinity_matrix()
        
        # Find frequent product pairs
        self.frequent_pairs = self._find_frequent_product_pairs()
        
        # Generate product groups
        self.product_groups = self._generate_product_groups()
        
        # Calculate location affinity scores
        self.location_affinity_scores = self._calculate_location_affinity_scores()
        
        self.last_cache_update = datetime.now()
        
        self.logger.info("Affinity data initialization completed")
    
    def _calculate_product_affinity_matrix(self) -> pd.DataFrame:
        """Calculate product co-occurrence matrix with time decay"""
        self.logger.info("Calculating product affinity matrix...")
        
        # Filter recent orders
        cutoff_date = datetime.now() - timedelta(days=self.affinity_window_days)
        recent_orders = self.order_history[
            self.order_history['timestamp'] >= cutoff_date
        ]
        
        # Create co-occurrence matrix
        product_pairs = []
        pair_weights = []
        
        for order_id, group in recent_orders.groupby('order_id'):
            products = group['product_id'].tolist()
            order_date = group['timestamp'].iloc[0]
            
            # Calculate time decay weight
            days_old = (datetime.now() - order_date).days
            time_weight = np.exp(-self.time_decay_factor * days_old)
            
            # Generate product pairs
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    product_pairs.append((products[i], products[j]))
                    pair_weights.append(time_weight)
        
        # Create affinity matrix
        unique_products = self.order_history['product_id'].unique()
        affinity_matrix = pd.DataFrame(
            0,
            index=unique_products,
            columns=unique_products
        )
        
        # Fill matrix
        for (prod1, prod2), weight in zip(product_pairs, pair_weights):
            affinity_matrix.loc[prod1, prod2] += weight
            affinity_matrix.loc[prod2, prod1] += weight
        
        # Normalize matrix
        row_sums = affinity_matrix.sum(axis=1)
        affinity_matrix = affinity_matrix.div(row_sums, axis=0)
        
        return affinity_matrix
    
    def _find_frequent_product_pairs(self) -> List[Tuple[int, int, float]]:
        """Find frequently co-occurring product pairs"""
        frequent_pairs = []
        total_orders = len(self.order_history['order_id'].unique())
        
        for prod1 in self.product_affinity_matrix.index:
            for prod2 in self.product_affinity_matrix.columns:
                if prod1 < prod2:  # Avoid duplicates
                    support = self.product_affinity_matrix.loc[prod1, prod2]
                    if support >= self.min_support:
                        frequent_pairs.append((prod1, prod2, support))
        
        return sorted(frequent_pairs, key=lambda x: x[2], reverse=True)
    
    def _generate_product_groups(self) -> List[Set[int]]:
        """Generate groups of related products using community detection"""
        # Create graph from affinity matrix
        G = nx.Graph()
        
        for prod1, prod2, support in self.frequent_pairs:
            G.add_edge(prod1, prod2, weight=support)
        
        # Detect communities
        communities = nx.community.greedy_modularity_communities(G)
        
        return [set(community) for community in communities]
    
    def _calculate_location_affinity_scores(self) -> Dict[str, Dict[int, float]]:
        """Calculate affinity scores for each location-product pair"""
        location_scores = {}
        
        for bin_id, location in self.storage_locations.items():
            location_scores[bin_id] = {}
            
            # Calculate base scores for each product
            for product_id in self.product_affinity_matrix.index:
                score = self._calculate_location_product_score(
                    location,
                    product_id
                )
                location_scores[bin_id][product_id] = score
        
        return location_scores
    
    def _calculate_location_product_score(
        self,
        location: StorageLocation,
        product_id: int
    ) -> float:
        """Calculate affinity score for a specific location-product pair"""
        # Get products currently in nearby locations
        nearby_products = self._get_nearby_products(location)
        
        if not nearby_products:
            return 0.0
        
        # Calculate affinity with nearby products
        affinities = [
            self.product_affinity_matrix.loc[product_id, other_id]
            for other_id in nearby_products
        ]
        
        return np.mean(affinities) if affinities else 0.0
    
    def _get_nearby_products(
        self,
        location: StorageLocation,
        radius: int = 2
    ) -> Set[int]:
        """Get products in nearby locations"""
        nearby_products = set()
        loc_coords = location.coordinates
        
        for bin_id, other_loc in self.storage_locations.items():
            other_coords = other_loc.coordinates
            
            # Calculate Manhattan distance
            distance = sum(
                abs(a - b)
                for a, b in zip(loc_coords, other_coords)
            )
            
            if 0 < distance <= radius:
                for product_id in other_loc.current_items.keys():
                    nearby_products.add(product_id)
        
        return nearby_products
    
    def calculate_affinity_score(
        self,
        product_id: int,
        bin_id: str
    ) -> float:
        """Calculate affinity score for placing a product in a specific bin"""
        # Check if cache needs refresh
        if self._needs_cache_refresh():
            self._initialize_affinity_data()
        
        # Get cached score
        return self.location_affinity_scores.get(bin_id, {}).get(product_id, 0.0)
    
    def get_recommended_locations(
        self,
        product_id: int,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k recommended locations for a product"""
        scores = [
            (bin_id, self.calculate_affinity_score(product_id, bin_id))
            for bin_id in self.storage_locations.keys()
        ]
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_product_group(self, product_id: int) -> Optional[Set[int]]:
        """Get the group of related products for a given product"""
        for group in self.product_groups:
            if product_id in group:
                return group
        return None
    
    def update_order_history(self, new_orders: pd.DataFrame):
        """Update order history with new orders"""
        self.order_history = pd.concat(
            [self.order_history, new_orders]
        ).reset_index(drop=True)
        
        # Remove old orders
        cutoff_date = datetime.now() - timedelta(days=self.affinity_window_days)
        self.order_history = self.order_history[
            self.order_history['timestamp'] >= cutoff_date
        ]
        
        # Reinitialize affinity data
        self._initialize_affinity_data()
    
    def _needs_cache_refresh(self) -> bool:
        """Check if cache needs to be refreshed"""
        if self.last_cache_update is None:
            return True
            
        hours_since_update = (
            datetime.now() - self.last_cache_update
        ).total_seconds() / 3600
        
        return hours_since_update >= self.cache_refresh_hours


# This `affinity_manager.py` provides:

# 1. Core Affinity Features:
# - Product co-occurrence matrix calculation
# - Time-decay weighted affinities
# - Location-based affinity scoring
# - Product grouping

# 2. Advanced Analytics:
# - Community detection for product groups
# - Frequent pattern mining
# - Time-weighted analysis
# - Spatial relationship handling

# 3. Optimization Features:
# - Cache management
# - Efficient matrix operations
# - Incremental updates
# - Score normalization

# 4. Location Management:
# - Nearby product detection
# - Location-product scoring
# - Spatial relationship handling
# - Recommendations

# 5. Data Management:
# - Order history processing
# - Cache refresh handling
# - Incremental updates
# - Data cleanup