from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from datetime import datetime

from .environment import WarehouseEnvironment
from .types import (
    ProductAttributes,
    OptimizationState,
    OptimizationMetrics,
    WarehouseState
)

class OptimizationCallback(BaseCallback):
    """Custom callback for training monitoring and early stopping"""
    
    def __init__(
        self,
        check_freq: int = 1000,
        min_improvement: float = 0.01,
        patience: int = 5,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_improvement = min_improvement
        self.patience = patience
        self.best_reward = -np.inf
        self.steps_without_improvement = 0
        self.training_metrics = []
        
    def _on_step(self) -> bool:
        """Called at each step during training"""
        if self.n_calls % self.check_freq == 0:
            # Get mean reward
            mean_reward = np.mean(self.locals["rewards"])
            self.training_metrics.append({
                'step': self.n_calls,
                'mean_reward': mean_reward,
                'success_rate': self._get_success_rate()
            })
            
            # Check for improvement
            if mean_reward > self.best_reward + self.min_improvement:
                self.best_reward = mean_reward
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1
            
            # Early stopping
            if self.steps_without_improvement >= self.patience:
                self.logger.info("Early stopping triggered!")
                return False
        
        return True
    
    def _get_success_rate(self) -> float:
        """Calculate success rate of placements"""
        episode_infos = self.locals.get("infos", [])
        if not episode_infos:
            return 0.0
        
        successes = sum(1 for info in episode_infos if not info.get("violations", []))
        return successes / len(episode_infos)

class WarehouseOptimizer:
    """
    Main optimizer class for warehouse optimization using RL
    """
    
    def __init__(
        self,
        env: WarehouseEnvironment,
        model_config: Optional[Dict] = None,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None
    ):
        self.env = DummyVecEnv([lambda: env])
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Default model configuration
        default_config = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "tensorboard_log": tensorboard_log,
            "policy_kwargs": dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]
            ),
            "verbose": verbose
        }
        
        # Update with custom config
        if model_config:
            default_config.update(model_config)
        
        # Initialize PPO model
        self.model = PPO(env=self.env, **default_config)
        
        # Training metrics
        self.training_history: List[Dict] = []
        self.optimization_metrics: OptimizationMetrics = None
    
    def train(
        self,
        total_timesteps: int,
        callback_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train the optimization model
        
        Args:
            total_timesteps: Total number of training timesteps
            callback_config: Configuration for the training callback
        
        Returns:
            Dictionary containing training metrics
        """
        # Setup callback
        default_callback_config = {
            "check_freq": 1000,
            "min_improvement": 0.01,
            "patience": 5,
            "verbose": self.verbose
        }
        if callback_config:
            default_callback_config.update(callback_config)
        
        callback = OptimizationCallback(**default_callback_config)
        
        # Train model
        try:
            self.logger.info(f"Starting training for {total_timesteps} timesteps...")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            
            # Store training metrics
            self.training_history = callback.training_metrics
            
            return {
                "success": True,
                "training_history": self.training_history,
                "final_reward": callback.best_reward,
                "early_stopped": callback.steps_without_improvement >= callback.patience
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def optimize_placement(
        self,
        item: ProductAttributes,
        current_state: WarehouseState
    ) -> Tuple[Dict[str, Any], OptimizationMetrics]:
        """
        Optimize placement for a single item
        
        Args:
            item: Item to be placed
            current_state: Current warehouse state
        
        Returns:
            Tuple of (placement_decision, metrics)
        """
        # Reset environment with current item
        obs, _ = self.env.reset(options={"item": item})
        
        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Execute action
        new_obs, reward, done, info = self.env.step(action)
        
        # Calculate metrics
        metrics = self._calculate_optimization_metrics(
            item,
            action,
            reward,
            info
        )
        
        # Store metrics
        self.optimization_metrics = metrics
        
        return {
            "action": action,
            "bin_id": info[0].get("placement_bin"),
            "success": len(info[0].get("violations", [])) == 0,
            "reward": reward,
            "violations": info[0].get("violations", [])
        }, metrics
    
    def batch_optimize(
        self,
        items: List[ProductAttributes],
        current_state: WarehouseState
    ) -> List[Tuple[Dict[str, Any], OptimizationMetrics]]:
        """
        Optimize placement for multiple items
        
        Args:
            items: List of items to be placed
            current_state: Current warehouse state
        
        Returns:
            List of (placement_decision, metrics) tuples
        """
        results = []
        
        for item in items:
            result = self.optimize_placement(item, current_state)
            results.append(result)
            
            # Update current state if placement was successful
            if result[0]["success"]:
                current_state = self.env.get_attr("warehouse_state")[0]
        
        return results
    
    def evaluate(
        self,
        num_episodes: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate the current model
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Dictionary containing evaluation metrics
        """
        episode_rewards = []
        success_rate = []
        picking_times = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if done:
                    success_rate.append(
                        1 if len(info[0].get("violations", [])) == 0 else 0
                    )
                    picking_times.append(
                        info[0].get("picking_time", float('inf'))
                    )
            
            episode_rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "success_rate": np.mean(success_rate),
            "mean_picking_time": np.mean(picking_times),
            "std_picking_time": np.std(picking_times)
        }
    
    def save(self, path: str):
        """Save the optimization model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a saved optimization model"""
        self.model = PPO.load(path, env=self.env)
    
    def _calculate_optimization_metrics(
        self,
        item: ProductAttributes,
        action: np.ndarray,
        reward: float,
        info: Dict
    ) -> OptimizationMetrics:
        """Calculate comprehensive optimization metrics"""
        return OptimizationMetrics(
            picking_efficiency=info[0].get("picking_efficiency", 0.0),
            space_utilization=info[0].get("utilization", {}).get("total", 0.0),
            constraint_satisfaction=1.0 if len(info[0].get("violations", [])) == 0 else 0.0,
            demand_satisfaction=info[0].get("demand_satisfaction", 0.0),
            travel_distance=info[0].get("travel_distance", 0.0),
            total_time=info[0].get("total_time", 0.0),
            num_moves=info[0].get("num_moves", 0)
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("WarehouseOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


# This `optimizer.py` file provides:

# 1. Core Optimization Features:
# - PPO-based RL optimization
# - Custom training callbacks
# - Batch optimization support
# - Model evaluation capabilities

# 2. Training Capabilities:
# - Configurable training parameters
# - Early stopping
# - Training metrics tracking
# - Progress monitoring

# 3. Optimization Functions:
# - Single item placement
# - Batch item placement
# - State management
# - Metrics calculation

# 4. Advanced Features:
# - Custom callback for monitoring
# - Tensorboard logging support
# - Model saving/loading
# - Comprehensive metrics tracking

# 5. Evaluation Tools:
# - Success rate tracking
# - Performance metrics
# - Picking time analysis
# - Constraint satisfaction monitoring

# The optimizer integrates with the environment and provides a complete optimization pipeline for warehouse item placement.
