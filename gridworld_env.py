"""
Minimal 2D GridWorld Environment for VEM Demonstration

This is a simple simulation environment to demonstrate the VEM architecture
without requiring complex physics engines. The agent learns to navigate,
collect resources, and maintain homeostasis.

Features:
- 2D grid with objects (food, obstacles, hazards)
- Agent with visual perception, proprioception, and homeostasis
- Continuous perception-action loop
- Suitable for rapid prototyping and validation
"""

import numpy as np
from typing import Tuple, Optional, Dict
from enum import IntEnum


class Action(IntEnum):
    """Available actions for the agent."""
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    INTERACT = 4
    REST = 5


class ObjectType(IntEnum):
    """Types of objects in the environment."""
    EMPTY = 0
    WALL = 1
    FOOD = 2
    HAZARD = 3
    GOAL = 4


class GridWorldEnvironment:
    """
    Simple 2D GridWorld for embodied AI experiments.
    
    The agent has:
    - Visual perception (egocentric view)
    - Proprioception (position, orientation, movement state)
    - Homeostasis (energy, damage)
    
    The environment provides:
    - Spatial navigation challenges
    - Resource collection (food)
    - Hazards (damage)
    - Goals (rewards)
    """
    
    def __init__(self, 
                 grid_size: int = 16,
                 view_distance: int = 5,
                 max_energy: float = 100.0,
                 energy_decay: float = 0.1,
                 food_energy: float = 20.0,
                 hazard_damage: float = 10.0):
        """
        Initialize the GridWorld environment.
        
        Args:
            grid_size: Size of the square grid
            view_distance: How far the agent can see
            max_energy: Maximum energy level
            energy_decay: Energy lost per timestep
            food_energy: Energy gained from eating food
            hazard_damage: Damage from hazards
        """
        self.grid_size = grid_size
        self.view_distance = view_distance
        self.max_energy = max_energy
        self.energy_decay = energy_decay
        self.food_energy = food_energy
        self.hazard_damage = hazard_damage
        
        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Agent state
        self.agent_x = 0
        self.agent_y = 0
        self.agent_orientation = 0  # 0=North, 1=East, 2=South, 3=West
        self.agent_energy = max_energy
        self.agent_damage = 0.0
        
        # Episode tracking
        self.timestep = 0
        self.max_timesteps = 1000
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Create walls, food, hazards, and goals."""
        # Add border walls
        self.grid[0, :] = ObjectType.WALL
        self.grid[-1, :] = ObjectType.WALL
        self.grid[:, 0] = ObjectType.WALL
        self.grid[:, -1] = ObjectType.WALL
        
        # Add some interior walls
        for _ in range(self.grid_size // 2):
            x, y = np.random.randint(1, self.grid_size-1, size=2)
            self.grid[x, y] = ObjectType.WALL
        
        # Add food
        for _ in range(self.grid_size):
            x, y = np.random.randint(1, self.grid_size-1, size=2)
            if self.grid[x, y] == ObjectType.EMPTY:
                self.grid[x, y] = ObjectType.FOOD
        
        # Add hazards
        for _ in range(self.grid_size // 2):
            x, y = np.random.randint(1, self.grid_size-1, size=2)
            if self.grid[x, y] == ObjectType.EMPTY:
                self.grid[x, y] = ObjectType.HAZARD
        
        # Add goal
        x, y = np.random.randint(1, self.grid_size-1, size=2)
        while self.grid[x, y] != ObjectType.EMPTY:
            x, y = np.random.randint(1, self.grid_size-1, size=2)
        self.grid[x, y] = ObjectType.GOAL
        
        # Place agent in empty space
        x, y = np.random.randint(1, self.grid_size-1, size=2)
        while self.grid[x, y] != ObjectType.EMPTY:
            x, y = np.random.randint(1, self.grid_size-1, size=2)
        self.agent_x, self.agent_y = x, y
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.agent_energy = self.max_energy
        self.agent_damage = 0.0
        self.timestep = 0
        self._setup_environment()
        return self.get_observations()
    
    def get_egocentric_view(self) -> np.ndarray:
        """
        Get agent's first-person view.
        
        Returns a local grid centered on agent, oriented with agent's heading.
        Shape: [view_distance*2+1, view_distance*2+1]
        """
        view_size = self.view_distance * 2 + 1
        view = np.ones((view_size, view_size), dtype=np.int32) * ObjectType.WALL
        
        # Direction vectors for each orientation
        direction_vectors = [
            (0, -1),  # North
            (1, 0),   # East
            (0, 1),   # South
            (-1, 0)   # West
        ]
        
        # Get view in world coordinates
        for dy in range(-self.view_distance, self.view_distance + 1):
            for dx in range(-self.view_distance, self.view_distance + 1):
                # Rotate based on orientation
                if self.agent_orientation == 0:  # North
                    world_x = self.agent_x + dx
                    world_y = self.agent_y + dy
                elif self.agent_orientation == 1:  # East
                    world_x = self.agent_x - dy
                    world_y = self.agent_y + dx
                elif self.agent_orientation == 2:  # South
                    world_x = self.agent_x - dx
                    world_y = self.agent_y - dy
                else:  # West
                    world_x = self.agent_x + dy
                    world_y = self.agent_y - dx
                
                # Check bounds
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    view[dy + self.view_distance, dx + self.view_distance] = \
                        self.grid[world_x, world_y]
        
        return view
    
    def get_visual_features(self) -> np.ndarray:
        """
        Convert egocentric view to visual features.
        
        Returns flattened one-hot encoded visual field.
        Shape: [view_cells, n_object_types]
        """
        view = self.get_egocentric_view()
        n_cells = view.size
        n_types = len(ObjectType)
        
        # One-hot encode
        features = np.zeros((n_cells, n_types), dtype=np.float32)
        flat_view = view.flatten()
        features[np.arange(n_cells), flat_view] = 1.0
        
        return features
    
    def get_audio_features(self) -> np.ndarray:
        """
        Generate simple audio features (distance to nearest objects).
        
        In a real implementation, this would be spatial audio simulation.
        """
        # Simplified: distance-based "audio" for nearby objects
        audio = np.zeros(4, dtype=np.float32)  # [food, hazard, wall, goal]
        
        # Check 8-connected neighbors
        neighbors = [
            (self.agent_x-1, self.agent_y),
            (self.agent_x+1, self.agent_y),
            (self.agent_x, self.agent_y-1),
            (self.agent_x, self.agent_y+1),
        ]
        
        for x, y in neighbors:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                obj = self.grid[x, y]
                if obj == ObjectType.FOOD:
                    audio[0] = 1.0
                elif obj == ObjectType.HAZARD:
                    audio[1] = 1.0
                elif obj == ObjectType.WALL:
                    audio[2] = 1.0
                elif obj == ObjectType.GOAL:
                    audio[3] = 1.0
        
        return audio
    
    def get_proprioception(self) -> np.ndarray:
        """
        Get proprioceptive state (body configuration).
        
        Returns: [x_norm, y_norm, orientation_x, orientation_y, velocity]
        """
        # Normalize position to [0, 1]
        x_norm = self.agent_x / self.grid_size
        y_norm = self.agent_y / self.grid_size
        
        # Encode orientation as unit vector
        angle = self.agent_orientation * np.pi / 2
        ori_x = np.cos(angle)
        ori_y = np.sin(angle)
        
        # Velocity (simplified: always 0 in discrete grid)
        velocity = 0.0
        
        return np.array([x_norm, y_norm, ori_x, ori_y, velocity], dtype=np.float32)
    
    def get_homeostasis(self) -> np.ndarray:
        """
        Get homeostatic state (internal physiological signals).
        
        Returns: [energy_norm, damage_norm, hunger, pain]
        """
        energy_norm = self.agent_energy / self.max_energy
        damage_norm = np.clip(self.agent_damage / 100.0, 0, 1)
        
        # Hunger increases as energy decreases
        hunger = 1.0 - energy_norm
        
        # Pain signal
        pain = damage_norm
        
        return np.array([energy_norm, damage_norm, hunger, pain], dtype=np.float32)
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get complete sensory observation."""
        return {
            'visual': self.get_visual_features(),
            'audio': self.get_audio_features(),
            'proprioception': self.get_proprioception(),
            'homeostasis': self.get_homeostasis()
        }
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (from Action enum)
        
        Returns:
            observation: New sensory state
            reward: Scalar reward
            done: Whether episode is finished
            info: Additional information
        """
        reward = 0.0
        info = {}
        
        # Decay energy
        self.agent_energy -= self.energy_decay
        
        # Execute action
        if action == Action.MOVE_FORWARD:
            new_x, new_y = self._get_forward_position()
            if self._is_valid_position(new_x, new_y):
                self.agent_x, self.agent_y = new_x, new_y
            else:
                reward -= 0.1  # Penalty for bumping into wall
        
        elif action == Action.MOVE_BACKWARD:
            new_x, new_y = self._get_backward_position()
            if self._is_valid_position(new_x, new_y):
                self.agent_x, self.agent_y = new_x, new_y
            else:
                reward -= 0.1
        
        elif action == Action.TURN_LEFT:
            self.agent_orientation = (self.agent_orientation - 1) % 4
        
        elif action == Action.TURN_RIGHT:
            self.agent_orientation = (self.agent_orientation + 1) % 4
        
        elif action == Action.INTERACT:
            # Interact with object at current position
            obj = self.grid[self.agent_x, self.agent_y]
            
            if obj == ObjectType.FOOD:
                self.agent_energy = min(self.max_energy, 
                                       self.agent_energy + self.food_energy)
                self.grid[self.agent_x, self.agent_y] = ObjectType.EMPTY
                reward += 1.0
                info['ate_food'] = True
            
            elif obj == ObjectType.GOAL:
                reward += 10.0
                info['reached_goal'] = True
        
        elif action == Action.REST:
            # Resting reduces damage
            self.agent_damage = max(0, self.agent_damage - 1.0)
            reward += 0.1
        
        # Check for hazards at current position
        if self.grid[self.agent_x, self.agent_y] == ObjectType.HAZARD:
            self.agent_damage += self.hazard_damage
            reward -= 1.0
            info['hit_hazard'] = True
        
        # Update timestep
        self.timestep += 1
        
        # Check termination conditions
        done = False
        if self.agent_energy <= 0:
            done = True
            reward -= 10.0
            info['death'] = 'starvation'
        elif self.agent_damage >= 100:
            done = True
            reward -= 10.0
            info['death'] = 'damage'
        elif self.timestep >= self.max_timesteps:
            done = True
            info['timeout'] = True
        elif 'reached_goal' in info:
            done = True
        
        observation = self.get_observations()
        return observation, reward, done, info
    
    def _get_forward_position(self) -> Tuple[int, int]:
        """Get position if moving forward."""
        direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = direction_vectors[self.agent_orientation]
        return self.agent_x + dx, self.agent_y + dy
    
    def _get_backward_position(self) -> Tuple[int, int]:
        """Get position if moving backward."""
        direction_vectors = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        dx, dy = direction_vectors[self.agent_orientation]
        return self.agent_x + dx, self.agent_y + dy
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within bounds and not a wall."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        return self.grid[x, y] != ObjectType.WALL
    
    def render(self, mode: str = 'ascii') -> Optional[str]:
        """
        Render the environment.
        
        Args:
            mode: 'ascii' for text rendering
        """
        if mode == 'ascii':
            symbols = {
                ObjectType.EMPTY: '.',
                ObjectType.WALL: '#',
                ObjectType.FOOD: 'F',
                ObjectType.HAZARD: 'X',
                ObjectType.GOAL: 'G'
            }
            
            agent_symbols = ['^', '>', 'v', '<']
            
            lines = []
            for y in range(self.grid_size):
                row = []
                for x in range(self.grid_size):
                    if x == self.agent_x and y == self.agent_y:
                        row.append(agent_symbols[self.agent_orientation])
                    else:
                        row.append(symbols[self.grid[x, y]])
                lines.append(' '.join(row))
            
            status = f"\nEnergy: {self.agent_energy:.1f}/{self.max_energy} | " \
                    f"Damage: {self.agent_damage:.1f}/100 | " \
                    f"Step: {self.timestep}/{self.max_timesteps}"
            
            return '\n'.join(lines) + status
        
        return None


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("GridWorld Environment - Demonstration")
    print("=" * 70)
    
    env = GridWorldEnvironment(grid_size=12, view_distance=3)
    
    print("\nInitial State:")
    print(env.render())
    
    # Show observation structure
    obs = env.get_observations()
    print("\n\nObservation Structure:")
    print(f"  Visual Features: {obs['visual'].shape}")
    print(f"  Audio Features: {obs['audio'].shape}")
    print(f"  Proprioception: {obs['proprioception'].shape}")
    print(f"  Homeostasis: {obs['homeostasis'].shape}")
    
    print(f"\nProprioception: {obs['proprioception']}")
    print(f"Homeostasis: {obs['homeostasis']}")
    
    # Run a few random steps
    print("\n" + "=" * 70)
    print("Running Random Actions")
    print("=" * 70)
    
    for i in range(5):
        action = np.random.randint(0, len(Action))
        action_name = Action(action).name
        
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {i+1}: {action_name}")
        print(f"  Reward: {reward:.2f}")
        if info:
            print(f"  Info: {info}")
        print(env.render())
        
        if done:
            print("\nEpisode finished!")
            break
    
    print("\n" + "=" * 70)
