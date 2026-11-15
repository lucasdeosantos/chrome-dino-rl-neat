import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ..game.game import DinoGame
from ..game.physics import DINO_X, GROUND_Y
from ..game.config import DEFAULT_SCENARIO, NUM_ACTIONS
from .config import OBSERVATION_SIZE, VISIBLE_OBSTACLES, OBSTACLE_FEATURE
from .config import HEIGHT_SCALE, WIDTH_SCALE, DISTANCE_SCALE, VERTICAL_SPEED_SCALE, HORIZONTAL_SPEED_SCALE
from .rewards import RewardCalculator


class DinoEnv(gym.Env):
    def __init__(self, scenario=DEFAULT_SCENARIO, max_steps=None, config=None, fixed_seed=None, sequential_seed=None, reward='r1'):
        self.reward_calculator = RewardCalculator(reward)
        self.next_seed = sequential_seed
        self.fixed_seed = fixed_seed
        self.max_steps = max_steps
        self.game = DinoGame(scenario=scenario, config=config)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self._needs_reset = True
        self._elapsed_steps = 0

    def reset(self, *, seed=None, options=None):
        if self.next_seed is not None:
            seed = self.next_seed
            self.next_seed += 1
        elif self.fixed_seed is not None:
            seed = self.fixed_seed
        
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**63))
        self.game.reset(seed=game_seed)
        self.reward_calculator.reset()
        self._elapsed_steps = 0
        self._needs_reset = False
        return self.get_observation(), self.get_info(False)

    def step(self, action):
        self.reward_calculator.before_step(self.game)
        self.game.update(action)
        self._elapsed_steps += 1
        terminated = self.game.terminated
        truncated = (not terminated and self.max_steps is not None and self._elapsed_steps >= self.max_steps)
        self._needs_reset = terminated or truncated
        reward = self.reward_calculator.calculate(self.game, action)
        return self.get_observation(), reward, terminated, truncated, self.get_info(truncated)

    def get_observation(self):
        dino = self.game.dinosaur
        values = [
            (GROUND_Y - dino.y) / HEIGHT_SCALE,
            dino.get_velocity_y() / VERTICAL_SPEED_SCALE,
            float(dino.is_grounded()), float(dino.ducking),
            self.game.get_speed() / HORIZONTAL_SPEED_SCALE,
        ]
        obstacles = [obstacle for obstacle in self.game.obstacles if obstacle.x + obstacle.width > DINO_X]
        obstacles.sort(key=lambda obstacle: obstacle.x)
        
        for index in range(VISIBLE_OBSTACLES):
            if index >= len(obstacles):
                values.extend([0.0] * OBSTACLE_FEATURE)
                continue
            obstacle = obstacles[index]
            values.extend([
                1.0,
                (obstacle.x - DINO_X) / DISTANCE_SCALE,
                obstacle.y / HEIGHT_SCALE,
                obstacle.width / WIDTH_SCALE,
                obstacle.height / HEIGHT_SCALE,
                obstacle.speed_offset / HORIZONTAL_SPEED_SCALE,
                float(obstacle.variant == 'bird'),
            ])
        
        return np.array(values, dtype=np.float32)

    def get_info(self, reached_limit):
        return { 'points': self.game.get_points(), 'reached_limit': bool(reached_limit) }
