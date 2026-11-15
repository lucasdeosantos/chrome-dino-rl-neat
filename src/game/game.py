from math import ceil
from random import Random
from .config import GameConfig, DEFAULT_SCENARIO, CLEAR_STEPS, SCENARIO_4_ACCELERATION, SCENARIO_4_MAX_SPEED, POINTS_PER_PIXEL
from .dinosaur import Dinosaur
from .obstacles import ObstacleGenerator
from .physics import SIMULATION_STEP_DURATION, SPEED
from .utils import round_up


class DinoGame:
    def __init__(self, scenario=DEFAULT_SCENARIO, seed=None, config=None):
        self.scenario = int(scenario)
        if config is None:
            config = GameConfig()
        
        self.config = config
        self._types = self.config.types_for(scenario)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is None:
            seed = Random().getrandbits(64)
        
        self.seed = int(seed)
        self._rng = Random(self.seed)
        self.dinosaur = Dinosaur()
        self._generator = ObstacleGenerator(self._rng, self.config, self._types, self.scenario)
        self.obstacles = []
        self._steps = 0
        self._survived_steps = 0
        self._distance = 0.0
        self.terminated = False
        return self.get_state()

    def get_speed(self):
        if self.scenario == 4:
            return min(SCENARIO_4_MAX_SPEED, SPEED + self._survived_steps * SIMULATION_STEP_DURATION * SCENARIO_4_ACCELERATION)
        
        return SPEED

    def get_points(self):
        return round_up(ceil(self._distance) * POINTS_PER_PIXEL)

    def update(self, action):
        if self.terminated:
            return
        
        speed = self.get_speed()
        self.dinosaur.update(int(action))
        
        for obstacle in self.obstacles:
            obstacle.update(speed)
        
        self._steps += 1
        if self._steps > CLEAR_STEPS:
            self.obstacles.extend(self._generator.advance(self.obstacles, speed))
        
        self.terminated = self.has_collision()
        if not self.terminated:
            self._distance += speed * SIMULATION_STEP_DURATION
            self._survived_steps += 1
        
        visible_obstacles = []
        for obstacle in self.obstacles:
            if obstacle.x + obstacle.width > 0:
                visible_obstacles.append(obstacle)
        
        self.obstacles = visible_obstacles

    def has_collision(self):
        dinosaur_boxes = self.dinosaur.get_collision_boxes()
        for obstacle in self.obstacles:
            for obstacle_box in obstacle.get_collision_boxes():
                for dinosaur_box in dinosaur_boxes:
                    if dinosaur_box.intersects(obstacle_box):
                        return True
        
        return False

    def get_state(self):
        obstacles = []
        for obstacle in self.obstacles:
            obstacles.append(obstacle.get_state())
        
        obstacles.sort(key=obstacle_x)

        return {
            'seed': self.seed,
            'scenario': self.scenario,
            'timestep': SIMULATION_STEP_DURATION,
            'speed': self.get_speed(),
            'dinosaur': self.dinosaur.get_state(),
            'obstacles': obstacles,
            'score': { 'points': self.get_points(), },
            'scene': { 'steps': self._steps, 'offset': self._distance },
            'terminated': self.terminated,
        }


def obstacle_x(obstacle):
    return obstacle['x']
