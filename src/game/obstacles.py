from math import ceil
from .config import DEFAULT_SCENARIO, BIRD_MIN_SPEED, SCENARIO_4_MAX_SPEED, SCENARIO_4_ACCELERATION
from .physics import SIMULATION_STEP_DURATION, DINO_X, DUCKING_WIDTH, WORLD_WIDTH
from .hitbox import Rect, position_hitboxes, COLLISION_OFFSET_X, COLLISION_OFFSET_Y
from .utils import round_up


BIRD_SPEED_OFFSET = 48.0
SAFE_TRANSITION_TIME = 0.50
MAX_CACTUS_GROUP_SIZE = 3
MAX_GAP_FACTOR = 1.5
VARIANTS = {
    'small': {
        'width': 17, 
        'height': 35, 
        'positions': (105,),
        'group_speed': 240,
        'boxes': ((0, 7, 5, 27), (4, 0, 6, 34), (10, 4, 7, 14)),
    },
    'large': {
        'width': 25, 
        'height': 50, 
        'positions': (90,),
        'group_speed': 420,
        'boxes': ((0, 12, 7, 38), (8, 0, 7, 49), (13, 10, 10, 38)),
    },
    'bird': {
        'width': 46, 
        'height': 40, 
        'positions': (100, 75, 50, 25),
        'group_speed': 99999,
        'boxes': ((15, 15, 16, 5), (18, 21, 24, 6), (2, 14, 4, 3), (6, 10, 4, 7), (10, 8, 6, 9)),
    },
}


class Obstacle:
    def __init__(self, id, x, y, width, height, gap=0, age_steps=0):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.gap = gap
        self.age_steps = age_steps
        self.count = 1
        self.speed_offset = 0.0

    def get_hitbox(self):
        return Rect(self.x, self.y, self.width, self.height)

    def get_collision_boxes(self):
        return position_hitboxes(self.boxes, self.x + COLLISION_OFFSET_X, self.y + COLLISION_OFFSET_Y)

    def update(self, speed):
        self.x -= (speed + self.speed_offset) * SIMULATION_STEP_DURATION
        self.age_steps += 1


    def get_state(self):
        boxes = []
        for box in self.get_collision_boxes():
            boxes.append(box.to_dict())
        
        return {
            'id': self.id,
            'variant': self.variant,
            'x': self.x,
            'y': self.y,
            'count': self.count,
            'speed_offset': self.speed_offset,
            'gap': self.gap,
            'age_steps': self.age_steps,
            'kind': self.kind,
            'width': self.width,
            'height': self.height,
            'collision_boxes': boxes,
        }


class Cactus(Obstacle):
    def __init__(self, id, variant, x, y=None, count=1, gap=0, age_steps=0):
        data = VARIANTS[variant]
        if y is None:
            y = data['positions'][0]
        super().__init__(id, x, y, data['width'] * count, data['height'], gap, age_steps)
        self.kind = 'cactus'
        self.variant = variant
        self.count = count
        self.boxes = list(data['boxes'])
        
        if count > 1:
            x, y, _, height = self.boxes[1]
            self.boxes[1] = (x, y, self.width - self.boxes[0][2] - self.boxes[2][2], height)
            _, y, width, height = self.boxes[2]
            self.boxes[2] = (self.width - width, y, width, height)


class Bird(Obstacle):
    def __init__(self, id, x, y, speed_offset=0.0, gap=0, age_steps=0):
        data = VARIANTS['bird']
        super().__init__(id, x, y, data['width'], data['height'], gap, age_steps)
        self.kind = 'bird'
        self.variant = 'bird'
        self.speed_offset = speed_offset
        self.boxes = data['boxes']


class ObstacleGenerator:
    def __init__(self, rng, config, types, scenario=DEFAULT_SCENARIO):
        self.rng = rng
        self.config = config
        self.types = types
        self.scenario = scenario
        self.next_id = 0
        self.history = []
        self.pending = None
        self.previous = None

    def create(self, speed):
        candidates = []

        if 'cactus' in self.types:
            candidates += ['small', 'large']
        
        if 'bird' in self.types and (self.scenario != 4 or speed >= BIRD_MIN_SPEED):
            candidates.append('bird')

        choices = []
        for variant in candidates:
            if self.history[-2:] != [variant, variant]:
                choices.append(variant)
        
        if not choices:
            choices = candidates
        
        variant = self.rng.choice(choices)
        data = VARIANTS[variant]
        width = data['width']
        heights = data['positions']
        multiple_speed = data['group_speed']
        
        count = 1
        if speed >= multiple_speed:
            count = self.rng.randint(1, MAX_CACTUS_GROUP_SIZE)
        
        offset = 0.0
        if variant == 'bird':
            offset = self.rng.choice((-BIRD_SPEED_OFFSET, BIRD_SPEED_OFFSET))
        
        y = self.rng.choice(heights)
        if variant == 'bird':
            obstacle = Bird(self.next_id, WORLD_WIDTH + width, y, offset)
        else:
            obstacle = Cactus(self.next_id, variant, WORLD_WIDTH + width, y, count)
        
        return obstacle

    def choose_gap(self, previous, following, speed):
        safe = self.safety_gap(previous, following, speed)
        if self.config.gap_min is not None:
            low = max(safe, self.config.gap_min)
            high = max(low, self.config.gap_max)
        else:
            low = safe
            high = round_up(low * MAX_GAP_FACTOR)
        
        return self.rng.randint(low, high)

    def safety_gap(self, previous, following, speed):
        travel_time = (following.x + following.width - DINO_X) / (speed - BIRD_SPEED_OFFSET)
        future_speed = speed
        
        if self.scenario == 4:
            future_speed = min(SCENARIO_4_MAX_SPEED, speed + SCENARIO_4_ACCELERATION * travel_time)
        
        speed_difference = max(0, following.speed_offset - previous.speed_offset)
        closing_distance = speed_difference * travel_time
        jump_distance = (future_speed + max(0, following.speed_offset)) * SAFE_TRANSITION_TIME
        dinosaur_width = DUCKING_WIDTH
        return ceil(jump_distance + dinosaur_width + closing_distance)

    def advance(self, obstacles, speed):
        if self.pending is None:
            self.pending = self.create(speed)
            if self.pending is not None and self.previous is not None:
                self.previous.gap = self.choose_gap(self.previous, self.pending, speed)
        
        if self.pending is None:
            return []
        
        if self.previous is not None:
            if self.previous not in obstacles:
                self.previous.update(speed)
            
            gap = max(self.previous.gap, self.safety_gap(self.previous, self.pending, speed))
            if self.previous.get_hitbox().right + gap >= self.pending.x:
                return []
        
        result = self.pending
        self.pending = None
        self.previous = result
        self.next_id += 1
        self.history = (self.history + [result.variant])[-2:]
        return [result]
