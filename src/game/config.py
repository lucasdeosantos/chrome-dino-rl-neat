RUN = 0
JUMP = 1
DUCK = 2
NUM_ACTIONS = 3

SCENARIOS = (1, 2, 3, 4)
DEFAULT_SCENARIO = 4

RENDER_FPS = 120
POINTS_PER_PIXEL = 0.025

SCENARIO_4_ACCELERATION = 3.6
SCENARIO_4_MAX_SPEED = 780.0
BIRD_MIN_SPEED = 510.0
CLEAR_STEPS = 180


class GameConfig:
    def __init__(self, gap_min=None, gap_max=None, obstacle_types=None):
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.obstacle_types = obstacle_types

    def types_for(self, scenario):
        if scenario == 1:
            selected = ('cactus',)
        elif scenario == 2:
            selected = ('bird',)
        else:
            selected = ('cactus', 'bird')
        
        return selected
