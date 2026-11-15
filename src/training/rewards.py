from ..game.config import JUMP, DUCK
from ..game.dinosaur import Dinosaur
from ..game.physics import DINO_X


REWARDS = {
    'r1': { 'steps': 0.1, 'collision': -5, 'jump_or_duck': -0.01 },
    'r2': { 'steps': 0.1 },
    'r3': { 'steps': 0.1, 'collision': -5, 'jump': -0.01 },
    'r4': { 'steps': 0.1, 'collision': -5 },
    'r5': { 'steps': 0.1, 'collision': -5, 'obstaclePassed': 1 },
    'r6': { 'steps': 0.1, 'collision': -5, 'successfulDuck': 1 },
    'r7': { 'steps': 0.1, 'collision': -5, 'successfulDuck': 1, 'unnecessaryJump': -0.05 },
    'r8': { 'steps': 0.1, 'collision': -5, 'successfulDuck': 1, 'successfulJump': 0.5 },
}


class RewardCalculator:
    def __init__(self, reward='r1'):
        self.weights = dict(reward) if isinstance(reward, dict) else REWARDS[reward]
        self.track_obstacles = any(name in self.weights for name in ('obstaclePassed', 'successfulDuck', 'successfulJump', 'unnecessaryJump'))
        self.standing_boxes = Dinosaur().get_collision_boxes()
        self.reset()

    def reset(self):
        self.grounded = True
        self.useful_jump = False
        self.ducked = set()
        self.jumped = set()
        self.obstacles = []

    def before_step(self, game):
        if self.track_obstacles:
            self.obstacles = [obstacle for obstacle in game.obstacles if obstacle.x + obstacle.width > DINO_X]

    def calculate(self, game, action):
        if game.terminated:
            return float(self.weights.get('collision', 0))
        
        events = {'steps': 1, 'jump_or_duck': int(action in (JUMP, DUCK)), 'jump': int(action == JUMP)}
        
        if self.track_obstacles:
            dino = game.dinosaur
            
            grounded = dino.is_grounded()
            if self.grounded and not grounded:
                self.useful_jump = False
            
            events.update(obstaclePassed=0, successfulDuck=0, successfulJump=0, unnecessaryJump=0)
            
            for obstacle in self.obstacles:
                would_collide = any(a.intersects(b) for a in self.standing_boxes for b in obstacle.get_collision_boxes())
                if would_collide:
                    if grounded and dino.ducking and obstacle.kind == 'bird':
                        self.ducked.add(obstacle.id)
                    elif not grounded:
                        self.jumped.add(obstacle.id)
                        self.useful_jump = True
                
                if obstacle.x + obstacle.width <= DINO_X:
                    events['obstaclePassed'] += 1
                    events['successfulDuck'] += int(obstacle.id in self.ducked)
                    jumped = obstacle.id in self.jumped
                    events['successfulJump'] += int(jumped)
                    self.ducked.discard(obstacle.id)
                    self.jumped.discard(obstacle.id)
            
            if not self.grounded and grounded:
                events['unnecessaryJump'] = int(not self.useful_jump)
            
            self.grounded = grounded
        
        return sum(weight * events.get(name, 0) for name, weight in self.weights.items())