from .physics import GROUND_Y, DINO_X, SIMULATION_STEPS_PER_SECOND
from .physics import STANDING_WIDTH, STANDING_HEIGHT, DUCKING_WIDTH, DUCKING_HEIGHT
from .physics import DUCKING_HITBOX_OFFSET_Y, JUMP_INITIAL_VELOCITY_PER_STEP
from .physics import GRAVITY_PER_STEP, JUMP_CUTOFF_TOP_Y, JUMP_CUTOFF_VELOCITY
from .hitbox import Rect, position_hitboxes, COLLISION_OFFSET_X, COLLISION_OFFSET_Y
from .config import JUMP, DUCK
from .utils import round_up


RUN_BOXES = ((22, 0, 17, 16), (1, 18, 30, 9), (10, 35, 14, 8), (1, 24, 29, 5), (5, 30, 21, 4), (9, 34, 15, 4))
DUCK_BOXES = ((1, 18, 55, 25),)


class Dinosaur:
    def __init__(self):
        self.y = GROUND_Y
        self._velocity_frame = 0.0
        self.ducking = False

    def get_velocity_y(self):
        return self._velocity_frame * SIMULATION_STEPS_PER_SECOND

    def is_grounded(self):
        return self.y == GROUND_Y and self.get_velocity_y() == 0.0

    def get_hitbox(self):
        if self.ducking:
            return Rect(DINO_X, self.y - DUCKING_HITBOX_OFFSET_Y, DUCKING_WIDTH, DUCKING_HEIGHT)
        
        return Rect(DINO_X, self.y - STANDING_HEIGHT, STANDING_WIDTH, STANDING_HEIGHT)

    def get_sprite_box(self):
        width = STANDING_WIDTH
        if self.ducking:
            width = DUCKING_WIDTH
        
        return Rect(DINO_X, self.y - STANDING_HEIGHT, width, STANDING_HEIGHT)

    def get_collision_boxes(self):
        layout = RUN_BOXES
        if self.ducking:
            layout = DUCK_BOXES
        
        return position_hitboxes(layout, DINO_X + COLLISION_OFFSET_X, self.y - STANDING_HEIGHT + COLLISION_OFFSET_Y)

    def update(self, action):
        if self.is_grounded():
            self.ducking = action == DUCK
            if action == JUMP:
                self._velocity_frame = JUMP_INITIAL_VELOCITY_PER_STEP
        
        if not self.is_grounded():
            self.y += round_up(self._velocity_frame)
            self._velocity_frame += GRAVITY_PER_STEP
            
            if self.y - STANDING_HEIGHT < JUMP_CUTOFF_TOP_Y:
                self._velocity_frame = max(self._velocity_frame, JUMP_CUTOFF_VELOCITY)
            
            if self.y > GROUND_Y:
                self.y = GROUND_Y
                self._velocity_frame = 0.0

    def get_state(self):
        boxes = []
        for box in self.get_collision_boxes():
            boxes.append(box.to_dict())
        
        return {
            'y': self.y,
            'velocity_y': self.get_velocity_y(),
            'ducking': self.ducking,
            'grounded': self.is_grounded(),
            'hitbox': self.get_hitbox().to_dict(),
            'sprite_box': self.get_sprite_box().to_dict(),
            'collision_boxes': boxes,
        }
