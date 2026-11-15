import time
from ..game.physics import SIMULATION_STEP_DURATION


RENDER_SPEEDS = { 'normal': 1, '2x': 2, '4x': 4, '8x': 8 }


class StopRendering(Exception):
    """Request a clean stop at a training boundary."""


class TrainingRenderer:
    def __init__(self, speed):
        import pygame
        from ..game.renderer import Renderer
        self.pygame = pygame
        self.renderer = Renderer(agent_mode=True)
        self.speed = RENDER_SPEEDS[speed]
        self.stopped = False
        self.last_state = None
        self.pending_steps = 0
        self.last_frame = time.perf_counter()

    def poll(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT or (event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE):
                self.stopped = True
        
        return not self.stopped

    def step(self, state, population=None, force=False):
        self.last_state = state
        self.pending_steps += 1
        
        if not self.poll():
            return
        
        if self.pending_steps < self.speed and not force:
            return
        
        self.renderer.draw(state, population=population)
        
        remaining = self.pending_steps * SIMULATION_STEP_DURATION / self.speed - (time.perf_counter() - self.last_frame)
        if remaining > 0:
            time.sleep(remaining)
        
        self.pending_steps = 0
        self.last_frame = time.perf_counter()
        self.poll()

    def updating(self):
        if self.last_state is not None:
            self.renderer.draw(self.last_state, status='Updating model...')
        
        self.poll()

    def close(self):
        self.renderer.close()