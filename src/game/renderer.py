from pathlib import Path
import pygame
from .physics import SIMULATION_STEP_DURATION, DINO_X, STANDING_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT


ASSET_DIR = Path(__file__).resolve().parents[2] / 'assets' / 'reference'
SCREEN_SIZE = (1100, 600)
SCALE = 1.8
ORIGIN = (10, 150)
GROUND_SINK = 5
DINO_IDLE_SPRITE = (848, 2, 44, 47)
DINO_DEAD_SPRITE = (1068, 2, 44, 47)
DINO_RUN_SPRITES = ((936, 2, 44, 47), (980, 2, 44, 47))
DINO_DUCK_SPRITES = ((1112, 2, 59, 47), (1171, 2, 59, 47))
BIRD_SPRITES = ((134, 2, 46, 40), (180, 2, 46, 40))
CACTUS_SPRITES = { 'small': (228, 2, 17, 35), 'large': (332, 2, 25, 50) }
GROUND_SPRITE = (2, 54, 600, 12)
CLOUD_SPRITE = (86, 2, 46, 14)
GAME_OVER_SPRITE = (655, 15, 191, 11)
RESTART_SPRITE = (2, 2, 36, 32)
DIGIT_SPRITE = (655, 2, 10, 13)
RUN_FRAME_STEPS = 5
DUCK_ANIMATION_FPS = 8
BIRD_FRAME_STEPS = 10
GROUND_DRAW_Y = 136
DINO_DRAW_OFFSET_Y = 2
CACTUS_DRAW_OFFSETS = { 'small': (1, 1, 1), 'large': (3, 3, 2) }
SCORE_SOUND_INTERVAL = 100
SCORE_DIGIT_SPACING = 11
SCORE_RIGHT_MARGIN = 5
SOUND_VOLUME = 0.25
BACKGROUND_COLOR = (247, 247, 247)
AGENT_COLOR = (40, 140, 230)
AGENT_ALPHA = 150
VERSUS_RESULT_HEIGHT = 15
VERSUS_RESULT_Y = 42


class Renderer:
    def __init__(self, debug=False, agent_mode=False):
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Chrome-Dino | experimental environment')
        self.scene = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
        self.sheet = pygame.image.load(str(ASSET_DIR / 'sprite-rgba.png')).convert_alpha()
        self.versus_results = {}
        
        for result, filename in [('you-win', 'YouWin.png'), ('you-lose', 'YouLose.png'), ('draw', 'Draw.png')]:
            image = pygame.image.load(str(ASSET_DIR.parent / 'versus' / filename)).convert_alpha()
            image = image.subsurface(image.get_bounding_rect(min_alpha=128))
            width = round(image.get_width() * VERSUS_RESULT_HEIGHT / image.get_height())
            self.versus_results[result] = pygame.transform.scale(image, (width, VERSUS_RESULT_HEIGHT))
        
        self.font = pygame.font.Font(None, 23)
        self.agent_mode = agent_mode
        self.debug = debug
        self.ready = False
        self.sounds = {}
        
        try:
            pygame.mixer.init()
            for key, name in [('jump', 'press'), ('hit', 'hit'), ('score', 'reached')]:
                self.sounds[key] = pygame.mixer.Sound(str(ASSET_DIR / f'offline-sound-{name}.ogg'))
                self.sounds[key].set_volume(SOUND_VOLUME)
        except (pygame.error, FileNotFoundError):
            self.sounds = {}
        
        self._last_state = None

    def blit(self, source, position):
        self.scene.blit(self.sheet, position, source)

    def text(self, text, position, color=(100, 100, 100), centered=False):
        surface = self.font.render(text, True, color)
        
        if centered:
            rect = surface.get_rect(center=position)
        else:
            rect = surface.get_rect(topleft=position)
        
        self.screen.blit(surface, rect)

    def draw_boxes(self, boxes):
        if self.debug:
            for box in boxes:
                pygame.draw.rect(self.scene, (210, 40, 70), pygame.Rect(box['x'], box['y'], box['width'], box['height']), 1)

    def play_events(self, state):
        old = self._last_state
        if old and state['scene']['steps'] > old['scene']['steps']:
            event = None
            if state['terminated'] and not old['terminated']:
                event = 'hit'
            elif old['dinosaur']['grounded'] and state['dinosaur']['velocity_y'] < 0:
                event = 'jump'
            elif state['score']['points'] // SCORE_SOUND_INTERVAL > old['score']['points'] // SCORE_SOUND_INTERVAL:
                event = 'score'
            if event in self.sounds:
                self.sounds[event].play()
        
        self._last_state = state

    def draw_score(self, target, points, position, scale=1, tinted=False):
        digits = f'{points:05d}'
        base, sprite_y, width, height = DIGIT_SPRITE
        
        counter = pygame.Surface((len(digits) * SCORE_DIGIT_SPACING, height), pygame.SRCALPHA)
        for i, digit in enumerate(digits):
            counter.blit(self.sheet, (i * SCORE_DIGIT_SPACING, 0), (base + int(digit) * width, sprite_y, width, height))
        
        if tinted:
            counter.fill(AGENT_COLOR, special_flags=pygame.BLEND_RGB_MAX)
            counter.set_alpha(AGENT_ALPHA)
        
        counter = pygame.transform.scale(counter, (round(counter.get_width() * scale), round(height * scale)))
        target.blit(counter, position)

    def draw_dinosaur(self, state, tinted=False, offset=0):
        steps = state['scene']['steps']
        d = state['dinosaur']
        
        if state['terminated']:
            source = DINO_DEAD_SPRITE
        elif d['ducking']:
            frame = int(steps * SIMULATION_STEP_DURATION * DUCK_ANIMATION_FPS) % len(DINO_DUCK_SPRITES)
            source = DINO_DUCK_SPRITES[frame]
        elif not d['grounded'] or self.ready:
            source = DINO_IDLE_SPRITE
        else:
            frame = steps // RUN_FRAME_STEPS % len(DINO_RUN_SPRITES)
            source = DINO_RUN_SPRITES[frame]

        x = DINO_X - offset
        if x + source[2] <= 0:
            return
        
        position = (round(x), round(d['y'] - STANDING_HEIGHT) + DINO_DRAW_OFFSET_Y + GROUND_SINK)
        
        if tinted:
            sprite = self.sheet.subsurface(source).copy()
            sprite.fill(AGENT_COLOR, special_flags=pygame.BLEND_RGB_MAX)
            sprite.set_alpha(AGENT_ALPHA)
            self.scene.blit(sprite, position)
        else:
            self.blit(source, position)
        
        boxes = [dict(box, x=box['x'] - offset) for box in d['collision_boxes']]
        self.draw_boxes(boxes)

    def draw(self, state, opponent=None, population=None, status=None):
        self.screen.fill(BACKGROUND_COLOR)
        self.scene.fill(BACKGROUND_COLOR)
        score = state['score']
        world = state
        
        if opponent is not None and opponent['scene']['steps'] > state['scene']['steps']:
            world = opponent
        
        finished = state['terminated'] and (opponent is None or opponent['terminated'])

        for start, y in ((300, 25), (650, 45)):
            x = (start - world['scene']['offset'] / 300 + 50) % 750 - 50
            self.blit(CLOUD_SPRITE, (round(x), y))
        
        offset = round(world['scene']['offset']) % WORLD_WIDTH
        self.blit(GROUND_SPRITE, (-offset, GROUND_DRAW_Y))
        self.blit(GROUND_SPRITE, (WORLD_WIDTH - offset, GROUND_DRAW_Y))
        
        if population is not None:
            for individual in population:
                if not individual['terminated']:
                    self.draw_dinosaur(individual)
        else:
            for dinosaur_state, tinted in ((state, False), (opponent, True)):
                if dinosaur_state is None or (self.agent_mode and dinosaur_state['terminated']):
                    continue

                displacement = 0
                if dinosaur_state['terminated'] and dinosaur_state['scene']['steps'] < world['scene']['steps']:
                    displacement = world['scene']['offset'] - dinosaur_state['scene']['offset']
                
                self.draw_dinosaur(dinosaur_state, tinted=tinted, offset=displacement)
        
        for obstacle in world['obstacles']:
            variant, count = obstacle['variant'], obstacle['count']
            draw_y = round(obstacle['y'])
            
            if variant == 'bird':
                frame = obstacle['age_steps'] // BIRD_FRAME_STEPS % len(BIRD_SPRITES)
                source = BIRD_SPRITES[frame]
            else:
                base, sprite_y, width, height = CACTUS_SPRITES[variant]
                source = (base + width * count * (count - 1) // 2, sprite_y, width * count, height)
                draw_y += CACTUS_DRAW_OFFSETS[variant][count - 1] + GROUND_SINK
            
            self.blit(source, (round(obstacle['x']), draw_y))
            self.draw_boxes(obstacle['collision_boxes'])
        
        if opponent is None:
            counter_width = len(f"{score['points']:05d}") * SCORE_DIGIT_SPACING
            score_x = WORLD_WIDTH - SCORE_RIGHT_MARGIN - counter_width
            self.draw_score(self.scene, score['points'], (score_x, 5))
        
        if finished:
            if opponent is None:
                self.blit(GAME_OVER_SPRITE, (205, 42))
            else:
                player_steps = state['scene']['steps']
                agent_steps = opponent['scene']['steps']
                if player_steps > agent_steps:
                    result = 'you-win'
                elif player_steps < agent_steps:
                    result = 'you-lose'
                else:
                    result = 'draw'
                image = self.versus_results[result]
                self.scene.blit(image, ((WORLD_WIDTH - image.get_width()) // 2, VERSUS_RESULT_Y))
            
            if not self.agent_mode:
                self.blit(RESTART_SPRITE, (282, 75))
        
        self.screen.blit(pygame.transform.scale(self.scene, (int(WORLD_WIDTH * SCALE), int(WORLD_HEIGHT * SCALE))), ORIGIN)
        self.text(f"SCENARIO {state['scenario']}", (30, 35), (130, 130, 130))
        
        if opponent is not None:
            score_y = ORIGIN[1] + round(5 * SCALE)
            player_points = score['points']
            counter_width = round(len(f'{player_points:05d}') * SCORE_DIGIT_SPACING * SCALE)
            player_x = self.screen.get_width() - 30 - counter_width
            self.draw_score(self.screen, player_points, (player_x, score_y), SCALE)
            agent_points = opponent['score']['points']
            self.draw_score(self.screen, agent_points, (30, score_y), SCALE, tinted=True)
        
        if status:
            self.text(status, (550, 505), centered=True)
        elif self.ready and not self.agent_mode:
            self.text('SPACE / ENTER to start', (550, 505), centered=True)
        elif finished and not self.agent_mode:
            self.text('SPACE / ENTER to restart', (550, 505), centered=True)
        
        self.play_events(state)
        pygame.display.flip()

    def close(self):
        pygame.quit()
