import argparse
from pathlib import Path
import pygame
from src.game.game import DinoGame
from src.game.renderer import Renderer
from src.game.config import SCENARIOS, DEFAULT_SCENARIO
from src.game.physics import SIMULATION_STEP_DURATION
from src.game.config import RUN, JUMP, DUCK, RENDER_FPS


def parse_args():
    parser = argparse.ArgumentParser(description="Chrome Dino clone")
    parser.add_argument("--scenario", type=int, choices=SCENARIOS, default=DEFAULT_SCENARIO)
    parser.add_argument("--versus", type=Path, help="Play against a saved model on the same track.")
    return parser.parse_args()


def main():
    args = parse_args()

    game = DinoGame(args.scenario)
    opponent = None
    
    if args.versus:
        from src.training.agents import Opponent
        opponent = Opponent(args.versus, args.scenario, game.seed)
    
    renderer = Renderer()
    renderer.ready = True
    clock = pygame.time.Clock()
    accumulator = 0.0
    pending_jump = False
    running = True
    
    try:
        while running:
            accumulator += clock.tick(RENDER_FPS) / 1000.0
            finished = game.terminated and (opponent is None or opponent.env.game.terminated)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif renderer.ready and event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_UP):
                        renderer.ready = False
                        pending_jump = event.key in (pygame.K_SPACE, pygame.K_UP)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE) and finished:
                        game.reset(seed=game.seed + 1)
                        
                        if opponent is not None:
                            opponent.reset(game.seed)
                        
                        renderer.ready = False
                        accumulator = 0.0
                        pending_jump = False
                    elif event.key in (pygame.K_SPACE, pygame.K_UP):
                        pending_jump = True
            
            if not running:
                break
            
            keys = pygame.key.get_pressed()
            
            if renderer.ready:
                accumulator = 0.0
                pending_jump = False
            
            while accumulator >= SIMULATION_STEP_DURATION:
                jump = pending_jump or keys[pygame.K_SPACE] or keys[pygame.K_UP]
                action = RUN
                
                if jump:
                    action = JUMP
                elif keys[pygame.K_DOWN]:
                    action = DUCK
                
                game.update(action)
                
                if opponent is not None:
                    opponent.update()
                
                pending_jump = False
                accumulator -= SIMULATION_STEP_DURATION
            
            if opponent is None:
                renderer.draw(game.get_state())
            else:
                renderer.draw(game.get_state(), opponent.env.game.get_state())
    finally:
        renderer.close()
        if opponent is not None:
            opponent.close()


if __name__ == "__main__":
    main()
