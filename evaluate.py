import argparse
import sys
import traceback
from pathlib import Path
from src.game.config import SCENARIOS
from src.training.config import MAX_STEPS, TEST_SEEDS
from src.training.agents import AGENTS, load_agent
from src.training.renderer import TrainingRenderer, RENDER_SPEEDS
from src.training.evaluation import evaluate
from src.logs import start_log, finish_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=AGENTS, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--scenario', type=int, choices=SCENARIOS, required=True)
    parser.add_argument('--render', nargs='?', const='8x', choices=RENDER_SPEEDS)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error('max-steps must be positive.')
    
    log = start_log(args.model, 'evaluation')
    
    try:
        print(f"Starting evaluation with {args.agent.upper()}...", flush=True)
        print(f'Model: {args.model}')
        print(f'Scenario: {args.scenario} | max steps: {args.max_steps}')
        model = load_agent(args.agent, args.model)
        viewer = TrainingRenderer(args.render) if args.render else None
        
        try:
            evaluation = evaluate(args.agent, model, args.scenario, args.max_steps, TEST_SEEDS, progress=True, viewer=viewer)
            results = evaluation['results']
            interrupted = evaluation['interrupted']
            mean = sum(r['points'] for r in results) / len(results) if results else None
            capped_maps = sum(r['reached_limit'] for r in results)
            print(f'Completed maps: {len(results)}/{len(TEST_SEEDS)}')
            print(f'Interrupted: {interrupted}')
            print(f'Maps reaching the time limit: {capped_maps}/{len(TEST_SEEDS)}')
            partial = evaluation['partial_episode']
            
            if partial is not None:
                print(f"Partial episode | seed: {partial['seed']} | score: {partial['points']} | steps: {partial['steps']}")
            
            label = 'partial mean score' if interrupted else 'mean score'
            print(f'{label}: {mean:.2f}' if mean is not None else 'No completed maps.', flush=True)
        finally:
            if viewer is not None:
                viewer.close()

        return 0
    except KeyboardInterrupt:
        traceback.print_exc()
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        finish_log(log)


if __name__ == '__main__':
    sys.exit(main())
