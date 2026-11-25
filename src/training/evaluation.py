import os
from pathlib import Path
from statistics import mean, median
from ..utils import write_json
from .agents import act
from .env import DinoEnv


ROOT = Path(__file__).resolve().parents[2]


def evaluate(kind, model, scenario, max_steps, seeds, progress=False, viewer=None, include_steps=False):
    results = []
    partial_episode = None
    env = DinoEnv(scenario=scenario, max_steps=max_steps)
    
    try:
        for seed in seeds:
            if viewer is not None and not viewer.poll():
                break
            
            observation, _ = env.reset(seed=int(seed))
            while True:
                action = act(kind, model, observation)
                observation, _, terminated, truncated, info = env.step(action)
                
                if viewer is not None:
                    viewer.step(env.game.get_state(), force=terminated or truncated)
                
                if terminated or truncated:
                    result = { 'seed': int(seed), **info }
                    if include_steps:
                        result['steps'] = env._elapsed_steps
                    results.append(result)
                    if progress:
                        print(f"seed: {int(seed)} | score: {info['points']}", flush=True)
                    break
                
                if viewer is not None and viewer.stopped:
                    partial_episode = { 'seed': int(seed), 'steps': env._elapsed_steps, **info }
                    break
            
            if viewer is not None and viewer.stopped:
                break
    finally:
        env.close()
    
    return { 'results': results, 'partial_episode': partial_episode, 'interrupted': viewer is not None and viewer.stopped }


def evaluation_directory(model_path):
    folder = Path(model_path).resolve().parent
    
    try:
        relative = folder.relative_to(ROOT / 'models')
    except ValueError:
        relative = Path('external') / folder.name / Path(model_path).stem
    
    return ROOT / 'results' / 'evaluation' / relative


def save_evaluation(agent, model_path, scenario, max_steps, seeds, result):
    folder = evaluation_directory(model_path)
    folder.mkdir(parents=True, exist_ok=True)
    
    episodes = result['results']
    scores = [episode['points'] for episode in episodes]
    summary = {
        'model': os.path.relpath(Path(model_path).resolve(), ROOT).replace(os.sep, '/'),
        'agent': agent,
        'scenario': scenario,
        'max_steps': max_steps,
        'expected_maps': len(seeds),
        'completed_maps': len(episodes),
        'maps_reaching_limit': sum(episode['reached_limit'] for episode in episodes),
        'mean_score': mean(scores) if scores else None,
        'median_score': median(scores) if scores else None,
        'min_score': min(scores) if scores else None,
        'max_score': max(scores) if scores else None,
        'stop_reason': 'interrupted' if result['interrupted'] else 'maps_completed',
        'partial_episode': result['partial_episode'],
    }
    
    if not result['interrupted'] and len(episodes) != len(seeds):
        summary['stop_reason'] = 'incomplete'
    
    write_json(folder / 'evaluation_episodes.json', episodes)
    write_json(folder / 'evaluation_result.json', summary)
    return folder

