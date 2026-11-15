from .agents import act
from .env import DinoEnv


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
