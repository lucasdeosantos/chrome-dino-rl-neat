import pickle
from pathlib import Path
import numpy as np


AGENTS = ('ppo', 'dqn', 'neat')


def rl_class(kind):
    from stable_baselines3 import PPO, DQN
    return { 'ppo': PPO, 'dqn': DQN }[kind]


def neat_network(genome, config):
    import neat
    return neat.nn.FeedForwardNetwork.create(genome, config)


def act(kind, model, observation):
    if kind == 'neat':
        return int(np.argmax(model.activate(observation)))
    
    action, _ = model.predict(observation, deterministic=True)
    return int(action)


def save_neat(genome, config, path):
    with Path(path).open('wb') as file:
        pickle.dump({'genome': genome, 'config': config}, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_agent(kind, path):
    if kind == 'neat':
        with Path(path).open('rb') as file:
            data = pickle.load(file)
        
        config = data['config']
        return neat_network(data['genome'], config)
    
    model = rl_class(kind).load(str(path), device='cpu')
    return model


class Opponent:
    def __init__(self, path, scenario, seed):
        from .env import DinoEnv
        from ..utils import read_json

        folder = path.parent
        metadata_path = folder / 'run.json'
        self.kind = read_json(metadata_path)['agent']
        self.model = load_agent(self.kind, path)
        self.env = DinoEnv(scenario=scenario)
        self.reset(seed)

    def reset(self, seed):
        self.env.reset(seed=seed)
        self.env.game.reset(seed=seed)

    def update(self):
        if not self.env.game.terminated:
            action = act(self.kind, self.model, self.env.get_observation())
            self.env.step(action)

    def close(self):
        self.env.close()
