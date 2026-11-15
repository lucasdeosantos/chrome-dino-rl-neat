import gymnasium as gym
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .agents import rl_class
from .env import DinoEnv
from .checkpoints import checkpoint_folder, read_checkpoint, save_checkpoint, restore_records
from .renderer import TrainingRenderer, StopRendering
from .agents import load_agent
from ..utils import write_json


class RenderedEnv(gym.Wrapper):
    def __init__(self, env, viewer):
        super().__init__(env)
        self.viewer = viewer

    def step(self, action):
        result = self.env.step(action)
        self.viewer.step(self.unwrapped.game.get_state(), force=result[2] or result[3])
        return result


class TrainingCallback(BaseCallback):
    def __init__(self, args, env, viewer):
        super().__init__()
        self.args = args
        self.env = env
        self.viewer = viewer
        self.history = []
        self.best_score = float('-inf')
        self.best_reached_limit = False
        self.success = False

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                self.record_episode(info)
        
        return self.viewer is None or not self.viewer.stopped

    def record_episode(self, info):
        record = { 'position': self.num_timesteps, 'seed': self.env.next_seed - 2,
                  'points': info['points'], 'reached_limit': info['reached_limit'] }
        self.history.append(record)
        write_json(self.args.output / 'training_episodes.json', self.history)
        self.success = self.success or info['reached_limit']
        
        if (info['reached_limit'], info['points']) >= (self.best_reached_limit, self.best_score):
            self.best_score = info['points']
            self.best_reached_limit = info['reached_limit']
            self.model.save(str(self.args.output / 'best.zip'))
            write_json(self.args.output / 'best_training.json', record)
        
        print(f"{self.num_timesteps} | seed: {record['seed']} | score: {info['points']}", flush=True)



def create_model(args, env, viewer):
    settings = {'policy_kwargs': {'net_arch': [64, 64]}, 'seed': args.seed, 'device': 'cpu', 'verbose': 0}
    
    if args.agent == 'ppo':
        settings.update(n_steps=args.rollout_steps, batch_size=args.batch_size)
    else:
        settings.update(buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                        batch_size=args.batch_size, train_freq=4, gradient_steps=1)
    
    model_class = rl_class(args.agent)

    class TrainingModel(model_class):
        checkpoint_callback = None

        def _excluded_save_params(self):
            return super()._excluded_save_params() + ['checkpoint_callback']

        def train(self, *train_args, **train_kwargs):
            if viewer is not None:
                if args.agent == 'ppo':
                    viewer.updating()
                if not viewer.poll():
                    raise StopRendering()
            
            super().train(*train_args, **train_kwargs)
            
            if self.checkpoint_callback is not None:
                self.checkpoint_callback()
            
            if viewer is not None and not viewer.poll():
                raise StopRendering()

    if args.resume is not None:
        folder = checkpoint_folder(args.output)
        model = TrainingModel.load(str(folder / 'model.zip'), env=env, device='cpu')
        
        if args.agent == 'dqn':
            model.load_replay_buffer(str(folder / 'replay_buffer.pkl'))
        
        return model
    
    if args.pretrained is not None:
        source = load_agent(args.agent, args.pretrained)
        settings['policy_kwargs'] = source.policy_kwargs
        model = TrainingModel(source.policy_class, env, **settings)
        model.policy.load_state_dict(source.policy.state_dict())
        return model
    
    return TrainingModel('MlpPolicy', env, **settings)


def save_results(args, model, env, callback, interrupted):
    if not callback.history and callback.best_score == float('-inf'):
        callback.best_score = env.game.get_points()
        model.save(str(args.output / 'best.zip'))
    
    model.save(str(args.output / 'last.zip'))
    if args.agent == 'dqn':
        model.save_replay_buffer(str(args.output / 'replay_buffer.pkl'))
    
    partial_episode = None
    if interrupted and env._elapsed_steps:
        partial_episode = {'seed': env.next_seed - 1, 'steps': env._elapsed_steps,
                           'points': env.game.get_points()}
    
    write_json(args.output / 'training_episodes.json', callback.history)
    write_json(args.output / 'training_result.json',
               {'agent': args.agent, 'completed_steps': model.num_timesteps,
                'best_points': callback.best_score, 'success': callback.success,
                'stop_reason': 'interrupted' if interrupted else 'step_limit',
                'interrupted': interrupted, 'partial_episode': partial_episode})


def train_rl(args):
    print(f"Starting training with {args.agent.upper()}...", flush=True)
    print(f'Scenario: {args.scenario} | max steps: {args.max_steps}', flush=True)
    torch.set_num_threads(1)
    viewer = TrainingRenderer(args.render) if args.render else None
    state = read_checkpoint(args.output) if args.resume is not None else None
    seed = state['restart_seed'] if state is not None else args.training_seed_start
    raw_env = DinoEnv(args.scenario, max_steps=args.max_steps, sequential_seed=seed, reward=args.reward)
    env = raw_env
    try:
        if viewer is not None:
            env = RenderedEnv(env, viewer)
       
        env = Monitor(env)
        model = create_model(args, env, viewer)
        callback = TrainingCallback(args, raw_env, viewer)
        
        if state is not None:
            for name in ('history', 'best_score', 'best_reached_limit', 'success'):
                setattr(callback, name, state[name])
            
            restore_records(args, state)
            print(f'Resuming at step {model.num_timesteps} | map seed: {seed}', flush=True)
        
        checkpoint_step = model.num_timesteps

        def checkpoint(force=False):
            nonlocal checkpoint_step
            if not force and model.num_timesteps - checkpoint_step < args.checkpoint_every and model.num_timesteps < args.timesteps:
                return

            restart_seed = raw_env.next_seed - 1 if model._last_obs is not None else raw_env.next_seed
            progress = {name: getattr(callback, name) for name in
                        ('history', 'best_score', 'best_reached_limit', 'success')}
            progress.update(completed_steps=model.num_timesteps, restart_seed=restart_seed)
            save_checkpoint(args, progress, model=model)
            checkpoint_step = model.num_timesteps

        model.checkpoint_callback = checkpoint
        
        if state is None:
            checkpoint(force=True)
        
        remaining = args.timesteps - model.num_timesteps
        
        try:
            if remaining > 0:
                model.learn(total_timesteps=remaining, callback=callback,
                            reset_num_timesteps=args.resume is None)
        except StopRendering:
            pass
        
        interrupted = viewer is not None and viewer.stopped
        save_results(args, model, raw_env, callback, interrupted)
        
        if not interrupted and remaining > 0:
            checkpoint(force=True)
        
        print(f'Completed steps: {model.num_timesteps}/{args.timesteps}')
        print(f'Interrupted: {interrupted}', flush=True)
        
        if interrupted:
            print(f'Resumable checkpoint: step {checkpoint_step}', flush=True)
        
        return model
    finally:
        env.close()
        if viewer is not None:
            viewer.close()
