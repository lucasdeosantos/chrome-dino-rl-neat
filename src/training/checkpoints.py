import os
import pickle
import random
import torch
import numpy as np
from pathlib import Path
from shutil import copyfile
from ..utils import read_json, write_json


def checkpoint_folder(output):
    root = Path(output) / 'checkpoint'
    marker = root / 'current.json'
    return root / str(read_json(marker)['slot'])


def read_checkpoint(output):
    folder = checkpoint_folder(output)
    with (folder / 'state.pkl').open('rb') as file:
        state = pickle.load(file)
    
    return state


def save_checkpoint(args, state, model=None):
    root = args.output / 'checkpoint'
    root.mkdir(exist_ok=True)
    marker = root / 'current.json'
    slot = 1 - read_json(marker)['slot'] if marker.exists() else 0
    folder = root / str(slot)
    folder.mkdir(exist_ok=True)
    state = dict(state, agent=args.agent, random_state=random.getstate())
    extension = 'pkl' if args.agent == 'neat' else 'zip'
    best_path = args.output / f'best.{extension}'
    state['has_best'] = best_path.exists()
    
    if state['has_best']:
        copyfile(best_path, folder / f'best.{extension}')
    
    if args.agent == 'neat':
        last_path = args.output / 'last.pkl'
        state['has_last'] = last_path.is_file()
        if state['has_last']:
            copyfile(last_path, folder / 'last.pkl')
    
    record = args.output / 'best_training.json'
    state['best_training'] = read_json(record) if record.exists() else None
    
    if model is not None:
        state['numpy_state'] = np.random.get_state()
        state['torch_state'] = torch.get_rng_state()
        model.save(str(folder / 'model.zip'))
        if args.agent == 'dqn':
            model.save_replay_buffer(str(folder / 'replay_buffer.pkl'))
    
    with (folder / 'state.pkl').open('wb') as file:
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    write_json(root / 'next.json', {'slot': slot})
    os.replace(root / 'next.json', marker)


def restore_records(args, state):
    folder = checkpoint_folder(args.output)
    write_json(args.output / 'training_episodes.json', state['history'])
    extension = 'pkl' if args.agent == 'neat' else 'zip'
    
    if state['has_best']:
        copyfile(folder / f'best.{extension}', args.output / f'best.{extension}')
    else:
        (args.output / f'best.{extension}').unlink(missing_ok=True)
    
    if args.agent == 'neat':
        if state['has_last']:
            copyfile(folder / 'last.pkl', args.output / 'last.pkl')
        else:
            (args.output / 'last.pkl').unlink(missing_ok=True)
    
    if state['best_training'] is not None:
        write_json(args.output / 'best_training.json', state['best_training'])
    else:
        (args.output / 'best_training.json').unlink(missing_ok=True)
    
    random.setstate(state['random_state'])
    
    if args.agent != 'neat':
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])


def resume_settings(args):
    metadata = read_json(args.resume / 'run.json')
    args.output = args.resume
    fields = ('agent', 'scenario', 'seed', 'max_steps',
              'timesteps', 'batch_size', 'rollout_steps', 'learning_starts',
              'buffer_size', 'generations', 'population', 'training_seed_start',
              'checkpoint_every')
    
    for name in fields:
        if name in metadata:
            setattr(args, name, metadata[name])
    
    if args.agent == 'neat':
        args.neat_config = args.output / metadata['neat_config']
    else:
        args.reward = metadata['reward']
    
    checkpoint_folder(args.output)
