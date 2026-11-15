import argparse
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from src.training import config
from src.game.config import SCENARIOS
from src.training.agents import AGENTS
from src.training.config import MAX_STEPS, TRAINING_SEED_START, TEST_SEEDS
from src.training.rewards import REWARDS
from src.training.checkpoints import resume_settings
from src.training.renderer import RENDER_SPEEDS
from src.utils import read_json, write_json
from src.logs import start_log, finish_log


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=AGENTS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scenario', type=int, choices=SCENARIOS)
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--render', nargs='?', const='8x', choices=RENDER_SPEEDS)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--timesteps', type=int, default=config.TIMESTEPS)
    parser.add_argument('--reward', choices=REWARDS, default='r1')
    parser.add_argument('--generations', type=int, default=config.GENERATIONS)
    parser.add_argument('--population', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', type=Path, metavar='FOLDER', help='Continue from the last consistent training checkpoint.')
    mode.add_argument('--pretrained', type=Path, metavar='MODEL', help='Start a new run from a model of the same algorithm.')
    
    args = parser.parse_args(argv)
    args.rollout_steps = config.ROLLOUT_STEPS
    args.batch_size = config.BATCH_SIZE
    args.learning_starts = config.LEARNING_STARTS
    args.buffer_size = config.BUFFER_SIZE
    args.neat_config = config.NEAT_CONFIG
    args.training_seed_start = TRAINING_SEED_START
    args.checkpoint_every = config.CHECKPOINT_EVERY

    if args.resume is not None:
        supplied = argv if argv is not None else sys.argv[1:]
        options = {item.split('=')[0] for item in supplied if item.startswith('--')}
        
        if options - {'--resume', '--render'}:
            parser.error('--resume keeps the original settings, only --render may be supplied with it.')
        
        resume_settings(args)
        return args

    if args.agent is None:
        parser.error('Specify --agent to train or --resume to continue a saved run.')

    if args.output is None:
        parser.error('Specify --output to start a new training run.')

    if args.scenario is None:
        parser.error('Specify --scenario to start a new training run.')
    
    if args.pretrained is not None:
        if not args.pretrained.is_file():
            parser.error('Pretrained model not found.')
        source = args.pretrained.parent
        if (source / 'run.json').exists() and read_json(source / 'run.json')['agent'] != args.agent:
            parser.error('--pretrained must use the same algorithm as --agent.')
    
    return args


def run(args):
    if args.resume is not None:
        execute_training(args)
        return
    
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('The output folder is not empty. Choose another folder to preserve the previous training run.')
    
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ('agent', 'scenario', 'seed', 'max_steps')
    metadata = {key: getattr(args, key) for key in fields}
    packages = ['gymnasium', 'numpy']
    
    if args.agent == 'neat':
        from src.training.train_neat import load_start_config
        metadata.update(
            generations=args.generations,
            population=args.population if args.population is not None else load_start_config(args).pop_size,
            neat_config='effective_neat_config.txt',
            neat_fitness='points',
            checkpoint_every=1,
        )
        packages.append('neat-python')
    else:
        metadata.update(
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            reward={name: weight for name, weight in REWARDS[args.reward].items() if weight != 0},
            checkpoint_every=args.checkpoint_every,
        )
        if args.agent == 'ppo':
            metadata['rollout_steps'] = args.rollout_steps
        else:
            metadata.update(learning_starts=args.learning_starts, buffer_size=args.buffer_size)
        packages += ['stable-baselines3', 'torch']
   
    if args.pretrained is not None:
        metadata['pretrained'] = str(args.pretrained.resolve())
    
    metadata['training_seed_start'] = args.training_seed_start
    metadata['test_seeds'] = {'start': TEST_SEEDS[0], 'end': TEST_SEEDS[-1]}
    metadata['versions'] = {name: version(name) for name in packages}
    write_json(args.output / 'run.json', metadata, compact_keys=('test_seeds',))
    execute_training(args)


def execute_training(args):
    if args.agent == 'neat':
        from src.training.train_neat import train_neat
        train_neat(args)
    else:
        from src.training.train_rl import train_rl
        train_rl(args)
    
    print('Models saved to:', args.output)


def main(argv=None):
    args = parse_args(argv)
    log = start_log(args.output, 'train')
    
    try:
        run(args)
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
