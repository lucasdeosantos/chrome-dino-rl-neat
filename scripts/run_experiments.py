import argparse
import subprocess
import sys
from pathlib import Path
from src.training.config import SEED
from src.game.config import SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', nargs='?', type=int, default=SEED, help='Algorithm seed; defaults to SEED in src/training/config.py.')
    args = parser.parse_args(argv)
    
    if args.seed < 0:
        parser.error('seed must be non-negative.')
    
    for agent in ('neat', 'dqn', 'ppo'):
        extension = 'pkl' if agent == 'neat' else 'zip'
        for scenario in SCENARIOS:
            folder = f'models/{agent}/s{scenario}/seed{args.seed}'
            subprocess.run([
                sys.executable, 'train.py',
                '--agent', agent, '--scenario', str(scenario),
                '--seed', str(args.seed), '--output', folder,
            ], cwd=ROOT, check=True)
            subprocess.run([
                sys.executable, 'evaluate.py',
                '--agent', agent, '--scenario', str(scenario),
                '--model', f'{folder}/best.{extension}',
            ], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()