import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.game.config import SCENARIOS
from src.training.rewards import REWARDS

TIMESTEPS = 1000000


def main():
    for scenario in SCENARIOS:
        for agent in ('dqn', 'ppo'):
            for reward in REWARDS:
                subprocess.run([
                    sys.executable, 'train.py', '--agent', agent,
                    '--scenario', str(scenario), '--timesteps', str(TIMESTEPS),
                    '--reward', reward,
                    '--output', f'models/{agent}/rewards/{reward}/s{scenario}',
                ], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()