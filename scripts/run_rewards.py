import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.game.config import SCENARIOS
from src.training.rewards import REWARDS


def main():
    for scenario in SCENARIOS:
        for agent in ('dqn', 'ppo'):
            for reward in REWARDS:
                output = f'models/{agent}/rewards/{reward}/s{scenario}'
                model = f'{output}/best.zip'
                
                subprocess.run([
                    sys.executable, 'train.py', '--agent', agent,
                    '--scenario', str(scenario), '--reward', reward,
                    '--output', f'models/{agent}/rewards/{reward}/s{scenario}',
                ], cwd=ROOT, check=True)

                subprocess.run([
                    sys.executable, 'evaluate.py',
                    '--agent', agent,
                    '--scenario', str(scenario),
                    '--model', model,
                ], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()