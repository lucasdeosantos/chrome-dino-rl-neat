import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.agents import AGENTS
from src.game.config import SCENARIOS


def main():
    for agent in AGENTS:
        extension = 'pkl' if agent == 'neat' else 'zip'
        for scenario in SCENARIOS:
            folder = f'models/{agent}/s{scenario}'
            subprocess.run([
                sys.executable, 'train.py',
                '--agent', agent, '--scenario', str(scenario),
                '--output', folder,
            ], cwd=ROOT, check=True)
            subprocess.run([
                sys.executable, 'evaluate.py',
                '--agent', agent, '--scenario', str(scenario),
                '--model', f'{folder}/best.{extension}',
            ], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()