from datetime import datetime
from pathlib import Path
import os
import re
import sys
import warnings


def open_log(model_path, operation):
    root = Path(__file__).resolve().parents[1]
    path = Path(model_path).resolve()

    if operation == 'evaluation':
        path = path.parent
    
    try:
        parts = path.relative_to(root / 'models').parts
    except ValueError:
        parts = path.parts[path.parts.index('models') + 1:] if 'models' in path.parts else (path.name,)
    
    name = '_'.join(parts) or 'model'
    name = re.sub(r'[^\w.-]+', '_', name)
    folder = root / 'results' / 'logs'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = folder / f'{name}_{operation}_{stamp}.log'
    
    number = 1
    while True:
        try:
            log = log_path.open('x', encoding='utf-8', buffering=1)
            break
        except FileExistsError:
            log_path = folder / f'{name}_{operation}_{stamp}_{number}.log'
            number += 1
    
    return log_path, log


def start_log(model_path, operation):
    path, file = open_log(model_path, operation)
    print(f'log: {os.path.relpath(path).replace(os.sep, "/")}', flush=True)
    sys.stderr.flush()
    log = {'path': path, 'file': file, 'stdout': sys.stdout, 'stderr': sys.stderr,
           'stdout_fd': os.dup(1), 'stderr_fd': os.dup(2),
           'warning_filters': warnings.filters[:]}
    
    try:
        warnings.filterwarnings('ignore', message=r'CUDA initialization: The NVIDIA driver on your system is too old',
                                category=UserWarning, module=r'torch\.cuda(?:\.|$)')
        os.dup2(file.fileno(), 1)
        os.dup2(file.fileno(), 2)
        sys.stdout = file
        sys.stderr = file
        print(f'Started: {datetime.now().astimezone().isoformat(timespec="seconds")}')
        return log
    except BaseException:
        finish_log(log)
        raise


def finish_log(log):
    try:
        print(f'Finished: {datetime.now().astimezone().isoformat(timespec="seconds")}', file=log['file'])
        log['file'].flush()
    finally:
        warnings.filters[:] = log['warning_filters']
        sys.stdout = log['stdout']
        sys.stderr = log['stderr']
        os.dup2(log['stdout_fd'], 1)
        os.dup2(log['stderr_fd'], 2)
        os.close(log['stdout_fd'])
        os.close(log['stderr_fd'])
        log['file'].close()
