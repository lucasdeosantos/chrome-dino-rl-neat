import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data, compact_keys=()):
    text = json.dumps(data, indent=2)
    
    for key in compact_keys:
        expanded = json.dumps(data[key], indent=2).replace('\n', '\n  ')
        text = text.replace(f'"{key}": {expanded}', f'"{key}": {json.dumps(data[key])}')
    
    Path(path).write_text(text, encoding='utf-8')
