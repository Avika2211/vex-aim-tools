"""Minimal, zero-dependency .env loader.

Reads KEY=VALUE lines from a local `.env` file and puts them in os.environ.
Existing environment variables are NOT overwritten, so a real `export` still
takes precedence over the file. This lets you keep API keys (e.g.
ELEVENLABS_API_KEY, OPENAI_API_KEY) in a local .env instead of exporting them.
"""

import os


def _find_env_file():
    # Search the current working directory and the project root (one level
    # above this package) for a `.env` file.
    candidates = [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _strip_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def load_dotenv(path=None, override=False):
    """Load variables from a .env file into os.environ.

    Returns the path that was loaded, or None if no file was found.
    """
    path = path or _find_env_file()
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[len('export '):].strip()
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = _strip_quotes(value.strip())
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f'*** Failed to load .env file {path}: {e}')
        return None
    return path
