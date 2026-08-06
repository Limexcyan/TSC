from typing import Any
from pathlib import Path
import tomllib

def read_config(file: Any) -> dict[str, Any]:
    return tomllib.load(file)

def load_flask_config(file: Any, flask_app_prefix: str = "FLASK_") -> dict[str, Any]:
    return read_config(file)['flask']
