from typing import Any
from pathlib import Path
import tomllib

def read_config(file: Any) -> dict[str, Any]:
    return tomllib.load(file)

def load_flask_config(file: Any) -> dict[str, Any]:
    return read_config(file)['flask']

def load_orchestrator_config(file: Any) -> dict[str, Any]:
    conf = None
    with open(file, "rb") as fo:
        conf = read_config(fo)['orchestrator']
    conf['database_path'] = Path(conf['database_path']).resolve()
    conf['appdata_path'] = Path(conf['appdata_path']).resolve()
    return conf

def load_webapp_config(file: Any) -> dict[str, Any]:
    conf = None
    with open(file, "rb") as fo:
        conf = read_config(fo)['webapp']
    conf['file_path'] = Path(conf['file_path']).resolve()
    return conf