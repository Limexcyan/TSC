from typing import Tuple
from pathlib import Path
from flask import Flask
from flask_socketio import SocketIO

from ...common.cli import CliRunnerMixin
from ...common.settings import ConfigBase, SettingsBase

from .webapp_config import load_flask_config
from .webapp_logic import register_urls

from ..diagnostic.admin_pages import register_admin_urls

import logging

logger = logging.getLogger(__name__)

class WebappConfig(ConfigBase):
    sites_static_dir: Path = Path(r"src\tscapp\webapp\resources\templates")
    sites_templates_dir: Path = Path(r"src\tscapp\webapp\resources")
    flask_config_file: Path = Path(r"src\tscapp\webapp\config.toml")
    flask_debug: bool = True

class WebappCli(CliRunnerMixin, SettingsBase):
    webapp: WebappConfig

    def cmd(self):
        webapp(config=self.webapp)

def webapp(config:WebappConfig):
    app, sio = webapp_init(config=config)
    register_urls(app=app, sio=sio)
    register_admin_urls(app=app, sio=sio)
    webapp_start(app=app, sio=sio, config=config)

def webapp_init(config: WebappConfig) -> Tuple[Flask, SocketIO]:
    logger.info("static folder is %s", config.sites_static_dir.resolve())
    logger.info("template folder is %s", config.sites_templates_dir.resolve())
    logger.info("config file is %s", config.flask_config_file.resolve())
    flask_app = Flask(
        "TSC",
        static_folder = config.sites_static_dir.resolve(),
        template_folder = config.sites_templates_dir.resolve(),
    )
    flask_app.config.from_file(config.flask_config_file.resolve(), load=load_flask_config, text=False)

    logger.info("Attaching socketIO...")
    sio = SocketIO(flask_app)

    return flask_app, sio

def webapp_start(app: Flask, sio: SocketIO, config: WebappConfig) -> None:
    logger.info("Spooling flask app...")
    app.run(debug=config.flask_debug)

if __name__ == "__main__":
    WebappCli.run()