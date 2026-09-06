import logging
from pathlib import Path

from flask import Flask, session
from flask_socketio import SocketIO

from flask_session import Session
from .webapp_config import load_webapp_config

from ...common.cli import CliRunnerMixin
from ...common.settings import ConfigBase, SettingsBase
from ..diagnostic.admin_pages import register_admin_urls
from ..orchestrator.orchestrator import orchestrator_init
from .webapp_config import load_flask_config, load_orchestrator_config

logger = logging.getLogger(__name__)

class WebappConfig(ConfigBase):
    sites_static_dir: Path = Path(r"src\tscapp\webapp\core\resources\static")
    sites_templates_dir: Path = Path(r"src\tscapp\webapp\core\resources\templates")
    config_file: Path = Path(r"src\tscapp\webapp\config.toml")
    flask_debug: bool = True

class WebappCli(CliRunnerMixin, SettingsBase):
    webapp: WebappConfig

    def cmd(self):
        webapp(config=self.webapp)

def webapp(config:WebappConfig):
    app, sio = webapp_init(config=config)
    wappconf = load_webapp_config(config.config_file.resolve())
    odb = orchestrator_init(load_orchestrator_config(config.config_file.resolve()), wappconf, sio)

    from .webapp_logic import register_urls
    register_urls(app=app, sio=sio, odb=odb, session=session, appdata_path=wappconf['file_path'])
    register_admin_urls(app=app, sio=sio, odb=odb)
    webapp_start(app=app, config=config)

def webapp_init(config: WebappConfig) -> tuple[Flask, SocketIO]:
    logger.info("static folder is %s", config.sites_static_dir.resolve())
    logger.info("template folder is %s", config.sites_templates_dir.resolve())
    logger.info("config file is %s", config.config_file.resolve())
    flask_app = Flask(
        "TSC",
        static_folder = config.sites_static_dir.resolve(),
        template_folder = config.sites_templates_dir.resolve(),
    )
    flask_app.config.from_file(config.config_file.resolve(), load=load_flask_config, text=False)

    logger.info("Attaching socketIO...")
    sio = SocketIO(flask_app)

    logger.info("Initializing session...")
    Session(flask_app)

    return flask_app, sio

def webapp_start(app: Flask, config: WebappConfig) -> None:
    logger.info("Spooling flask app...")
    app.run(debug=config.flask_debug)

if __name__ == "__main__":
    WebappCli.run()
