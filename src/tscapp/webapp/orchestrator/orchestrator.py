from tscapp.webapp.orchestrator.orchestrator_tasks import registered_functions
from flask_socketio import SocketIO
import logging
from json import dumps
from threading import Thread
from typing import Any

from ...common.names import random_sentence
from ...core.processor import ProcessorConfig, processor
from .orchestrator_db import OrchestratorDB
from .worker import (
    RegisteredTasksPayload,
    worker_thread_init,
)

logger = logging.getLogger(__name__)

def orchestrator_init(config: dict[str, Any], wa_config: dict[str, Any], sio: SocketIO):
    logger.info("loading orchestrator...")
    orchestrator_db = OrchestratorDB(config['database_path'],sio)
    logger.info("starting workers...")
    worker_count = config['worker_count']
    logger.info("registered: %s",registered_functions)
    while(worker_count > 0):
        worker_thread = Thread(target=worker_thread_init, kwargs={"worker_functions":registered_functions, "db_path":config['database_path'], "appdata_path":config['appdata_path'], "file_save_path":wa_config['file_path'], "sio":sio},daemon=True)
        worker_thread.start()
        worker_count -= 1
    return orchestrator_db
