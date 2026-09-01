from tscapp.common.names import random_sentence
from pathlib import Path
from json import dumps
from threading import Thread
from tscapp.core.processor import ProcessorConfig, processor
from tscapp.webapp.orchestrator.worker import worker_thread_init, RegisteredTasksPayload
from tscapp.webapp.orchestrator.orchestrator_db import OrchestratorDB
from typing import Any
import logging

logger = logging.getLogger(__name__)

registered_functions = dict()

def register_task(f):
    registered_functions[f.__name__] = f
    return f

@register_task
def webapp_processor(payload: RegisteredTasksPayload, config=None):
    if not config:
        raise RuntimeError("wanted kwarg config")
    path_prefix = payload.appdata_path / "files" / payload.task_client / payload.task_session
    files = config["input_files"]
    for i, file in enumerate(files):
        files[i] = str(path_prefix / file)
    config["input_files"] = files
    output_file = str(path_prefix / (random_sentence() + ".csv"))
    config["output_file"] = output_file
    config = ProcessorConfig.model_validate_json(dumps(config))
    processor(config)
    payload.odb.table_files_create(output_file, "csv", payload.task_client, payload.task_session)

def orchestrator_init(config: dict[str, Any]):
    logger.info("loading orchestrator...")
    orchestrator_db = OrchestratorDB(config['database_path'])
    logger.info("starting workers...")
    worker_count = config['worker_count']
    logger.info("registered: %s",registered_functions)
    while(worker_count > 0):
        worker_thread = Thread(target=worker_thread_init, kwargs={"worker_functions":registered_functions, "db_path":config['database_path'], "appdata_path":config['appdata_path']},daemon=True)
        worker_thread.start()
        worker_count -= 1
    return orchestrator_db
