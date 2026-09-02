from json import loads
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
    path_prefix = payload.appdata_path / payload.task_client / payload.task_session
    files = config["input_files"]
    for i, file in enumerate(files):
        files[i] = str(path_prefix / file)
    config["input_files"] = files
    res = payload.full_request.get("result", None)
    if res:
        logger.info("task has been resubmitted, will overwrite files")
        output_file_name = res["output-filename"]
    else:
        output_file_name = random_sentence()
    output_file = payload.file_save_path / (output_file_name + ".csv")
    clustered = payload.file_save_path / (output_file_name + ".bin.clustered")
    preliminary = payload.file_save_path / (output_file_name + "_preliminary.html")
    config["output_file"] = str(output_file)
    config = ProcessorConfig.model_validate_json(dumps(config))
    processor(config)
    if not res:
        payload.odb.table_files_create(str(output_file.relative_to(payload.appdata_path)), "csv", payload.task_client, payload.task_session)
        payload.odb.table_files_create(str(clustered.relative_to(payload.appdata_path)), "clu", payload.task_client, payload.task_session)
        payload.odb.table_files_create(str(preliminary.relative_to(payload.appdata_path)), "ana", payload.task_client, payload.task_session)
    return {"output-filename":output_file_name}

def orchestrator_init(config: dict[str, Any], wa_config: dict[str, Any]):
    logger.info("loading orchestrator...")
    orchestrator_db = OrchestratorDB(config['database_path'])
    logger.info("starting workers...")
    worker_count = config['worker_count']
    logger.info("registered: %s",registered_functions)
    while(worker_count > 0):
        worker_thread = Thread(target=worker_thread_init, kwargs={"worker_functions":registered_functions, "db_path":config['database_path'], "appdata_path":config['appdata_path'], "file_save_path":wa_config['file_path']},daemon=True)
        worker_thread.start()
        worker_count -= 1
    return orchestrator_db
