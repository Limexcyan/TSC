from json import dumps
from ...core.processor import ProcessorConfig, processor
from tscapp.common.names import random_sentence
import logging
from enum import IntFlag, auto

logger = logging.getLogger(__name__)

class TaskStatus(IntFlag):
    WAITING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

    ENQUEUED = auto()
    LATE = auto()

    CRASHED = auto()

TaskStatusTranslation = {
        TaskStatus.WAITING: "Waiting",
        TaskStatus.RUNNING: "Running",
        TaskStatus.COMPLETED: "Completed",
        TaskStatus.FAILED: "Failed",
        TaskStatus.ENQUEUED: "Enqueued",
        TaskStatus.LATE: "Late",
        TaskStatus.CRASHED: "Crashed",
    }

def status_to_string(status: TaskStatus):
    if status in TaskStatusTranslation:
        return TaskStatusTranslation[status]
    res = ""
    for _status, value in TaskStatusTranslation.items():
        if status & _status:
            res += value
    return res

def status_code_to_status(code: int):
    res_status = None
    for status in TaskStatusTranslation:
        if status & code:
            if not res_status:
                res_status = status
            res_status |= status
    return res_status


def status_code_to_status_string(code: int):
    return status_to_string(status_code_to_status(code))


registered_functions = {}

def register_task(f):
    registered_functions[f.__name__] = f
    return f

@register_task
def webapp_processor(payload, config=None):
    if not config:
        raise RuntimeError("wanted kwarg config")
    path_prefix = payload.appdata_path / payload.task_client / payload.task_session
    files = config["input_files"]
    for i, file in enumerate(files):
        files[i] = str(path_prefix / file)
    config["input_files"] = files
    completed_flag = payload.full_request.get("completed_flag", None)
    output_file_name = payload.full_request["name"]
    output_file = payload.file_save_path / (output_file_name + ".csv")
    clustered = payload.file_save_path / (output_file_name + ".bin.clustered")
    preliminary = payload.file_save_path / (output_file_name + "_preliminary.html")
    config["output_file"] = str(output_file)
    config = ProcessorConfig.model_validate_json(dumps(config))
    response = processor(config)
    if not completed_flag:
        payload.odb.table_files_create(str(output_file.relative_to(payload.appdata_path)), "csv", payload.task_client, payload.task_session)
        payload.odb.table_files_create(str(clustered.relative_to(payload.appdata_path)), "clu", payload.task_client, payload.task_session)
        payload.odb.table_files_create(str(preliminary.relative_to(payload.appdata_path)), "ana", payload.task_client, payload.task_session)
    return {"completed_flag":True,"kwargs":{"config":{"clustering_config":{"seed":response}}}}