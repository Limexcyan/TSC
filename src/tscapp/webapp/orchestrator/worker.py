from flask_socketio import SocketIO
import logging
from collections.abc import Callable
from dataclasses import dataclass
from json import loads
from pathlib import Path
from sqlite3 import Connection, Cursor

from .orchestrator_db import (
    OrchestratorDB,
    _notify_worker,
    under_worker_lock,
)
from .orchestrator_tasks import TaskStatus

logger = logging.getLogger(__name__)

@dataclass
class RegisteredTasksPayload:
    odb: OrchestratorDB
    appdata_path: Path
    file_save_path: Path
    task_client: str
    task_session: str
    full_request: str

def worker_thread_init(worker_functions: dict[str,Callable], db_path: Path, appdata_path: Path, file_save_path: Path, sio: SocketIO):
    worker = Worker(worker_functions, db_path, appdata_path, file_save_path, sio)
    worker.worker_loop()
    return True

class Worker:
    worker_functions: dict[str,Callable]
    should_exit: bool = False
    appdata_path: Path
    worker_orchestrator_db_con: Connection
    worker_orchestrator_db_cur: Cursor
    orchestrator_db: OrchestratorDB

    def __init__(self, worker_functions: dict[str,Callable], db_path: Path, appdata_path: Path, file_save_path: Path, sio: SocketIO):
        self.worker_functions = worker_functions
        self.appdata_path = file_save_path
        self.orchestrator_db = OrchestratorDB(db_path,sio)
        self.file_save_path = file_save_path / "task"
    
    @under_worker_lock
    def fetch_work_order(self):
        work_orders_enqueued = self.orchestrator_db.table_tasks_query_status_absolute(status=TaskStatus.ENQUEUED)
        if len(work_orders_enqueued) < 1:
            logger.info("no enqueued work orders, releasing lock and waiting for orchestrator to raise event")
            return None
        oldest_work_order = work_orders_enqueued[0]
        work_order_uuid = oldest_work_order[0]
        self.orchestrator_db.table_tasks_update_status_absolute(uuid=work_order_uuid, status=TaskStatus.RUNNING)
        logger.info(f"worker took on task {work_order_uuid}")
        return oldest_work_order

    def worker_loop(self):
        while(not self.should_exit):
            if(_notify_worker.is_set()):
                _notify_worker.clear()
                logger.info("worker ready, clearing event flag")
            res = self.fetch_work_order()
            if not res:
                _notify_worker.wait()
                continue
            task_uuid, task_client, task_session, task_data = res[:4]
            task_data = loads(task_data)
            try:
                func = task_data['func']
                kwargs = task_data['kwargs']
                if func in self.worker_functions:
                    func = self.worker_functions[func]
                    func_payload = RegisteredTasksPayload(self.orchestrator_db, self.appdata_path, self.file_save_path / task_uuid, task_client, task_session, task_data)
                    try:
                        result = func(func_payload, **kwargs)
                    except Exception:
                        logger.exception()
                        self.orchestrator_db.table_tasks_update_status(uuid=task_uuid, client=task_client, status=TaskStatus.FAILED)
                    else:
                        task_data["result"] = result
                        logger.info(f"task run {task_uuid} success")
                        self.orchestrator_db.table_tasks_update_data_absolute(uuid=task_uuid, data=task_data)
                        self.orchestrator_db.table_tasks_update_status(uuid=task_uuid, client=task_client, status=TaskStatus.COMPLETED)
            except Exception:
                logger.exception()
                self.orchestrator_db.table_tasks_update_status_absolute(uuid=task_uuid, status=TaskStatus.CRASHED)