from tscapp.webapp.core.webapp_common import merge_dicts
from json import dumps
from tscapp.common.names import random_sentence
import atexit
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
    current_work_order: str

    def __init__(self, worker_functions: dict[str,Callable], db_path: Path, appdata_path: Path, file_save_path: Path, sio: SocketIO):
        self.worker_functions = worker_functions
        self.appdata_path = file_save_path
        self.orchestrator_db = OrchestratorDB(db_path,sio)
        self.file_save_path = file_save_path / "task"
        self.current_work_order = None
        atexit.register(self.close_current_work)
    
    @under_worker_lock
    def fetch_work_order(self):
        work_orders_enqueued = self.orchestrator_db.table_tasks_query_status_absolute(status=TaskStatus.ENQUEUED)
        if len(work_orders_enqueued) < 1:
            logger.info("no enqueued work orders, releasing lock and waiting for orchestrator to raise event")
            return None
        oldest_work_order = work_orders_enqueued[0]
        work_order_uuid = oldest_work_order[0]
        work_order_data = loads(oldest_work_order[3])
        if not work_order_data.get('name', None):
            work_order_name = random_sentence()
            work_order_data['name'] = work_order_name
            self.orchestrator_db.table_tasks_update_data_absolute(uuid=work_order_uuid, data=work_order_data)
            logger.info("task named, returning to queue")
            return None
        else:
            work_order_name = work_order_data['name']
        self.current_work_order = work_order_uuid
        self.orchestrator_db.table_tasks_update_status_absolute(uuid=work_order_uuid, status=TaskStatus.RUNNING)
        logger.info(f"worker took on task {work_order_uuid} (alias {work_order_name})")
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
                name = task_data['name']
                if func in self.worker_functions:
                    func = self.worker_functions[func]
                    func_payload = RegisteredTasksPayload(self.orchestrator_db, self.appdata_path, self.file_save_path / task_uuid, task_client, task_session, task_data)
                    try:
                        result = func(func_payload, **kwargs)
                    except Exception:
                        logger.exception("task exception")
                        self.orchestrator_db.table_tasks_update_status(uuid=task_uuid, client=task_client, status=TaskStatus.FAILED)
                        self.current_work_order = None
                    else:
                        logger.info(f"task run {task_uuid} success")
                        task_data = merge_dicts(task_data, result)
                        self.orchestrator_db.table_tasks_update_data_absolute(uuid=task_uuid, data=task_data)
                        self.orchestrator_db.table_tasks_update_status(uuid=task_uuid, client=task_client, status=TaskStatus.COMPLETED)
                        self.current_work_order = None
            except Exception:
                logger.exception("worker exception")
                self.orchestrator_db.table_tasks_update_status_absolute(uuid=task_uuid, status=TaskStatus.CRASHED)
                self.current_work_order = None
    
    @under_worker_lock
    def close_current_work(self):
        if self.current_work_order:
            logger.error("closing work order...")
            self.orchestrator_db.table_tasks_update_status_absolute(uuid=self.current_work_order, status=TaskStatus.ENQUEUED)