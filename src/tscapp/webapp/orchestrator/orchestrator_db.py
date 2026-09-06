from tscapp.webapp.orchestrator.orchestrator_tasks import status_to_string
from flask_socketio import SocketIO
import logging
import sqlite3 as sql
from _thread import LockType
from datetime import UTC, datetime
from json import dumps, loads
from pathlib import Path
from threading import Event, Lock

from .orchestrator_tasks import TaskStatus

logger = logging.getLogger(__name__)

_dblock: LockType = Lock()
_worker_global_lock: LockType = Lock()
_notify_worker: Event = Event()

def under_lock(f):
    def wrapper(*args, **kwargs):
        _dblock.acquire()
        err = None
        try:
            return f(*args, **kwargs)
        except Exception as e:
            err = e
        finally:
            _dblock.release()
        if err:
            raise err
    return wrapper

def under_worker_lock(f):
    def wrapper(*args, **kwargs):
        _worker_global_lock.acquire()
        err = None
        try:
            return f(*args, **kwargs)
        except Exception as e:
            err = e
        finally:
            _worker_global_lock.release()
        if err:
            raise err
    return wrapper

def worker_event_signal():
    _notify_worker.set()

def worker_event_clear():
    _notify_worker.clear()

class OrchestratorDB:
    db_con: sql.Connection
    db_cur: sql.Cursor
    sio: SocketIO
    
    def __init__(self, db_path: Path, sio: SocketIO):
        self.db_con, self.db_cur = load_db(db_path)
        self.sio = sio

    @property
    def cursor(self):
        return self.db_cur

    def _send_info_task_update(self, uuid: str, status: TaskStatus, client: str | None = None,):
        logger.info(f"emitting update to {client}")
        try:
            if client:
                self.sio.emit("update", {"status":status.value,"name":status_to_string(status),"id":uuid}, to=client)
            else:
                self.sio.emit("update", {"status":status.value,"name":status_to_string(status),"id":uuid})
        except Exception:
            logger.error("socketio emit failed!")

    # SESSIONS

    @under_lock
    def table_sessions_create(self, uuid: str, client: str, alias: str ):
        created = datetime.now(UTC)
        self.db_cur.execute("INSERT INTO sessions(uuid,client,alias,created) VALUES (?, ?, ?, ?)", (uuid, client, alias, created))
        self.db_con.commit()
        return {"id":uuid, "client":client, "alias":alias, "created":created}

    @under_lock
    def table_sessions_retrieve_alias(self, uuid: str, client: str):
        self.db_cur.execute("SELECT alias FROM sessions WHERE uuid = ? AND client = ?", (uuid, client))
        res = self.db_cur.fetchone()
        if res:
            return res[0]
        return None

    @under_lock
    def table_sessions_retrieve_created(self, uuid: str, client: str):
        self.db_cur.execute("SELECT created FROM sessions WHERE uuid = ? AND client = ?", (uuid, client))
        res = self.db_cur.fetchone()
        if res:
            return res[0]
        return None
    
    @under_lock
    def table_sessions_query_client(self, client: str):
        self.db_cur.execute("SELECT * FROM sessions WHERE client = ?", (client,))
        res = self.db_cur.fetchall()
        if res:
            return res
        return []

    @under_lock
    def table_sessions_update_alias(self, uuid: str, client: str, alias: str):
        self.db_cur.execute("UPDATE sessions SET alias = ? WHERE uuid = ? AND client = ?", (alias,uuid,client))
        self.db_con.commit()

    @under_lock
    def table_sessions_delete(self, uuid: str, client: str):
        self.db_cur.execute("DELETE FROM sessions WHERE uuid = ? AND client = ?", (uuid, client))
        self.db_con.commit()
    
    # FILES

    @under_lock
    def table_files_create(self, path: str, type: str, client: str, session: str):
        self.db_cur.execute("INSERT INTO files(path,type,client,session) VALUES (?, ?, ?, ?)", (path, type, client, session))
        self.db_con.commit()
        return {"path":path, "type":type, "client":client, "session":session}
    
    @under_lock
    def table_files_retrieve_type(self, path: str, client: str):
        self.db_cur.execute("SELECT type FROM files WHERE path = ? AND client = ?", (path,client))
        res = self.db_cur.fetchone()
        if res:
            return res[0]
        return None
    
    @under_lock
    def table_files_retrieve_session(self, path: str, client: str):
        self.db_cur.execute("SELECT session FROM files WHERE path = ? AND client = ?", (path,client))
        res = self.db_cur.fetchone()
        if res:
            return res[0]
        return None

    @under_lock
    def table_files_query_client(self, client: str):
        self.db_cur.execute("SELECT * FROM files WHERE client = ?", (client,))
        res = self.db_cur.fetchall()
        if res:
            return res
        return None
    
    @under_lock
    def table_files_query_session(self, client: str, session: str):
        self.db_cur.execute("SELECT * FROM files WHERE client = ? AND session = ?", (client,session))
        res = self.db_cur.fetchall()
        if res:
            return res
        return None
    
    @under_lock
    def table_files_delete(self, path: str, client: str):
        self.db_cur.execute("DELETE FROM files WHERE path = ? AND client = ?", (path,client))
        self.db_con.commit()

    # TASKS
    
    @under_lock
    def table_tasks_create(self, uuid: str, client: str, session: str, data: any):
        data_json = dumps(data)
        status = TaskStatus.ENQUEUED.value
        modified = datetime.now(UTC)
        self.db_cur.execute("INSERT INTO tasks(uuid,client,session,data,status,modified) VALUES (?, ?, ?, ?, ?, ?)", (uuid, client, session, data_json, status, modified))
        self.db_con.commit()
        worker_event_signal()
        self._send_info_task_update(uuid,status)
        return {"uuid":uuid, "client":client, "session":session, "data":loads(data_json)}
    
    @under_lock
    def table_tasks_retrieve_data(self, uuid: str, client: str):
        self.db_cur.execute("SELECT data FROM tasks WHERE uuid = ? AND client = ?", (uuid,client))
        res = self.db_cur.fetchone()
        if res:
            return loads(res[0])
        return None

    @under_lock
    def table_tasks_retrieve_status(self, uuid: str, client: str):
        self.db_cur.execute("SELECT status FROM tasks WHERE uuid = ? AND client = ?", (uuid,client))
        res = self.db_cur.fetchone()
        if res:
            return loads(res[0])
        return None

    @under_lock
    def table_tasks_query_client(self, client: str):
        self.db_cur.execute("SELECT * FROM tasks WHERE client = ?", (client,))
        res = self.db_cur.fetchall()
        if res:
            return res
        return []
    
    @under_lock
    def table_tasks_query_session(self, client: str, session: str):
        self.db_cur.execute("SELECT * FROM tasks WHERE client = ? AND session = ?", (client,session))
        res = self.db_cur.fetchall()
        if res:
            return res
        return []
    
    @under_lock
    def table_tasks_query_status(self, client: str, status: TaskStatus):
        self.db_cur.execute("SELECT * FROM tasks WHERE client = ? AND status = ?", (client, status.value))
        res = self.db_cur.fetchall()
        if res:
            return res
        return []
    
    @under_lock
    def table_tasks_query_status_absolute(self, status: TaskStatus):
        self.db_cur.execute("SELECT * FROM tasks WHERE status = ?", (status.value,))
        res = self.db_cur.fetchall()
        if res:
            return res
        return []

    @under_lock
    def table_tasks_update_status(self, uuid: str, client: str, status: TaskStatus):
        self.db_cur.execute("UPDATE tasks SET status = ?, modified = ? WHERE uuid = ? AND client = ?", (status.value,datetime.now(UTC),uuid,client))
        self.db_con.commit()
        self._send_info_task_update(uuid,status)
        worker_event_signal()

    @under_lock
    def table_tasks_update_status_absolute(self, uuid: str, status: TaskStatus):
        self.db_cur.execute("UPDATE tasks SET status = ?, modified = ? WHERE uuid = ?", (status.value,datetime.now(UTC),uuid))
        self.db_con.commit()
        self._send_info_task_update(uuid,status)
        worker_event_signal()
        
    @under_lock
    def table_tasks_update_data_absolute(self, uuid: str, data):
        self.db_cur.execute("UPDATE tasks SET data = ?, modified = ? WHERE uuid = ?", (dumps(data),datetime.now(UTC),uuid))
        self.db_con.commit()

    @under_lock
    def table_tasks_delete(self, uuid: str, client: str):
        self.db_cur.execute("DELETE FROM tasks WHERE uuid = ? AND client = ?", (uuid,client))
        self.db_con.commit()

@under_lock
def init_db(cur: sql.Cursor) -> None:
    cur.executescript(
        """
        BEGIN;
        CREATE TABLE IF NOT EXISTS sessions (
            uuid TEXT PRIMARY KEY,
            client TEXT NOT NULL,
            alias TEXT,
            created TIMESTAMP NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            client TEXT NOT NULL,
            session TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            uuid TEXT PRIMARY KEY,
            client TEXT NOT NULL,
            session TEXT NOT NULL,
            data TEXT NOT NULL,
            status INTEGER NOT NULL,
            modified TIMESTAMP NOT NULL
        );
        COMMIT;
        """
    )


def load_db(db_file: Path) -> tuple[sql.Connection, sql.Cursor]:
    will_init = not db_file.exists()
    db_con = sql.connect(db_file, check_same_thread=False)
    db_cur = db_con.cursor()
    if will_init:
        init_db(db_cur)
    return db_con, db_cur
