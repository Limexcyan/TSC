from _thread import LockType
from threading import Thread
from typing import Tuple
import sqlite3 as sql
from threading import Lock
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_dblock: LockType = Lock()

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

class OrchestratorDB():
    db_con: sql.Connection
    db_cur: sql.Cursor
    orchestrator_daemon: Thread
    
    def __init__(self, db_path: Path):
        self.db_con, self.db_cur = load_db(db_path)

    @property
    def cursor(self):
        return self.db_cur

    @under_lock
    def table_sessions_create(self, uuid: str, client: str, alias: str | None = None):
        expires = datetime.now()
        if alias:
            self.db_cur.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (uuid, client, alias, expires))
        else:
            self.db_cur.execute("INSERT INTO sessions(uuid,client,expires) VALUES (?, ?, ?)", (uuid, client, expires))
        self.db_con.commit()

    @under_lock
    def table_sessions_retrieve_alias(self, uuid: str, client: str) -> str:
        self.db_cur.execute("SELECT alias FROM sessions WHERE uuid = ? AND client = ?", (uuid, client))
        res = self.db_cur.fetchone()[0]
        return res

    @under_lock
    def table_sessions_retrieve_expires(self, uuid: str, client: str) -> str:
        self.db_cur.execute("SELECT expires FROM sessions WHERE uuid = ? AND client = ?", (uuid, client))
        res = self.db_cur.fetchone()[0]
        return res

    @under_lock
    def table_sessions_delete(self, uuid: str, client: str):
        self.db_cur.execute("DELETE FROM sessions WHERE uuid = ? AND client = ?", (uuid, client))
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
            expires TIMESTAMP NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            associated_session INTEGER
        );
        COMMIT;
        """
    )


def load_db(db_file: Path) -> Tuple[sql.Connection, sql.Cursor]:
    will_init = not db_file.exists()
    db_con = sql.connect(db_file, check_same_thread=False)
    db_cur = db_con.cursor()
    if will_init:
        init_db(db_cur)
    return db_con, db_cur
