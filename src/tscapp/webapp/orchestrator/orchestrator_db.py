from threading import Lock
from pathlib import Path

class OrchestratorLock():
    _instance = None
    _lock: Lock | None = None

    def __new__(cls):
        if cls._instance:
            return cls._instance
        cls._instance = super().__new__(cls)
        cls._init = False
        return cls._instance
    
    def __init__(self):
        if self._init:
            return
        self._lock = Lock()
    
    def acquire(self):
        if not self._lock:
            raise RuntimeError("OrchestratorLock singleton called acquire yet not initialized!")
        self._lock.acquire()
    
    def release(self):
        if not self._lock:
            raise RuntimeError("OrchestratorLock singleton called release yet not initialized!")
        self._lock.release()
        

def load_db(db_file: Path):
    db = None
    if db_file.exists():
        db = db_file
    else:
        db = create_db(db_file)
    sql_con = ...

def create_db(db_file: Path):
    pass