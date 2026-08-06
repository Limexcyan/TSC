from ..orchestrator.orchestrator_db import OrchestratorDB, under_lock
from flask_socketio import SocketIO
from flask import Flask

import logging

logger = logging.getLogger(__name__)

def register_admin_urls(app: Flask, sio: SocketIO, odb: OrchestratorDB) -> None:
    @app.route("/admin")
    def admin():
        return "admin"
    
    @under_lock
    @app.route("/admin/orchestrator/db/<table>")
    def admin_orchestrator_db(table):
        odb.db_cur.execute(f"SELECT * FROM {table}")
        records = odb.db_cur.fetchall()
        logger.info(f"<body><p>{"</p><p>".join(records)}</p></body>")
        return f"<body><p>{"</p><p>".join(records)}</p></body>"