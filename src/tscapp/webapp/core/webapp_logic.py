import logging
from pathlib import Path

from flask import Flask, render_template
from flask_socketio import SocketIO

from flask_session import Session
from tscapp.webapp.core.webapp_common import with_session

from ..orchestrator.orchestrator_db import OrchestratorDB
from .api.file import api_file_init
from .api.session import api_session_init
from .api.task import api_task_init

logger = logging.getLogger(__name__)

def init_sessions(app: Flask) -> Session:
    Session(app)


def register_urls(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path) -> None:

    api_session_init(app, sio, odb, session, appdata_path)
    api_file_init(app, sio, odb, session, appdata_path)
    api_task_init(app, sio, odb, session, appdata_path)

    @app.route('/')
    def index():
        return render_template("landing.html")

    @app.route('/session')
    @with_session(session=session)
    def browse():
        return render_template("client_sessions.html")

    @app.route('/session/<session_id>')
    @with_session(session=session)
    def browse_session(session_id):
        return render_template("client_session_view.html", uuid=session_id)
    
    @app.route('/task/<task_id>')
    @with_session(session=session)
    def browse_results(task_id):
        task_data = appdata_path / "task" / task_id
        html_embed_preliminary = ""
        for graph in task_data.glob("*.html"):
            if "_preliminary" in graph.stem:
                with open(graph, "r") as graph:
                    html_embed_preliminary = graph.read()
        return render_template("results.html", task_id=task_id,html_graph_analysis_preliminary=html_embed_preliminary)
