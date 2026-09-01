from flask import Response
import shutil
import os
from uuid import uuid4
import logging
from flask import make_response
from os.path import getsize
from werkzeug.utils import secure_filename
from pathlib import Path
from flask import Flask, request
from flask_socketio import SocketIO
from ...orchestrator.orchestrator_db import OrchestratorDB
from ..webapp_common import with_session
from ....common.names import random_sentence
from flask_session import Session

logger = logging.getLogger(__name__)

def api_session_init(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path):

    @app.route("/api/session", methods=["POST"])
    @with_session(session=session, create=True)
    def api_session_post():
        session_uuid = uuid4().hex
        session_client = session["uuid"]
        request_form = request.get_json()
        session_alias = request_form.get("name", random_sentence(long=True))
        session_path = appdata_path / session_client / session_uuid
        if session_path.exists():
            logger.error("Session %s exists", session_path)
            return make_response(('exists', 400))
        os.makedirs(session_path)
        metadata = odb.table_sessions_create(
            session_uuid,
            session_client,
            session_alias
        )
        return make_response((metadata,201))
    
    @app.route("/api/session", methods=["GET"])
    @with_session(session=session)
    def api_session_get():
        sessions = odb.table_sessions_query_client(session["uuid"])
        return make_response((sessions,200))
    
    @app.route("/api/session/file/<session_id>", methods=["GET"])
    @with_session(session=session)
    def api_file_session_file_session_get(session_id):
        metadata = odb.table_files_query_session(session["uuid"],session_id)
        if not metadata:
            metadata = []
        return make_response((metadata,200))

    @app.route("/api/session/<session_uuid>", methods=["PUT"])
    @with_session(session=session)
    def api_session_session_put(session_uuid):
        session_alias = request.form.get("name", default=None)
        if session_alias:
            return make_response(('no alias',400))
        odb.table_sessions_update_alias(
            session_uuid,
            session["uuid"],
            session_alias
        )
        return make_response(('ok',200))
    
    @app.route("/api/session/<session_uuid>", methods=["DELETE"])
    @with_session(session=session)
    def api_session_session_delete(session_uuid):
        session_client = session["uuid"]
        odb.table_sessions_delete(
            session_uuid,
            session_client
        )
        session_path = appdata_path / session_client / session_uuid
        if session_path.exists():
            shutil.rmtree(session_path)
        return make_response(('deleted',204))

    pass