from tscapp.webapp.orchestrator.orchestrator_tasks import status_code_to_status_string
from json import loads
from ...orchestrator.orchestrator_tasks import TaskStatus, status_code_to_status
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

def api_task_init(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path):

    @app.route("/api/task", methods=["POST"])
    @with_session(session=session)
    def api_task_post():
        session_client = session["uuid"]
        request_body = request.get_json()
        session_uuid = request_body.get("session", None)
        if not session_uuid:
            return make_response(('no session',400))
        task_data = request_body.get("data", None)
        if not task_data:
            return make_response(('no data', 400))
        task_uuid = uuid4().hex
        metadata = odb.table_tasks_create(
            task_uuid,
            session_client,
            session_uuid,
            task_data
        )
        os.makedirs(appdata_path / "task" / task_uuid)
        return make_response(('{"status":"enqueued"}',201))

    def _aux_status_id_to_message(record):
        record = list(record)
        record.append(status_code_to_status_string(record[4]))
        return record

    @app.route("/api/task", methods=["GET"])
    @with_session(session=session)
    def api_task_get():
        session_client = session["uuid"]
        metadata = odb.table_tasks_query_client(session_client)
        if not metadata:
            metadata = []
        metadata = list(map(_aux_status_id_to_message,metadata))
        return make_response((metadata, 200))

    @app.route("/api/task/session/<session_id>", methods=["GET"])
    @with_session(session=session)
    def api_task_session_session_get(session_id):
        session_client = session["uuid"]
        metadata = odb.table_tasks_query_session(session_client, session_id)
        if not metadata:
            metadata = []
        metadata = list(map(_aux_status_id_to_message,metadata))
        return make_response((metadata, 200))
    
    @app.route("/api/task/<task_id>", methods=["GET"])
    @with_session(session=session)
    def api_task_task_get(task_id):
        session_client = session["uuid"]
        data = loads(odb.table_tasks_retrieve_data(task_id, session_client))
        status = odb.table_tasks_retrieve_status(task_id, session_client)
        return make_response(({"status":status,"data":data},200))
    
    @app.route("/api/task/<task_id>", methods=["PUT"])
    @with_session(session=session)
    def api_task_task_put(task_id):
        session_client = session["uuid"]
        request_form = request.get_json()
        updated_status = request_form.get("status", None)
        if not updated_status:
            return make_response(('no update', 400))
        odb.table_tasks_update_status(task_id, session_client, status_code_to_status(int(updated_status)))
        return make_response(('{"status":"updated"}', 200))
    
    @app.route("/api/task/<task_id>", methods=["DELETE"])
    @with_session(session=session)
    def api_task_task_delete(task_id):
        session_client = session["uuid"]
        tasks = odb.table_tasks_query_client(session_client)
        tasks_uuid = list(map(lambda x: x[0], tasks))
        if task_id in tasks_uuid:
            odb.table_tasks_delete(task_id, session_client)
            for file in (appdata_path / "task" / task_id).glob("*"):
                odb.table_files_delete(str(file.relative_to(appdata_path)), session_client)
            shutil.rmtree(appdata_path / "task" / task_id)
            return make_response(('{"status":"deleted"}',204))
        return make_response(('{"status":"forbidden"}', 403))
    

