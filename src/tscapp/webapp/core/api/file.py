import logging
import os
from os.path import getsize
from pathlib import Path

from flask import Flask, make_response, request, send_file
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from flask_session import Session

from ...orchestrator.orchestrator_db import OrchestratorDB
from ..webapp_common import with_session

logger = logging.getLogger(__name__)

def api_file_init(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path):

    @app.route("/api/file", methods=["POST"])
    @with_session(session=session)
    def api_file_post():
        session_uuid = request.form.get("session", default=None)
        if not session_uuid:
            return make_response(('no session',400))
        file_type = request.form.get("type", default=None)
        if not file_type:
            return make_response(('ambiguous type',403))
        session_client = session["uuid"]
        session_path = appdata_path / session_client / session_uuid
        if not session_path.exists():
            logger.error("Session %s does not exist", session_path)
            return make_response(('no session', 403))

        file = request.files['file']
        save = session_path / secure_filename(file.filename)
        current_chunk = int(request.form['dzchunkindex'])

        if save.exists() and current_chunk == 0:
            return make_response(('file exists', 400))
        
        try:
            with open(save, 'ab') as fo:
                fo.seek(int(request.form['dzchunkbyteoffset']))
                fo.write(file.stream.read())
        except OSError:
            logger.exception("Could not write to file")
            return make_response(("exception writing the file to disk", 500))
        
        total_chunks = int(request.form['dztotalchunkcount']) - 1

        if current_chunk == total_chunks:
            if getsize(save) != int(request.form['dztotalfilesize']):
                logger.error("Filesize mismatch")
                return make_response(("filesize mismatch", 500))
            else:
                logger.info("upload successfull")
                metadata = odb.table_files_create(
                    str(save.relative_to(appdata_path)),
                    file_type,
                    session_client,
                    session_uuid
                )
                return make_response((metadata, 201))
        
        return make_response(('chunck write ok', 200))

    @app.route("/api/file", methods=["GET"])
    @with_session(session=session)
    def api_file_get():
        metadata = odb.table_files_query_client(session["uuid"])
        return make_response((metadata,200))
    
    @app.route("/api/file/<uuid>/<fn>", methods=["GET"])
    @with_session(session=session)
    def api_file_file_download(uuid,fn):
        session_client = session["uuid"]
        session_uuid = uuid
        path = str(Path(session_client) / session_uuid / fn)
        print(appdata_path, path)
        return send_file(appdata_path / path)
    
    @app.route("/api/file/task/<uuid>/<fn>", methods=["GET"])
    @with_session(session=session)
    def api_file_task_file_download(uuid,fn):
        task_uuid = uuid
        path = str(Path("task") / task_uuid / fn)
        print(appdata_path, path)
        return send_file(appdata_path / path)

    @app.route("/api/file/session/<session_id>", methods=["GET"])
    @with_session(session=session)
    def api_file_session_session_get(session_id):
        metadata = odb.table_files_query_session(session["uuid"],session_id)
        return make_response((metadata,200))

    @app.route("/api/file/<path>", methods=["GET"])
    @with_session(session=session)
    def api_file_file_get(path):
        session_client = session["uuid"]
        file_associated_session = odb.table_files_retrieve_session(path, session_client)
        if not file_associated_session:
            return (('not tracked',404))
        file_path = Path(path)
        if not file_path.exists():
            return (('does not exist', 404))
        if str(file_path.parent.resolve().absolute()) != str((appdata_path / session_client / file_associated_session).resolve().absolute()):
            return (('no access',403))
        metadata = odb.table_files_retrieve_type(path, session_client)
        return make_response((metadata,200))
    
    @app.route("/api/file/<fn>", methods=["DELETE"])
    @with_session(session=session)
    def api_file_file_delete(fn):
        session_client = session["uuid"]
        request_body = request.get_json()
        session_uuid = request_body.get("session", None)
        task_uuid = request_body.get("task", None)
        if not session_uuid and not task_uuid:
            return make_response(('no data',400))
        if session_uuid:
            path = str(Path(session_client) / session_uuid / fn)
        else:
            path = str(Path("task") / task_uuid / fn)
        file_associated_session = odb.table_files_retrieve_session(path, session_client)
        if not file_associated_session:
            return (('not tracked',404))
        file_path = appdata_path / path
        if not file_path.exists():
            return (('does not exist', 404))
        if str(file_path.parent.resolve().absolute()) != str((appdata_path / session_client / file_associated_session).resolve().absolute()):
            return (('no access',403))
        odb.table_files_delete(
            path,
            session_client
        )
        os.remove(file_path)
        return make_response(('deleted',204))

