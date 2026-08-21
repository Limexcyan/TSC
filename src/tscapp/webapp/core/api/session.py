from flask import Response
import shutil
import os
from uuid import uuid4
import logging
from flask import make_response
from os.path import getsize
from werkzeug.utils import secure_filename
from pathlib import Path
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from ...orchestrator.orchestrator_db import OrchestratorDB
from ..webapp_common import list_files, assert_session
from ....common.names import random_sentence
from flask_session import Session

logger = logging.getLogger(__name__)

def api_session_init(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path):

    @app.route('/api/session', methods=["POST"])
    def session_create():
        assertion: Response = assert_session(session, create=True)
        if assertion.status_code >= 400:
            return assertion
        session_uuid = uuid4().hex
        path_to_session = appdata_path / session["uuid"] / session_uuid
        if path_to_session.exists():
            logger.error("Session named %s already exists", session_uuid)
            return make_response(("already exists", 400))
        os.makedirs(path_to_session)
        odb.table_sessions_create(session_uuid, session["uuid"], request.args.get("name", default=random_sentence(long=True)))
        return make_response(("created", 201))

    @app.route('/api/session', methods=["GET"])
    def session_enumerate():
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        session_root = appdata_path / session["uuid"]
        if not session_root.exists():
            logger.error("No session %s for user %s", session_root, session['uuid'])
            return make_response(("not found", 404))
        subfolders = list(list_files(session_root).keys())
        return make_response((subfolders, 200))
    
    @app.route('/api/session/<session_id>', methods=["GET"])
    def session_details(session_id):
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        files_root = appdata_path / session["uuid"] / session_id
        if not files_root.exists():
            logger.error("No session %s", files_root)
            return make_response(("not found", 404))
        details = {
            "session_id": session_id,
            "session_alias": odb.table_sessions_retrieve_alias(session_id, session["uuid"]),
            "session_expires": odb.table_sessions_retrieve_expires(session_id, session["uuid"]),
            "session_contents": list(list_files(files_root).keys())
        }
        return make_response((details, 200))
        pass
    
    @app.route('/api/session/<session_id>', methods=["POST"])
    def session_upload(session_id):
        assertion = assert_session(session, create=True)
        if assertion.status_code >= 400:
            return assertion
        file = request.files['file']
        save = appdata_path / session["uuid"] / session_id / secure_filename(file.filename)
        current_chunk = int(request.form.get('dzchunkindex', 0))
        if save.exists() and current_chunk == 0:
            return make_response(('file already exists', 400))
        try:
            with open(save, 'ab') as fo:
                fo.seek(int(request.form.get('dzchunkbyteoffset', 0)))
                fo.write(file.stream.read())
        except OSError:
            logger.exception("could not write to file")
            return make_response(("exception", 500))
        
        total_chunks = int(request.form.get('dztotalchunkcount', 1)) - 1
        if current_chunk == total_chunks:
            if getsize(save) != int(request.form.get('dztotalfilesize', getsize(save))):
                logger.error("filesize mismatch")
                return make_response(("filesize mismatch", 500))
            else:
                logger.info("upload successfull")
        return make_response(('write ok', 200))
    
    @app.route('/api/session/<session_id>/<file>', methods=["DELETE"])
    def session_delete_file(session_id,file):
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        file_path = appdata_path / session["uuid"] / session_id / file
        if not file_path.exists():
            logger.warning("File %s does not exist", file_path)
            return make_response(("no action", 204))
        file_path.unlink()
        return make_response(("deleted", 204))
    
    @app.route('/api/session/<session_id>', methods=["DELETE"])
    def session_delete(session_id):
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        session_path = appdata_path / session["uuid"] / session_id
        if not session_path.exists():
            logger.warning("No session %s", session_path)
            return make_response(("no action", 204))
        shutil.rmtree(session_path)
        odb.table_sessions_delete(session_id, session["uuid"])
        return make_response(("deleted", 204))