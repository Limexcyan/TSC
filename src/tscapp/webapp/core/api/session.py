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
from flask_session import Session

logger = logging.getLogger(__name__)

def api_session_init(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path):

    @app.route('/session', methods=["POST"])
    def session_create():
        assertion: Response = assert_session(session, create=True)
        if assertion.status_code >= 400:
            return assertion
        session_name = request.args.get("name", default=uuid4().hex)
        path_to_session = appdata_path / session["uuid"] / session_name
        if path_to_session.exist():
            logger.error("Session named %s already exists", session_name)
            return make_response(("already exists", 400))
        os.makedirs(path_to_session)
        return make_response(("created", 201))

    @app.route('/session', methods=["GET"])
    def session_enumerate():
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        session_root = appdata_path / session["uuid"]
        if not session_root.exists():
            logger.error("No session %s for user %s", session_root, session['uuid'])
            return make_response(("not found", 404))
        subfolders = list_files(session_root)
        return make_response((subfolders, 200))
    
    @app.route('/session/<session>', methods=["GET"])
    def session_details(session):
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        files_root = appdata_path / session["uuid"] / session
        if not files_root.exists():
            logger.error("No session %s", files_root)
            return make_response(("not found", 404))
        subfolders = list_files(files_root)
        return make_response((subfolders, 200))
        pass
    
    @app.route('/session/<session>', methods=["POST"])
    def session_upload(session):
        assertion = assert_session(session, create=True)
        if assertion.status_code >= 400:
            return assertion
        file = request.files['file']
        save = appdata_path / session["uuid"] / session / secure_filename(file.filename)
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
    
    @app.route('/session/<session>/<file>', methods=["DELETE"])
    def session_delete_file(session,file):
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        file_path = appdata_path / session["uuid"] / session / file
        if not file_path.exists():
            logger.warning("File %s does not exist", file_path)
            return make_response(("no action", 204))
        file_path.unlink()
        return make_response(("deleted", 204))
    
    @app.route('/session/<session>', methods=["DELETE"])
    def session_delete(session):
        assertion = assert_session(session)
        if assertion.status_code >= 400:
            return assertion
        session_path = appdata_path / session["uuid"] / session
        if not session_path.exists():
            logger.warning("No session %s", session_path)
            return make_response(("no action", 204))
        shutil.rmtree(session_path)
        return make_response(("deleted", 204))