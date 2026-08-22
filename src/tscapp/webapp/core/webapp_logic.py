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
from ..orchestrator.orchestrator_db import OrchestratorDB
from .webapp_common import list_files, assert_session
from flask_session import Session

from .api.session import api_session_init

logger = logging.getLogger(__name__)

def init_sessions(app: Flask) -> Session:
    Session(app)


def register_urls(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path) -> None:

    api_session_init(app, sio, odb, session, appdata_path)

    @app.route('/')
    def index():
        return render_template("landing.html")

    @app.route('/contents')
    def browse():
        assertion = assert_session(session, create=True)
        if assertion.status_code >= 400:
            return assertion
        return render_template("browse.html")

    @app.route('/session/<session_id>')
    def browse_session(session_id):
        assertion = assert_session(session, create=True)
        if assertion.status_code >= 400:
            return assertion
        return render_template("browse_session.html", uuid=session_id)

    @app.route('/session/<session_key>/<session_id>')
    def browse_public_session(session_key,session_id):
        return make_response(("ok",200))

    @app.route('/process')
    def process():
        return render_template("upload_pipeline_processor.html")

    @app.route('/upload', methods=["POST"])
    def upload():
        file = request.files['file']
        save = appdata_path / secure_filename(file.filename)
        current_chunk = int(request.form['dzchunkindex'])

        if save.exists() and current_chunk == 0:
            return make_response(('File already exists', 400))
        
        try:
            with open(save, 'ab') as fo:
                fo.seek(int(request.form['dzchunkbyteoffset']))
                fo.write(file.stream.read())
        except OSError:
            logger.exception("Could not write to file")
            return make_response(("Exception while writing the file to disk", 500))
        
        total_chunks = int(request.form['dztotalchunkcount']) - 1

        if current_chunk == total_chunks:
            if getsize(save) != int(request.form['dztotalfilesize']):
                logger.error("Filesize mismatch")
                return make_response(("Filesize mismatch", 500))
            else:
                logger.info("Upload successfull")
        
        return make_response(('Chunk write ok', 200))