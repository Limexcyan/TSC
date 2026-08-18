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
from flask_session import Session

logger = logging.getLogger(__name__)

def init_sessions(app: Flask) -> Session:
    Session(app)

def list_files(dir: Path):
    dirs = next(dir.walk())[1]
    dirs.sort(key=lambda x: -os.path.getmtime(dir / x))
    manifest = dict()
    for d in dirs:
        manifest.update({d:next((dir / d).walk())[2]})
    return manifest
    
def register_urls(app: Flask, sio: SocketIO, odb: OrchestratorDB, session: Session, appdata_path: Path) -> None:

    @app.route('/')
    def index():
        return render_template("landing.html")

    @app.route('/contents')
    def browse():
        if not session.get("uuid"):
            session["uuid"] = uuid4().hex
            logger.info("new session created: %s", session["uuid"])
        contents: Path = appdata_path / session["uuid"]
        if not contents.exists():
            os.makedirs(contents)
        return render_template("browse.html", contents=list_files(contents))

    @app.route('/process')
    def process():
        return render_template("upload_pipeline_processor.html")

    @app.route('/session', methods=["POST"])
    def session_init():
        name = request.args.get("name", default=uuid4().hex)
        if not session.get("uuid"):
            logger.error("requested to init processor session with no active webapp session")
            return make_response(("No active webapp session", 400))
        if (appdata_path / session["uuid"] / name).exists():
            logger.error("session %s already exists", name)
            return make_response(("Session name already exists, request different session name", 400))
        os.makedirs(appdata_path / session["uuid"] / name)
        return make_response(("Session created", 200))
    
    @app.route('/session/<name>', methods=["GET"])
    def session_get(name):
        if not session.get("uuid"):
            logger.error("requested to delete processor session with no active webapp session")
            return make_response(("No active webapp session", 400))
        if not (appdata_path / session["uuid"] / name).exists():
            logger.error("session %s does not exist", name)
            return make_response(("Session does not exist", 400))
        return render_template("browse_session.html")
    
    @app.route('/session/<name>', methods=["DELETE"])
    def session_delete(name):
        if not session.get("uuid"):
            logger.error("requested to delete processor session with no active webapp session")
            return make_response(("No active webapp session", 400))
        if not (appdata_path / session["uuid"] / name).exists():
            logger.error("session %s does not exist", name)
            return make_response(("Session does not exist", 400))
        shutil.rmtree(appdata_path / session["uuid"] / name)
        return make_response(("Session deleted", 200))

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