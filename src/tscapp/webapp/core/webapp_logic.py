from ..orchestrator.orchestrator_db import OrchestratorDB
from flask_socketio import SocketIO
from flask import Flask
def register_urls(app: Flask, sio: SocketIO, odb: OrchestratorDB) -> None:

    @app.route('/')
    def index():
        return "index"

    @app.route('/<site>')
    def site(site):
        return site