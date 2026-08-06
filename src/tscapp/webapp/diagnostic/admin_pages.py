from flask_socketio import SocketIO
from flask import Flask

def register_admin_urls(app: Flask, sio: SocketIO) -> None:
    @app.route("/admin")
    def admin():
        return "admin"