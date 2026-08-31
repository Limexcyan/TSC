from flask import make_response
from uuid import uuid4
from functools import wraps
from flask_session import Session
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def list_files(dir: Path):
    dirs = next(dir.walk())[1]
    dirs.sort(key=lambda x: -os.path.getmtime(dir / x))
    manifest = dict()
    for d in dirs:
        manifest.update({d:next((dir / d).walk())[2]})
    return manifest

def assert_session(session: Session, create=False):
    if not session.get("uuid"):
        if create:
            session_uuid = uuid4().hex
            session["uuid"] = session_uuid
            logger.info("Created new session %s", session_uuid)
            return make_response(("created", 201))
        logger.error("User does not have a session yet")
        return make_response(("no session", 400))
    if create:
        logger.info("User already has a session")
        return make_response(("satisfied", 100))
    return make_response(("ok", 200))

def with_session(session: Session, create=False):
    def with_session_wrapper(f):
        @wraps(f)
        def _f(*args, **kwargs):
            assertion = assert_session(session, create)
            if assertion.status_core >= 400:
                return assertion
            return f(*args, **kwargs)
        return _f
    return with_session_wrapper