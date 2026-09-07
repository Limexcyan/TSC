from collections.abc import Mapping
import logging
from functools import wraps
from uuid import uuid4

from flask import make_response

from flask_session import Session

logger = logging.getLogger(__name__)

def assert_session(session: Session, create=False):
    if not session.get("uuid"):
        if create:
            session_uuid = uuid4().hex
            session["uuid"] = session_uuid
            logger.info("Created new session %s", session_uuid)
            return make_response(({"result":"created"}, 201))
        logger.error("User does not have a session yet")
        return make_response(({"reason":"no session"}, 400))
    if create:
        logger.info("User already has a session")
        return make_response(({"reason":"satisfied"}, 100))
    return make_response(({"result":"ok"}, 200))

def with_session(session: Session, create=False):
    def with_session_wrapper(f):
        @wraps(f)
        def _f(*args, **kwargs):
            assertion = assert_session(session, create)
            if assertion.status_code >= 400:
                return assertion
            return f(*args, **kwargs)
        return _f
    return with_session_wrapper

def merge_dicts(o, n):
    for k,v in n.items():
        if isinstance(v, Mapping):
            o[k] = merge_dicts(o.get(k, {}), v)
        else:
            o[k] = v
    return o