from tscapp.webapp.orchestrator.orchestrator_db import OrchestratorDB
from typing import Any
import logging

logger = logging.getLogger(__name__)

def orchestrator_init(config: dict[str, Any]):
    logger.info("loading orchestrator...")
    orchestrator_db = OrchestratorDB(config['database_path'])
    return orchestrator_db
