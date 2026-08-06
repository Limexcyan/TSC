import logging
import sys

from pydantic import BeforeValidator, ValidationError
from pydantic_settings import BaseSettings, CliApp

from typing import Annotated

from .logging import configure_logging

logger = logging.getLogger(__name__)

def _coerce_log_level(v: str) -> str:
    v = v.strip().upper()
    if v.isdigit() and (name := logging._levelToName.get(int(v))):
        return name
    if v in logging._nameToLevel.keys():
        return v
    raise ValueError(f"Invalid log level: {v}")

LogLevel = Annotated[str, BeforeValidator(_coerce_log_level)]

class CliRunnerMixin(BaseSettings, cli_avoid_json=True):
    log_level: LogLevel = "INFO"

    @classmethod
    def run(cls):
        try:
            if len(sys.argv) == 1:
                CliApp().print_help(cls)
                sys.exit(0)
            return CliApp.run(cls)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(map(str, err["loc"]))
                print(f"    {err['msg']}: {loc}")
                sys.exit(2)
        except ValueError as exc:
            print(f"    {exc}")
            sys.exit(1)
    
    def cli_cmd(self):
        configure_logging(level=self.log_level)
        logger.info("Running cli command: %s", self.__class__.__name__)
        self.cmd()
    
    def cmd(self):
        raise NotImplementedError