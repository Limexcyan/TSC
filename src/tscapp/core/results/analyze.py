from typing import Literal
import logging
from pathlib import Path

from plotly.io import to_html

from ...common.cli import CliRunnerMixin
from ...common.settings import ConfigBase, SettingsBase
from .analyze_preliminary import preliminary_analysis
from .analyze_mcs import mcs_analysis

logger = logging.getLogger(__name__)

class AnalyzeConfig(ConfigBase):
    output_dir: Path
    input_dir: Path
    analyze_mode: list[Literal[
        "preliminary",
        "mcs"
    ]] = ["preliminary"]

class AnalyzeCli(CliRunnerMixin, SettingsBase):
    analyze: AnalyzeConfig

    def cmd(self):
        analyze(config=self.analyze)

def analyze(config: AnalyzeConfig):
    input_csv = config.input_dir.glob("*.csv").__next__()
    logger.info("running analysis on %s", input_csv)
    if "preliminary" in config.analyze_mode:
        preliminary_analysis(input_csv, config.output_dir / (input_csv.stem + "_preliminary.html"))
    if "mcs" in config.analyze_mode:
        mcs_analysis(input_csv, config.output_dir / (input_csv.stem + "_mcs.html"))
    logger.info("results saved under %s", config.output_dir)

if __name__ == "__main__":
    AnalyzeCli.run()