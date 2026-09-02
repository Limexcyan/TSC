from plotly.io import to_html
from .analyze_preliminary import preliminary_analysis
import pickle
from pathlib import Path
import logging
from ...common.cli import CliRunnerMixin
from ...common.settings import ConfigBase, SettingsBase


logger = logging.getLogger(__name__)

class AnalyzeConfig(ConfigBase):
    output_dir: Path
    input_csv: Path

class AnalyzeCli(CliRunnerMixin, SettingsBase):
    analyze: AnalyzeConfig

    def cmd(self):
        analyze(config=self.analyze)

def analyze(config: AnalyzeConfig):
    logger.info("running analysis on %s", config.input_csv)
    plot = preliminary_analysis(config.input_csv)
    with open(config.output_dir / (config.input_csv.stem + "_preliminary.html"), "w") as graph:
        graph.write(to_html(plot,full_html=False,include_plotlyjs="cdn"))
    logger.info("results saved under %s", config.output_dir)

if __name__ == "__main__":
    AnalyzeCli.run()