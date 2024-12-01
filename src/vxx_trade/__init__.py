import json
from pathlib import Path

from _utils import MatplotlibAxesLimit, MatplotlibFigSize, YearsResearch

DATA_PATH = Path(__file__).parent.resolve() / "data"
JSON_PATH = Path(__file__).parent.resolve() / "json"

with open(JSON_PATH / "data_generator.json") as f:
    DATAGEN_PARAMETERS = json.load(f)

with open(JSON_PATH / "strategy_exploratory.json") as f:
    exploratory_data = json.load(f)

with open(JSON_PATH / "matplotlib_style.json") as f:
    MATPLOTLIB_STYLE = json.load(f)

MATPLOTLIB_EXPLORATORY = exploratory_data["matplotlib_settings"]
for col, dic in MATPLOTLIB_EXPLORATORY.items():
    dic["x_lims"] = MatplotlibAxesLimit(*dic["x_lims"])

EXPLORATORY_PARAMETERS = exploratory_data["parameters"]
EXPLORATORY_PARAMETERS["y_lims"] = MatplotlibAxesLimit(
    *EXPLORATORY_PARAMETERS["y_lims"]
)
EXPLORATORY_PARAMETERS["figsize"] = MatplotlibFigSize(
    *EXPLORATORY_PARAMETERS["figsize"]
)
EXPLORATORY_PARAMETERS["years"] = YearsResearch(*EXPLORATORY_PARAMETERS["years"])
