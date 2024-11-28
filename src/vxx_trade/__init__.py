import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.resolve() / "data"
JSON_PATH = Path(__file__).parent.resolve() / "json"

with open(JSON_PATH / "data_generator.json") as f:
    DATAGEN_PARAMETERS = json.load(f)

with open(JSON_PATH / "strategy_exploratory.json") as f:
    exploratory_data = json.load(f)

with open(JSON_PATH / "matplotlib_style.json") as f:
    MATPLOTLIB_STYLE = json.load(f)["style"]

MATPLOTLIB_EXPLORATORY = exploratory_data["matplotlib_settings"]
EXPLORATORY_PARAMETERS = exploratory_data["parameters"]
