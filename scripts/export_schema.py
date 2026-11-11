"""Write docs/data/schema.json: the features, methods and default config the web page's run builder offers."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlexec import FEATURES, load_config  # noqa: E402
from rlexec.agents import METHODS  # noqa: E402


def schema() -> dict:
    return {"features": [{"name": k, "description": v[0]} for k, v in FEATURES.items()],
            "methods": [{"name": k, "label": v["label"], "description": v["description"]} for k, v in METHODS.items()],
            "defaults": load_config()}


if __name__ == "__main__":
    (ROOT / "docs/data/schema.json").write_text(json.dumps(schema(), indent=1))
    print("wrote docs/data/schema.json")
