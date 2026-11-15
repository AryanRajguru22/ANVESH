import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from demo_e2e import Api, credentials  # noqa: E402
def apis(base, data_dir):
    cr = credentials(Path(data_dir) / "initial_credentials.txt")
    return {u: Api(base, u, cr[u]) for u in ("admin", "investigator", "reviewer", "officer")}
