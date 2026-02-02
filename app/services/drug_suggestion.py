import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "faers_q3_ps_drug_adverse_events_serious_labeled_v2.csv"

# Load once when server starts
_df = pd.read_csv(DATA_PATH, usecols=["drugname"])

DRUG_LIST = (
    _df["drugname"]
    .dropna()
    .str.upper()
    .unique()
    .tolist()
)

def get_drug_suggestions(prefix: str, limit: int = 10):
    prefix = prefix.upper().strip()

    if len(prefix) < 2:
        return []

    return [d for d in DRUG_LIST if d.startswith(prefix)][:limit]
