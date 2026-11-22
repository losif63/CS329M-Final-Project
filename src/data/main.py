import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

df = pd.read_parquet(DATA_DIR / "swechat" / "conversations.parquet")

print(df.axes)
