
from pathlib import Path
import pandas as pd
from schema import SOURCE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "H2.csv"


def load_data() -> pd.DataFrame:
   

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Place H2.csv inside data/raw/."
        )

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        dtype="string",
        keep_default_na=False,
        skipinitialspace=True,
    )

    df.columns = SOURCE_COLUMNS
    return df


if __name__ == "__main__":
    df = load_data()
