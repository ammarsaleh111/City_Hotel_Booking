"""Prepare cleaned data using shared data-handling rules."""

from pathlib import Path

import pandas as pd

from src.data.loader import load_data
from src.data.schema import (
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    NUMERIC_COLUMNS,
    SOURCE_COLUMNS
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "H2_cleaned.csv"

MISSING_TOKENS = {"", "NA", "N/A", "NULL", "None", "NaN", "nan", "Null"}
COUNTRY_ALIASES = {"CN": "CHN"}
PLACEHOLDER_COLUMNS = ("Meal", "MarketSegment", "DistributionChannel")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared normalization without row or target filtering."""
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError("The dataframe must contain the expected H2 columns in order.")

    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(
            cleaned[column]
        ):
            cleaned[column] = cleaned[column].astype("string").str.strip()
    cleaned = cleaned.replace(list(MISSING_TOKENS), pd.NA)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype("float64")
    for column in DATE_COLUMNS:
        cleaned[column] = pd.to_datetime(
            cleaned[column], format="%Y-%m-%d", errors="coerce"
        )
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype(object)
        cleaned[column] = cleaned[column].where(cleaned[column].notna(), None)

    cleaned["Country"] = cleaned["Country"].str.upper().replace(COUNTRY_ALIASES)
    cleaned["ArrivalDateMonth"] = cleaned["ArrivalDateMonth"].str.title()
    for column in ("Agent", "Company"):
        cleaned[column] = cleaned[column].str.replace(r"\.0+$", "", regex=True)
    for column in PLACEHOLDER_COLUMNS:
        cleaned[column] = cleaned[column].replace({"Undefined": "Unknown"})

    return cleaned[SOURCE_COLUMNS]


def save_clean_dataframe(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    cleaned = clean_dataframe(load_data())
    save_clean_dataframe(cleaned)
    print(f"Saved {len(cleaned):,} cleaned rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
