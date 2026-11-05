
import json
from pathlib import Path

import pandas as pd

from loader import load_data
from schema import (
    SOURCE_COLUMNS,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    REGRESSION_TARGET,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "H2_cleaned.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "H2_cleaned.parquet"
OUTPUT_PATH = OUTPUT_CSV
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "preprocessing" / "cleaning_summary.json"


MISSING_TOKENS = ["NULL", "null", "Null", "NA", "N/A", "None", "nan", "NaN", ""]

DROP_ZERO_GUEST_ROWS = True
DROP_ZERO_NIGHT_ROWS = False
DROP_DUPLICATES = True

ADR_MAX_VALID = 1000.0

COUNTRY_ALIASES = {"CN": "CHN"}
PLACEHOLDER_COLUMNS = ["Meal", "MarketSegment", "DistributionChannel"]
PLACEHOLDER_VALUE = "Undefined"
PLACEHOLDER_REPLACEMENT = "Unknown"
ID_COLUMNS = ["Agent", "Company"]

VALID_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
BINARY_COLUMNS = ["IsCanceled", "IsRepeatedGuest"]
NON_NEGATIVE_COLUMNS = [c for c in NUMERIC_COLUMNS if c != REGRESSION_TARGET]


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _strip_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if _is_text(df[column]):
            df[column] = df[column].str.strip()
    return df


def _zero_guest_mask(df: pd.DataFrame) -> pd.Series:
    return df[["Adults", "Children", "Babies"]].fillna(0).sum(axis=1) == 0


def _zero_night_mask(df: pd.DataFrame) -> pd.Series:
    return df[["StaysInWeekendNights", "StaysInWeekNights"]].fillna(0).sum(axis=1) == 0


def inspect_data(df: pd.DataFrame) -> None:
    stripped = _strip_text(df)
    missing_mask = stripped.isna() | stripped.isin(MISSING_TOKENS)
    numeric_view = stripped[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    print("\n========== DATA OVERVIEW ==========")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES (including NULL/NA tokens) ==========")
    missing = pd.DataFrame({
        "missing_count": missing_mask.sum(),
        "missing_percentage": (missing_mask.mean() * 100).round(2),
    })
    missing = missing[missing["missing_count"] > 0]
    print(missing if not missing.empty else "No missing values detected.")

    print("\n========== DUPLICATE ROWS ==========")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\n========== LOGICAL VALIDITY CHECKS ==========")
    print(f"Rows with zero adults+children+babies: {_zero_guest_mask(numeric_view).sum():,}")
    print(f"Rows with zero total nights:           {_zero_night_mask(numeric_view).sum():,}")
    negatives = {c: int((numeric_view[c] < 0).sum()) for c in NON_NEGATIVE_COLUMNS}
    negatives = {c: n for c, n in negatives.items() if n}
    print(f"Negative values in non-negative columns: {negatives if negatives else 'none'}")

    print("\n========== ADR TARGET ==========")
    adr = numeric_view[REGRESSION_TARGET]
    print(f"Missing or non-numeric ADR: {adr.isna().sum():,}")
    print(f"Negative ADR values: {(adr < 0).sum():,}")
    print(f"Zero ADR values: {(adr == 0).sum():,}")
    print(f"ADR above cap ({ADR_MAX_VALID:g}): {(adr > ADR_MAX_VALID).sum():,}")
    print("\nADR statistical summary:")
    print(adr.describe(percentiles=[0.25, 0.50, 0.75, 0.95, 0.99, 0.999]))
    print("\nFive largest ADR values:")
    print(adr.nlargest(5).to_string())

    print("\n========== CATEGORICAL PROFILE ==========")
    for column in CATEGORICAL_COLUMNS:
        values = stripped[column].mask(missing_mask[column])
        counts = values.value_counts(dropna=True)
        coverage = counts.cumsum() / counts.sum()
        n_for_90 = int((coverage < 0.90).sum()) + 1
        undefined = int((values == PLACEHOLDER_VALUE).sum())
        print(
            f"{column}: {counts.shape[0]} unique | "
            f"missing {int(values.isna().sum()):,} | "
            f"'{PLACEHOLDER_VALUE}' {undefined:,} | "
            f"categories covering 90% of rows: {n_for_90}"
        )
        print("   top 5:", counts.head(5).to_dict())


def clean_data(df: pd.DataFrame):
    print("\n========== CLEANING ==========")
    summary = {"rows_loaded": int(len(df))}

    df = _strip_text(df)
    df = df.mask(df.isin(MISSING_TOKENS))

    # count values that become NaN because of the conversion
    coercion_failures = {}
    for column in NUMERIC_COLUMNS:
        before = int(df[column].isna().sum())
        df[column] = pd.to_numeric(df[column], errors="coerce")
        newly_missing = int(df[column].isna().sum()) - before
        if newly_missing > 0:
            coercion_failures[column] = newly_missing
    for column in DATE_COLUMNS:
        before = int(df[column].isna().sum())
        df[column] = pd.to_datetime(df[column], errors="coerce")
        newly_missing = int(df[column].isna().sum()) - before
        if newly_missing > 0:
            coercion_failures[column] = newly_missing
    summary["coercion_failures"] = coercion_failures
    print(f"Values turned into missing by type conversion: "
          f"{coercion_failures if coercion_failures else 'none'}")

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("string")

    df["Country"] = df["Country"].str.upper()
    alias_rows = int(df["Country"].isin(COUNTRY_ALIASES.keys()).sum())
    df["Country"] = df["Country"].replace(COUNTRY_ALIASES)
    df["ArrivalDateMonth"] = df["ArrivalDateMonth"].str.title()
    for column in ID_COLUMNS:  # "9.0" -> "9"
        df[column] = df[column].str.replace(r"\.0+$", "", regex=True)

    placeholder_rows = {}
    for column in PLACEHOLDER_COLUMNS:
        count = int((df[column] == PLACEHOLDER_VALUE).sum())
        if count:
            placeholder_rows[column] = count
        df[column] = df[column].replace({PLACEHOLDER_VALUE: PLACEHOLDER_REPLACEMENT})
    summary["country_alias_rows_merged"] = alias_rows
    summary["placeholder_rows_renamed"] = placeholder_rows
    print(f"Country alias rows merged (e.g. CN -> CHN): {alias_rows:,}")
    print(f"'{PLACEHOLDER_VALUE}' values renamed to '{PLACEHOLDER_REPLACEMENT}': "
          f"{placeholder_rows if placeholder_rows else 'none'}")

    zero_guest = _zero_guest_mask(df)
    summary["zero_guest_rows_found"] = int(zero_guest.sum())
    summary["zero_guest_rows_dropped"] = int(zero_guest.sum()) if DROP_ZERO_GUEST_ROWS else 0
    print(f"Zero-guest rows found: {zero_guest.sum():,} "
          f"({'dropped' if DROP_ZERO_GUEST_ROWS else 'kept'})")
    if DROP_ZERO_GUEST_ROWS:
        df = df.loc[~zero_guest].copy()

    zero_night = _zero_night_mask(df)
    summary["zero_night_rows_found"] = int(zero_night.sum())
    summary["zero_night_rows_dropped"] = int(zero_night.sum()) if DROP_ZERO_NIGHT_ROWS else 0
    print(f"Zero-night rows found: {zero_night.sum():,} "
          f"({'dropped' if DROP_ZERO_NIGHT_ROWS else 'kept'})")
    if DROP_ZERO_NIGHT_ROWS:
        df = df.loc[~zero_night].copy()

    # duplicates are checked after normalization so aliases collapse too
    duplicate_count = int(df.duplicated().sum())
    summary["duplicate_rows_found"] = duplicate_count
    summary["duplicate_rows_dropped"] = duplicate_count if DROP_DUPLICATES else 0
    print(f"Exact duplicate rows found: {duplicate_count:,} "
          f"({'dropped' if DROP_DUPLICATES else 'kept'})")
    if DROP_DUPLICATES:
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    summary["rows_final"] = int(len(df))
    summary["config"] = {
        "DROP_ZERO_GUEST_ROWS": DROP_ZERO_GUEST_ROWS,
        "DROP_ZERO_NIGHT_ROWS": DROP_ZERO_NIGHT_ROWS,
        "DROP_DUPLICATES": DROP_DUPLICATES,
        "ADR_MAX_VALID": ADR_MAX_VALID,
    }
    print(f"Rows remaining after cleaning: {len(df):,}")
    print("ADR rows are NOT filtered here; the regression pipeline applies "
          "filter_regression_rows().")

    return df, summary


def filter_regression_rows(df: pd.DataFrame, adr_max: float = ADR_MAX_VALID) -> pd.DataFrame:
    """Drop missing, negative and too-large ADR. Used by the regression pipeline only."""

    target = df[REGRESSION_TARGET]
    missing = target.isna()
    negative = target < 0
    too_large = target > adr_max
    keep = ~(missing | negative | too_large)

    print("\n---------- REGRESSION ROW FILTER ----------")
    print(f"Missing ADR:         {int(missing.sum()):,}")
    print(f"Negative ADR:        {int(negative.sum()):,}")
    print(f"ADR above {adr_max:g}: {int(too_large.sum()):,}")
    print(f"Rows kept for regression: {int(keep.sum()):,} of {len(df):,}")

    return df.loc[keep].copy()


def validate_data(df: pd.DataFrame) -> None:
    print("\n========== VALIDATION ==========")

    if df.columns.tolist() != SOURCE_COLUMNS:
        raise ValueError(
            "Schema validation failed: columns are missing, extra, or out of order."
        )
    print("PASS: The dataset has the expected columns in the expected order.")

    wrong_numeric = [
        c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if wrong_numeric:
        raise TypeError(f"These columns are not numeric: {wrong_numeric}")
    print("PASS: All declared numeric columns have numeric data types.")

    wrong_dates = [
        c for c in DATE_COLUMNS if not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if wrong_dates:
        raise TypeError(f"These columns are not dates: {wrong_dates}")
    print("PASS: Date columns have datetime data types.")

    wrong_categorical = [
        c for c in CATEGORICAL_COLUMNS if not pd.api.types.is_string_dtype(df[c])
    ]
    if wrong_categorical:
        raise TypeError(f"These columns are not string-typed: {wrong_categorical}")
    print("PASS: Categorical columns are string-typed.")

    for column in BINARY_COLUMNS:
        bad = ~df[column].dropna().isin([0, 1])
        if bad.any():
            raise ValueError(f"{column} contains values other than 0/1.")
    print("PASS: Binary columns contain only 0/1.")

    negatives = {c: int((df[c] < 0).sum()) for c in NON_NEGATIVE_COLUMNS}
    negatives = {c: n for c, n in negatives.items() if n}
    if negatives:
        raise ValueError(f"Negative values in non-negative columns: {negatives}")
    print("PASS: No negative values in count/duration columns.")

    if df["ArrivalDateMonth"].isna().any() or not df["ArrivalDateMonth"].isin(VALID_MONTHS).all():
        raise ValueError("ArrivalDateMonth contains missing or unrecognised month names.")
    print("PASS: ArrivalDateMonth contains only valid month names.")

    for column in CATEGORICAL_COLUMNS:
        if df[column].isin(MISSING_TOKENS).any():
            raise ValueError(f"{column} still contains missing-value tokens.")
    if (df["Country"] == "CN").any():
        raise ValueError("Country alias CN was not merged into CHN.")
    for column in PLACEHOLDER_COLUMNS:
        if (df[column] == PLACEHOLDER_VALUE).any():
            raise ValueError(f"{column} still contains '{PLACEHOLDER_VALUE}'.")
    print("PASS: No missing tokens, country aliases, or 'Undefined' placeholders remain.")

    if DROP_ZERO_GUEST_ROWS and _zero_guest_mask(df).any():
        raise ValueError("Zero-guest rows remain although they should be dropped.")
    if DROP_DUPLICATES and df.duplicated().any():
        raise ValueError("Duplicate rows remain although they should be dropped.")
    print("PASS: Row-level cleaning rules were applied.")

    adr = df[REGRESSION_TARGET]
    print(
        f"\nADR rows the regression pipeline will drop: "
        f"missing={int(adr.isna().sum()):,}, negative={int((adr < 0).sum()):,}, "
        f"above {ADR_MAX_VALID:g}={int((adr > ADR_MAX_VALID).sum()):,}"
    )

    remaining_missing = df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    print("\nMissing values remaining (imputed later, inside the pipeline):")
    print(remaining_missing if not remaining_missing.empty else "No missing values.")

    print("\nFinal dataset shape:", df.shape)
    print("Validation completed successfully.")


def save_cleaned_data(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCleaned dataset saved to: {OUTPUT_CSV}")

    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"Typed copy saved to:      {OUTPUT_PARQUET}")
    except (ImportError, ValueError) as error:
        print(f"Parquet copy skipped ({error}). Install pyarrow to keep dtypes exactly.")


def load_cleaned_data() -> pd.DataFrame:
    if OUTPUT_PARQUET.exists():
        return pd.read_parquet(OUTPUT_PARQUET)
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(
            f"{OUTPUT_CSV} not found. Run prepare_data.py first."
        )

    df = pd.read_csv(OUTPUT_CSV, dtype={c: "string" for c in CATEGORICAL_COLUMNS})
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df[SOURCE_COLUMNS]


def save_summary(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Cleaning summary saved to: {SUMMARY_PATH}")


def main() -> None:
    df = load_data()
    inspect_data(df)
    cleaned_df, summary = clean_data(df)
    validate_data(cleaned_df)
    filter_regression_rows(cleaned_df)  # preview only, not saved
    save_cleaned_data(cleaned_df)
    save_summary(summary)


if __name__ == "__main__":
    main()