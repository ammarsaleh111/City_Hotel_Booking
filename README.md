# CSE382 — Hotel Booking Demand (City Hotel H2)

## Project objective

Study the City Hotel H2 booking dataset as a university machine learning
project.

Implementation will be completed step by step for learning purposes.

## Current stage: restructuring and preprocessing

The project currently contains shared data handling and regression
preprocessing only. No ML models are implemented or trained.
Classification and clustering folders are reserved for future work.

The source layout is:

```text
src/
  data/
    loader.py
    prepare_data.py
    schema.py
  regression/
    preprocessing.py
  classification/
  clustering/
```

Prepare the cleaned dataset with:

```powershell
python -m src.data.prepare_data
```

### Preprocessing steps

1. **Load and validate the schema.** `H2.csv` is headerless, so the loader
   assigns the documented column names and fails if the shape is unexpected.
2. **Normalize text and missing values.** Whitespace is stripped and common
   tokens such as `NULL` and `N/A` become missing values. Country aliases and
   known placeholder values are normalized consistently.
3. **Convert data types.** Numeric columns become numeric, and the reservation
   date is parsed as a date. Missing feature values are retained for imputation.
4. **Preserve rows.** Duplicate rows and ADR values are not removed or filtered
   in this stage; those decisions remain for review.

### Regression preprocessing

The regression adapter in
[`src/regression/preprocessing.py`](src/regression/preprocessing.py):

1. Uses the shared cleaned dataframe.
2. Splits before fitting transformations.
3. Fits numeric and categorical transformations on training data only.
4. Imputes, scales, and encodes features. Numeric features use median imputation and
   standardization. Categorical features use an `Unknown` category for missing
   values and one-hot encoding with `handle_unknown="ignore"` so new categories
   in future data are safe.

Median imputation was chosen over mean imputation because booking counts and
lead times can be skewed. One-hot encoding was chosen over ordinal encoding
because hotel categories have no natural numeric order. Standardization is
kept in the shared pipeline because it is harmless for future linear models
and keeps feature preparation consistent; tree models can later omit it if
their experiment requires that comparison. Rare categories are grouped by
the configured `rare_category_threshold` instead of manually maintaining a
category list.

No regression model is fitted. Classification and clustering are not implemented.
