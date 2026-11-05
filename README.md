# CSE382 — Hotel Booking Demand (City Hotel H2)

## Project objective

Study the City Hotel H2 booking dataset as a university machine learning
project.

Implementation will be completed step by step for learning purposes.

## Planned tasks

- Regression
- Classification
- Clustering

## ADR regression experiments

Run the two feature-set experiments independently, then compare their
cross-validation results:

```powershell
python -m src.regression.version_a_strict
python -m src.regression.version_b_borderline
python -m src.regression.compare_versions
```

The equivalent combined entry point is:

```powershell
python main.py regression --version a
python main.py regression --version b
python main.py regression --version both
```

Version A uses only booking-time features. Version B adds the documented
borderline parking-space and special-request fields. Both versions use the
same persisted train/test split and keep test metrics separate from
cross-validation-based model selection.
