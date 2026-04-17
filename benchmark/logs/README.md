# Logs Directory

This directory stores runtime log files for each cleaning method.

## Naming Convention

```
{method}_{dataset}_{timestamp}.log
```

### Examples
- `simpleimputer_beers_20260113_010830.log` — SimpleImputer on the beers dataset
- `baran_adult_20260113_020000.log` — Baran on the adult dataset
- `uniclean_hospital_20260113_030000.log` — UniClean on the hospital dataset

## Method Name Reference

| Abbreviation | Full Name | Description |
|--------------|-----------|-------------|
| simpleimputer | SimpleImputer | Statistical imputation |
| mlimputer | MLImputer | ML-based imputation |
| deleteall | DeleteAll | Drop rows with missing/erroneous values |
| donothing | DoNothing | No cleaning |
| baran | Baran_Raha | Raha detection + Baran repair |
| horizon | Horizon | Functional-dependency pattern selection |
| holoclean | HoloClean | Probabilistic graphical model cleaning |
| activeclean | ActiveClean | Model-driven iterative cleaning |
| boostclean | BoostClean | Detector-repair ensemble |
| ctxpipe | CtxPipe | Context-aware data preparation |
| uniclean | UniClean | Multi-signal fusion cleaning |
| lopster | Lopster | Latent-space representation learning |

## Dataset Reference

| Dataset | Task Type | Source |
|---------|-----------|--------|
| adult | Classification | UCI |
| beers | Regression | Kaggle |
| bike | Regression | UCI |
| breast_cancer | Classification | UCI |
| har | Classification | UCI |
| mercedes | Regression | Kaggle |
| nasa | Classification | NASA |
| smartfactory | Classification | Industrial data |
| soilmoisture | Regression | Sensor data |

## Log Contents

Each log file typically contains:
1. Start time
2. Input data information (path, row count, column count)
3. Cleaning process details
4. Cleaning result statistics
5. End time and total runtime
