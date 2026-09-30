# TCAMS joint Dd2 / HepG2 neural network

One small network estimates four outcomes: both fail, only HepG2 passes, only Dd2
passes, and both pass. It takes 2,048 Morgan fingerprint bits plus known 3D7
potency. It can learn from one or two observed follow-up tests per compound.

The project implements the **first prediction stage** of
`Malarial_drug_prediction_updated_1.pdf`. The later budget-aware scheduler and full
research benchmark are separate stages. Read [PROTOCOL.md](PROTOCOL.md) for frozen
thresholds, chemical standardization, splits, model selection and evaluation rules.
After evaluation, [reports/MODEL_CARD.md](reports/MODEL_CARD.md) contains actual
performance, limitations and figures. No inference result is clinical safety advice.

## Use the trained model (PowerShell)

The local `.venv` contains the dependencies. The trained model is represented by
`artifacts/model_manifest.json` and its five checkpoint files. Keep them together.

Create a CSV with columns `SMILES` and `pXC50_3D7` (additional ID columns are kept):

```powershell
.venv/Scripts/python -m tcams.predict --input your_compounds.csv --output predictions.csv
```

Each output row contains four joint probabilities, two marginal pass probabilities,
and the probability of passing the remaining test conditional on the first passing.
`both_pass_between_seed_std` measures variation across trained ensembles; it is
not a validated confidence interval. Rows below potency 6.3 are flagged outside
the evaluated eligibility population. Invalid structures and missing/nonfinite
potency values are rejected rather than silently imputed.

## Reproduce from a fresh checkout

Python 3.11+; this run uses Python 3.13 on CPU. No paid API or GPU required.
Do not delete existing results to rerun experiments: use a fresh project copy.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python -m tcams.data
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m tcams.experiment tune
.venv/Scripts/python -m tcams.experiment partial
# Inspect reports/development_summary.json before freezing choices.
.venv/Scripts/python -m tcams.experiment freeze
.venv/Scripts/python -m tcams.experiment evaluate
.venv/Scripts/python -m tcams.report
.venv/Scripts/python -m tcams.verify
```

Data preparation downloads the official EMBL-EBI GSK contributed table only if
absent. Provenance records its URL, timestamp and SHA-256. Cleaned data, duplicate
audit, fixed split assignments and feature arrays go to `data/processed/`.
Cached training configurations and completed final runs can resume after an
interruption. Changing the frozen protocol or dataset invalidates evaluation.
Training never uses candidate labels for epoch or hyperparameter selection.

## Code map

- `tcams/data.py`: download, chemistry rules, duplicate audit, fingerprints, splits.
- `tcams/model.py`: joint network, partial-label likelihood, ensembles, calibration,
  baselines, checkpoint serialization and conditional probabilities.
- `tcams/experiment.py`: development selection, missing-label diagnostic, frozen
  evaluation across five seeds and three labelled training sizes.
- `tcams/predict.py`: inference from a CSV using saved models.
- `tcams/metrics.py`, `tcams/report.py`: probability metrics, scaffold bootstrap
  intervals, learning curves, reliability plots and model card.
- `tcams/verify.py`: recompute features, check checkpoint training/calibration
  provenance, and reproduce every held-out prediction through the inference API.
- `tests/`: mathematical, leakage, data-partition and checkpoint checks.

Large raw data, feature arrays, virtual environment and binary model artifacts are
ignored by Git but remain available locally. Reports, protocol, provenance, source
and dependency versions are kept as small reviewable artifacts.

## Interpretation

Dd2 >=80% and HepG2 <=20% are computational pass definitions. HepG2 is a cell-line
counterscreen, not human safety; Dd2 here is single-concentration inhibition, not a
matched potency ratio. This dataset is enriched for initial antimalarial hits.
Validation does not establish accuracy for arbitrary molecules, prospective
experiments or other datasets. Scaffold separation and calibration are measured
controls, not guarantees. Subsequent scheduler work must count the costs of all
training/calibration labels and compare matched budgets fairly.

Data source: [ChEMBL-NTD GSK TCAMS deposition](https://chembl.gitbook.io/chembl-ntd/downloads/deposited-set-1-gsk-tcams-dataset-20th-may-2010),
CC0; [Gamo et al., Nature (2010)](https://doi.org/10.1038/nature09107).
