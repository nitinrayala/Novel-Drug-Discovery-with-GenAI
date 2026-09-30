# First joint predictor: measured results

**Status: passes the prespecified first-predictor acceptance checks.**
This is a computational predictor of recorded TCAMS Dd2 and HepG2 outcomes,
not a clinical safety model or a completed adaptive-screening benchmark.

## Data and training

The official source has 13,519 rows. Deterministic largest-fragment
cleaning and duplicate medians produce 13,461 structures;
3,473 meet the frozen potency and measurement requirements.
1,224 pass both measured checks. See `data_audit.json` for counts,
duplicate conflicts, provenance, and exact splits.

The final candidate evaluation contains 695 compounds in
400 scaffold families. Its labels were excluded from
hyperparameter selection, epoch selection, training and calibration. Development
and final partitions are disjoint by non-stereochemical Murcko scaffold.

The selected model has 2,049 inputs, one hidden layer of 64
ReLU units, dropout 0.5, and four softmax outputs.
It trains for 7 epochs selected using development data.
The delivered manifest averages five separately calibrated three-member ensembles.
Each full-reservoir model uses 1215 training compounds
and approximately 100 separate calibration compounds; actual counts and measurement
costs are in `final_metrics.csv`. No candidate labels are used to choose which of
the five runs to deliver.

The delivered five-run aggregate collectively uses 1215 distinct
training compounds and 174 distinct calibration compounds,
costing **2778 follow-up assay units** at one
unit per test. This is higher than a single run's cost because its calibration
sets differ. Offline development used a separately reserved 1389
compounds with existing labels; these are disclosed research/tuning data and not
free information acquired during a replay. Known 3D7 potency is a declared sunk cost.
Including the offline development labels gives
5556 modelling assay units
in total. Another 1390 recorded
outcomes are used only for held-out evaluation, never by the learner.

## Held-out performance

| Model | Dd2 AUC | HepG2 AUC | Joint Brier (lower) | Both-pass top 20% precision |
|---|---:|---:|---:|---:|
| Joint neural network | 0.735 | 0.767 | 0.589 | 74.1% |
| Independent random forests | 0.672 | 0.806 | 0.584 | 69.1% |
| Independent logistic models | 0.733 | 0.766 | 0.599 | 70.5% |
| Training class frequencies | 0.500 | 0.500 | 0.703 | N/A (all scores tied) |

Both-pass prevalence in the candidate pool is 40.7%.
The neural model's top-quintile enrichment is 1.82x.
Point estimates above are for averages of calibrated predictions across five runs.
Per-run mean and standard deviation are separately reported in `final_summary.json`.
All prior scores tie: its raw-CSV top-quintile value uses deterministic ordering
and should not be interpreted as a random-ranking estimate. Random selection has
expected precision equal to the 40.7% candidate prevalence.

Scaffold-cluster bootstrap 95% intervals (1,000 replicates, conditional on the
trained models; these do not include all model-training uncertainty):

- Dd2 AUC: 0.669 to 0.793.
- HepG2 AUC: 0.708 to 0.824.
- Top-quintile enrichment: 1.60x to 2.08x.
- Neural minus independent random forests joint Brier:
  -0.016 to +0.025; negative favors neural.

See `heldout_roc.png`, `reliability.png`, and `learning_curves.png`.
The learning curve uses more fully purchased labels at each size, not adaptive
selection. Those labels carry the costs reported in the CSV. Small-seed quality
must be assessed separately from the full-reservoir result.

At the initial approximately 200 labelled training compounds, neural Dd2 AUC
ranged from 0.622 to 0.692, and HepG2 AUC
from 0.624 to 0.731. Some initial runs
therefore fail the quality target. The delivered model's readiness does not imply
that every small-seed model is equally reliable.

Joint Brier for the uncalibrated neural aggregate was
0.590, compared with
0.589 after calibration. Calibration was fixed using
separate data; it is retained as prespecified even when the held-out score worsens.
No test-based choice between calibrated and uncalibrated models was made.

## Partial records and validation

`partial_label_diagnostic.csv` compares marginal-likelihood learning with discarding
partial records on development data, across three artificial mask seeds. This is
a missingness stress test, not proof for adaptive missing-not-at-random data.
Automated tests verify exact complete/partial likelihoods, numerical stability,
hidden-label invariance, zero influence of unobserved rows, scaffold separation,
deterministic training, and saved-model prediction parity.

## Limits and next research stages

- HepG2 pass means inhibition <=20%, not human safety. Dd2 pass means inhibition
  >=80%, not a resistant/sensitive potency ratio. Known 3D7 potency must be >=6.3
  for the evaluated population. No arbitrary-molecule accuracy is established.
- Murcko separation does not exclude every similar chemical neighbor. Calibration
  may shift under acquisition or family shift; plots and Brier scores are empirical.
- Neural superiority is not a completion requirement: the baseline table must be
  retained even if a simpler model is stronger. AUC does not establish budget savings.
- Hidden-result replay, fixed routes, compound-only active learning, conditional
  scheduling, exploration, paired budget comparisons and sensitivity analyses are
  subsequent stages of the proposal and are not claimed as completed here.
- Do not tune on these final candidate labels. Any further model development must
  use the existing development families or a newly declared independent evaluation.

Source: [official ChEMBL-NTD GSK deposition](https://chembl.gitbook.io/chembl-ntd/downloads/deposited-set-1-gsk-tcams-dataset-20th-may-2010),
[Gamo et al., Nature (2010)](https://doi.org/10.1038/nature09107).
