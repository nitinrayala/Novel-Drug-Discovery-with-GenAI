# Locked first-predictor protocol

This implements the prediction stage of the 27 September 2026 proposal. It does
not claim to complete the seven-week adaptive-scheduling research benchmark.
Protocol written before training or examining model performance.

## Data and chemistry

- Official GSK TCAMS contributed table; keep compressed source plus SHA-256.
- Parse RDKit molecules, keep largest fragment by heavy-atom count, then atom
  count, then canonical SMILES for deterministic ties. Preserve formal charges,
  tautomers and stereochemistry (no neutralization or tautomer collapsing).
- Merge identical canonical isomeric SMILES; median each numeric endpoint across
  duplicate rows. Audit discordant duplicate threshold labels. Do not clip assay
  percentages to [0,100]; retain experimental values, including negatives.
- Known 3D7 potency >=6.3 defines eligibility. Dd2 >=80 passes; HepG2 <=20 passes.
  Missing results are unknown, never failures. Main evaluation requires both
  measured endpoints, exactly as in the proposal.
- Morgan radius 2, 2,048 binary bits, chirality enabled; append known potency.
  Standardize potency using training records only. No other source columns enter
  the predictor. The observation mask belongs in the loss, not in the inputs.
- Group by non-stereochemical Bemis-Murcko scaffold; all acyclic compounds share
  one group. Whole groups, no label stratification, approximately 40% development,
  35% replay training reservoir, 5% calibration, 20% held-out candidate families.
  Largest groups assigned first to relative size deficits, fixed seed 20260930.
  Within development: 65% train, 25% validation, 10% calibration, seed 409.
  Different Murcko scaffolds can still contain similar chemistry; separation is
  not a guarantee of absence of all near neighbors.

## Model selection and outcomes

One-hidden-layer ReLU network, initially 128 units, four softmax classes in order
(Dd2,HepG2) = (0,0), (0,1), (1,0), (1,1). Exact joint negative log likelihood for
complete records; negative log summed compatible probability for partial records.
Rows with neither observation contribute zero loss and no label information.

Tune a bounded grid of width, dropout, learning rate and weight decay on the
development validation set. Three model seeds per configuration; select by mean
joint log loss, with individual endpoint Brier scores and discrimination as
diagnostics. Select epoch counts on development only. No resampling/class weights
that would silently change probability interpretation. Ensemble three models;
temperature scale mean joint probabilities on a separate calibration set.

Baselines: training joint class frequencies (Laplace smoothing), independent
random forests with fingerprint+potency inputs, and independent logistic models.
Their joint probabilities are the product of endpoint probabilities. Tune baseline
regularization on development validation data. Calibrate by the same procedure.

Development acceptance targets for a useful first predictor: both endpoint ROC
AUC >=0.65; both endpoint Brier scores improve over training-prior predictions;
top-quintile both-pass enrichment >=1.25; joint Brier no more than 0.02 worse than
the strongest baseline. These are practical development criteria, not proof of
clinical usefulness, novelty, or budget savings. If these targets are unattainable
after reasonable development, report failure rather than manipulating the split.

Freeze model choices before reading candidate labels for evaluation. Assess five
fixed training seeds (11,23,37,53,71), three network members per seed, at initial
~200 training records, ~600, and the full replay-training reservoir. Seeds select
whole scaffold groups (nearest target without splitting groups). Separately sample
~100 calibration records by scaffold for every run. Calibration outcomes never
train the network or choose epochs. Both fully measured endpoints cost one unit
each; report actual training and calibration counts and their costs. This is a
static learning-curve diagnostic, not an adaptive replay or evidence of savings.
The held-out candidate set is identical at every training size and run.

Final acceptance uses the full-reservoir ensemble: improvement over prior Brier
for both assays, both ROC AUC >=0.65, and enrichment >=1.25, plus comparison to
the strongest baseline. Report bootstrap intervals clustered by scaffold and
variability across five training seeds. Do not tune further on final labels.
Evaluate artificial missingness on development only; it is a loss-function and
robustness check, not evidence for missing-not-at-random adaptive performance.

## Scope of claims

HepG2 pass is a cell-line counterscreen result, not human safety. Dd2 inhibition
is not a matched potency ratio or proof of resistance robustness. Generalization
applies to selected TCAMS hits; no clinical or prospective validation is claimed.
The simulator boundary, scheduler, budget comparisons, exploration and full
ablation/sensitivity experiments remain the subsequent project stages. Candidate
labels must remain held by the evaluator; future final research comparisons need
a frozen evaluation design and must not use these diagnostics to tune on them.
