"""Standalone result figures and a concise, evidence-based model card."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve


def create_report(root):
    reports = root / "reports"
    summary = json.loads((reports / "final_summary.json").read_text())
    audit = json.loads((reports / "data_audit.json").read_text())
    development = json.loads((reports / "development_summary.json").read_text())
    manifest = json.loads((root / "artifacts/model_manifest.json").read_text())
    training_ids, calibration_ids = set(), set()
    for checkpoint in manifest["checkpoints"]:
        metadata = torch.load(root / "artifacts" / checkpoint, map_location="cpu", weights_only=True)["metadata"]
        training_ids.update(metadata["training_ids"])
        calibration_ids.update(metadata["calibration_ids"])
    union_cost = {"distinct_training_compounds":len(training_ids),"distinct_calibration_compounds":len(calibration_ids),"total_followup_assay_units":2*len(training_ids | calibration_ids),"offline_development_assay_units":2*audit['partitions']['development'],"development_plus_delivered_model_assay_units":2*(len(training_ids | calibration_ids)+audit['partitions']['development']),"heldout_evaluation_only_assay_units":2*audit['partitions']['candidate'],"known_3d7_cost":"declared sunk cost","note":"Union of labels used by the delivered five-run aggregate; individual-run costs are in final_metrics.csv. Development labels are a separately disclosed offline research cost; held-out evaluation labels never enter the learner."}
    (reports / "delivered_model_information_cost.json").write_text(json.dumps(union_cost,indent=2)+"\n",encoding="utf-8")
    frame = pd.read_csv(reports / "final_metrics.csv")
    predictions = pd.read_csv(reports / "heldout_predictions.csv")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
    colors={"neural":"#1463b4","rf":"#21845c","logistic":"#ae6535","prior":"#777777"}
    labels={"neural":"Joint neural network","rf":"Independent random forests","logistic":"Independent logistic models","prior":"Training class frequencies"}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,endpoint,cols in zip(axes,["dd2","hepg2"],[[2,3],[1,3]]):
        y=predictions[f"{endpoint}_pass"].to_numpy()
        for method in ["neural","rf","logistic"]:
            p=predictions[[f"{method}_p{i}" for i in cols]].sum(axis=1)
            fpr,tpr,_=roc_curve(y,p)
            auc=summary["metrics_of_mean_prediction"][method][f"{endpoint}_auc"]
            ax.plot(fpr,tpr,label=f"{labels[method]} ({auc:.3f})",color=colors[method])
        ax.plot([0,1],[0,1],"--",color="#aaaaaa")
        ax.set(title=f"{endpoint.upper()} held-out scaffold families",xlabel="False positive rate",ylabel="True positive rate")
        ax.legend(loc="lower right",fontsize=8)
    fig.suptitle("Discrimination after full-reservoir training")
    fig.tight_layout()
    fig.savefig(reports / "heldout_roc.png")
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,endpoint,cols in zip(axes,["dd2","hepg2","both"],[[2,3],[1,3],[3]]):
        y=predictions.dd2_pass.to_numpy()*predictions.hepg2_pass.to_numpy() if endpoint=="both" else predictions[f"{endpoint}_pass"].to_numpy()
        for method in ["neural","neural_uncalibrated","rf"]:
            p=predictions[[f"{method}_p{i}" for i in cols]].sum(axis=1)
            actual,predicted=calibration_curve(y,p,n_bins=8,strategy="quantile")
            ax.plot(predicted,actual,"o-",label=method,color=colors.get(method,"#bc94df"),markersize=3)
        ax.plot([0,1],[0,1],"--",color="#aaaaaa")
        ax.set(title=endpoint.upper(),xlabel="Mean predicted pass probability",ylabel="Measured pass frequency",xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
    fig.suptitle("Reliability on held-out families (quantile bins; descriptive)")
    fig.tight_layout()
    fig.savefig(reports / "reliability.png")
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,title in zip(axes,["joint_brier","dd2_auc","hepg2_auc"],["Joint Brier (lower is better)","Dd2 ROC AUC","HepG2 ROC AUC"]):
        for method in colors:
            sub=frame[frame.method==method]
            aggregated=sub.groupby("size").agg(n=("training_count","mean"),mean=(metric,"mean"),std=(metric,"std")).sort_values("n")
            ax.errorbar(aggregated.n,aggregated["mean"],yerr=aggregated["std"],fmt="o-",capsize=3,label=labels[method],color=colors[method])
        ax.set(title=title,xlabel="Fully observed training compounds")
    axes[0].legend(fontsize=7)
    fig.suptitle("Static learning curves: five runs; bars show standard deviation")
    fig.tight_layout()
    fig.savefig(reports / "learning_curves.png")
    plt.close(fig)
    scores=summary["metrics_of_mean_prediction"]
    m=scores["neural"]
    strongest=summary["strongest_baseline"]
    passed=summary["acceptance"]["passed"]
    rows=["| Model | Dd2 AUC | HepG2 AUC | Joint Brier (lower) | Both-pass top 20% precision |", "|---|---:|---:|---:|---:|"]
    for method in ["neural","rf","logistic","prior"]:
        s=scores[method]
        precision = f"{s['both_top20_precision']:.1%}" if method != "prior" else "N/A (all scores tied)"
        rows.append(f"| {labels[method]} | {s['dd2_auc']:.3f} | {s['hepg2_auc']:.3f} | {s['joint_brier']:.3f} | {precision} |")
    bootstrap=summary["scaffold_bootstrap_95_percent"]
    initial=frame[(frame['size']=="200") & (frame.method=="neural")]
    model_card=f"""# First joint predictor: measured results

**Status: {'passes' if passed else 'does not pass'} the prespecified first-predictor acceptance checks.**
This is a computational predictor of recorded TCAMS Dd2 and HepG2 outcomes,
not a clinical safety model or a completed adaptive-screening benchmark.

## Data and training

The official source has {audit['raw_rows']:,} rows. Deterministic largest-fragment
cleaning and duplicate medians produce {audit['unique_structures']:,} structures;
{audit['eligible_complete']:,} meet the frozen potency and measurement requirements.
{audit['both_pass']:,} pass both measured checks. See `data_audit.json` for counts,
duplicate conflicts, provenance, and exact splits.

The final candidate evaluation contains {summary['test_count']} compounds in
{summary['test_scaffolds']} scaffold families. Its labels were excluded from
hyperparameter selection, epoch selection, training and calibration. Development
and final partitions are disjoint by non-stereochemical Murcko scaffold.

The selected model has 2,049 inputs, one hidden layer of {development['config']['width']}
ReLU units, dropout {development['config']['dropout']}, and four softmax outputs.
It trains for {development['config']['epochs']} epochs selected using development data.
The delivered manifest averages five separately calibrated three-member ensembles.
Each full-reservoir model uses {summary['training_reservoir_count']} training compounds
and approximately 100 separate calibration compounds; actual counts and measurement
costs are in `final_metrics.csv`. No candidate labels are used to choose which of
the five runs to deliver.

The delivered five-run aggregate collectively uses {len(training_ids)} distinct
training compounds and {len(calibration_ids)} distinct calibration compounds,
costing **{union_cost['total_followup_assay_units']} follow-up assay units** at one
unit per test. This is higher than a single run's cost because its calibration
sets differ. Offline development used a separately reserved {audit['partitions']['development']}
compounds with existing labels; these are disclosed research/tuning data and not
free information acquired during a replay. Known 3D7 potency is a declared sunk cost.
Including the offline development labels gives
{union_cost['development_plus_delivered_model_assay_units']} modelling assay units
in total. Another {union_cost['heldout_evaluation_only_assay_units']} recorded
outcomes are used only for held-out evaluation, never by the learner.

## Held-out performance

{chr(10).join(rows)}

Both-pass prevalence in the candidate pool is {m['both_prevalence']:.1%}.
The neural model's top-quintile enrichment is {m['both_top20_enrichment']:.2f}x.
Point estimates above are for averages of calibrated predictions across five runs.
Per-run mean and standard deviation are separately reported in `final_summary.json`.
All prior scores tie: its raw-CSV top-quintile value uses deterministic ordering
and should not be interpreted as a random-ranking estimate. Random selection has
expected precision equal to the 40.7% candidate prevalence.

Scaffold-cluster bootstrap 95% intervals (1,000 replicates, conditional on the
trained models; these do not include all model-training uncertainty):

- Dd2 AUC: {bootstrap['dd2_auc'][0]:.3f} to {bootstrap['dd2_auc'][1]:.3f}.
- HepG2 AUC: {bootstrap['hepg2_auc'][0]:.3f} to {bootstrap['hepg2_auc'][1]:.3f}.
- Top-quintile enrichment: {bootstrap['both_top20_enrichment'][0]:.2f}x to {bootstrap['both_top20_enrichment'][1]:.2f}x.
- Neural minus {labels[strongest].lower()} joint Brier:
  {bootstrap['joint_brier_difference'][0]:+.3f} to {bootstrap['joint_brier_difference'][1]:+.3f}; negative favors neural.

See `heldout_roc.png`, `reliability.png`, and `learning_curves.png`.
The learning curve uses more fully purchased labels at each size, not adaptive
selection. Those labels carry the costs reported in the CSV. Small-seed quality
must be assessed separately from the full-reservoir result.

At the initial approximately 200 labelled training compounds, neural Dd2 AUC
ranged from {initial.dd2_auc.min():.3f} to {initial.dd2_auc.max():.3f}, and HepG2 AUC
from {initial.hepg2_auc.min():.3f} to {initial.hepg2_auc.max():.3f}. Some initial runs
therefore fail the quality target. The delivered model's readiness does not imply
that every small-seed model is equally reliable.

Joint Brier for the uncalibrated neural aggregate was
{scores['neural_uncalibrated']['joint_brier']:.3f}, compared with
{scores['neural']['joint_brier']:.3f} after calibration. Calibration was fixed using
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
"""
    (reports / "MODEL_CARD.md").write_text(model_card,encoding="utf-8")
    print(f"Wrote model card and three figures to {reports}")


if __name__=="__main__":
    create_report(Path("."))
