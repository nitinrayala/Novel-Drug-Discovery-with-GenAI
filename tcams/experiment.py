"""Separate development selection and frozen final evaluation commands."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .data import load, sha256, write_json
from .metrics import bootstrap_comparison, metrics, prior_predict
from .model import Config, Ensemble, IndependentBaseline, fit_member

SEEDS = [11, 23, 37, 53, 71]
DEVELOPMENT_SEEDS = [101, 211, 307]


def implementation_hashes(root):
    return {name: sha256(root / "tcams" / name) for name in ["data.py", "model.py", "metrics.py", "experiment.py"]}


def indexes(table, partition, development=None):
    mask = table.partition.eq(partition)
    if development:
        mask &= table.development_partition.eq(development)
    return np.flatnonzero(mask)


def labels(table):
    return table[["dd2_pass", "hepg2_pass"]].to_numpy(dtype=int)


def train_ensemble(x, y, config, seeds, validation=None, observed=None):
    if observed is None:
        observed = np.ones_like(y, dtype=bool)
    members, histories = [], []
    for seed in seeds:
        member, history = fit_member(x, y, observed, config, seed, validation)
        members.append(member)
        histories.append(history)
    return Ensemble(members), histories


def acceptance(model, prior, strongest=None):
    checks = {
        "dd2_auc_at_least_065": model["dd2_auc"] is not None and model["dd2_auc"] >= .65,
        "hepg2_auc_at_least_065": model["hepg2_auc"] is not None and model["hepg2_auc"] >= .65,
        "dd2_brier_beats_prior": model["dd2_brier"] < prior["dd2_brier"],
        "hepg2_brier_beats_prior": model["hepg2_brier"] < prior["hepg2_brier"],
        "top20_enrichment_at_least_125": model["both_top20_enrichment"] is not None and model["both_top20_enrichment"] >= 1.25,
    }
    if strongest:
        checks["joint_brier_within_002_of_strongest_baseline"] = model["joint_brier"] <= strongest["joint_brier"] + .02
    return {"passed": all(checks.values()), "checks": checks}


def tune(root: Path):
    if (root / "artifacts/frozen_selection.json").exists():
        raise RuntimeError("Selection already frozen. Preserve it; use a new experiment directory for new development.")
    table, x = load(root)
    y = labels(table)
    tr, va, ca = (indexes(table, "development", p) for p in ["train", "validation", "calibration"])
    configurations = [
        Config(width=w, dropout=d, weight_decay=wd, lr=lr)
        for w, d, wd, lr in [
            (128, .0, .01, .001), (128, .2, .01, .001),
            (128, .4, .01, .001), (128, .5, .1, .001),
            (64, .3, .01, .001), (256, .4, .01, .001),
            (128, .3, .1, .0003), (128, .3, .001, .0003),
            (64, .5, .1, .001), (256, .5, .1, .0003),
            (128, .3, .1, .003), (128, .3, .1, .001),
        ]
    ]
    folder = root / "artifacts/development"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for i, config in enumerate(configurations):
        record_path = folder / f"config_{i:02}.json"
        model_path = folder / f"config_{i:02}.pt"
        if record_path.exists() and model_path.exists():
            record = json.loads(record_path.read_text())
            if record["config"] != asdict(config) or record["data_sha256"] != sha256(root / "data/processed/eligible_compounds.csv") or record["implementation_hashes"] != implementation_hashes(root):
                raise RuntimeError("Cached development run does not match inputs")
        else:
            start = time.perf_counter()
            model, histories = train_ensemble(x[tr], y[tr], config, DEVELOPMENT_SEEDS, validation=(x[va], y[va]))
            raw = metrics(y[va], model.predict(x[va]))
            model.calibrate(x[ca], 2*y[ca,0]+y[ca,1])
            record = {"config": asdict(config), "data_sha256": sha256(root / "data/processed/eligible_compounds.csv"), "implementation_hashes": implementation_hashes(root), "raw_metrics": raw, "calibrated_metrics": metrics(y[va], model.predict(x[va])), "epochs": [m.epochs for m in model.members], "temperature": model.temperature, "seconds": time.perf_counter()-start, "histories": histories}
            model.save(model_path)
            write_json(record_path, record)
        results.append(record)
        print(f"config {i:02}: loss={record['calibrated_metrics']['joint_log_loss']:.4f}, Dd2 AUC={record['calibrated_metrics']['dd2_auc']:.3f}, HepG2 AUC={record['calibrated_metrics']['hepg2_auc']:.3f}, epochs={record['epochs']}", flush=True)
    # Selection is by mean individual-member raw validation log loss, before calibration.
    selection_scores = [np.mean([min(h["validation_nll"] for h in history) for history in r["histories"]]) for r in results]
    best_index = int(np.argmin(selection_scores))
    best = results[best_index]
    config = replace(configurations[best_index], epochs=int(np.median(best["epochs"])))
    selected_model = Ensemble.load(folder / f"config_{best_index:02}.pt")
    baselines, baseline_settings = {}, {}
    for kind, strengths in [("rf", [1, 2, 4, 8]), ("logistic", [.01, .1, 1., 10.])]:
        choices = []
        for strength in strengths:
            model = IndependentBaseline(kind, strength, 101).fit(x[tr], y[tr], np.ones_like(y[tr], dtype=bool))
            raw = metrics(y[va], model.predict(x[va]))
            model.calibrate(x[ca], 2*y[ca,0]+y[ca,1])
            choices.append((raw["joint_log_loss"], strength, metrics(y[va], model.predict(x[va]))))
        chosen = min(choices, key=lambda item: item[0])
        baseline_settings[kind] = chosen[1]
        baselines[kind] = chosen[2]
        print(f"baseline {kind}: strength={chosen[1]}, joint Brier={chosen[2]['joint_brier']:.4f}", flush=True)
    prior = metrics(y[va], prior_predict(y[tr], len(va)))
    best_baseline = min(baselines, key=lambda k: baselines[k]["joint_brier"])
    gate = acceptance(best["calibrated_metrics"], prior, baselines[best_baseline])
    summary = {"selected_index": best_index, "config": asdict(config), "selection_scores": [float(s) for s in selection_scores], "neural": best["calibrated_metrics"], "prior": prior, "baselines": baselines, "baseline_settings": baseline_settings, "acceptance": gate}
    write_json(root / "reports/development_summary.json", summary)
    selected_model.save(folder / "selected.pt", metadata={"purpose": "development only", "selection": best_index})
    print(json.dumps(summary, indent=2), flush=True)


def partial_diagnostic(root):
    table, x = load(root)
    y = labels(table)
    tr, va, ca = (indexes(table, "development", p) for p in ["train", "validation", "calibration"])
    summary = json.loads((root / "reports/development_summary.json").read_text())
    config = Config(**summary["config"])
    records = []
    for seed in [101, 211, 307]:
        pattern = np.random.default_rng(seed).choice(3, len(tr), p=[.2,.4,.4])
        observed = np.column_stack([pattern!=2, pattern!=1])
        hidden_y = y[tr].copy()
        hidden_y[~observed] = -1
        for strategy in ["partial_likelihood", "complete_only"]:
            use = np.ones(len(tr), dtype=bool) if strategy=="partial_likelihood" else observed.all(axis=1)
            model, _ = train_ensemble(x[tr][use], hidden_y[use], config, [seed], observed=observed[use])
            model.calibrate(x[ca], y[ca,0]*2+y[ca,1])
            records.append({"seed": seed, "strategy": strategy, "training_rows": int(use.sum()), "complete_records": int(observed.all(axis=1).sum()), "observed_endpoints": int(observed[use].sum()), **metrics(y[va], model.predict(x[va]))})
    pd.DataFrame(records).to_csv(root / "reports/partial_label_diagnostic.csv", index=False)
    print(pd.DataFrame(records)[["seed","strategy","joint_brier","dd2_auc","hepg2_auc"]].to_string(index=False), flush=True)


def freeze(root):
    path = root / "artifacts/frozen_selection.json"
    if path.exists():
        raise RuntimeError("Model selection is already frozen")
    summary = json.loads((root / "reports/development_summary.json").read_text())
    record = {"config": summary["config"], "baseline_settings": summary["baseline_settings"], "development_acceptance": summary["acceptance"], "protocol_sha256": sha256(root / "PROTOCOL.md"), "data_sha256": sha256(root / "data/processed/eligible_compounds.csv"), "implementation_hashes": implementation_hashes(root), "seeds": SEEDS, "model_member_offsets": [0, 1000, 2000], "training_sizes": [200, 600, "all"], "calibration_target": 100}
    write_json(path, record)
    print(f"Frozen choices: {path}", flush=True)


def scaffold_subset(table, available, target, seed):
    """A reproducible prefix of shuffled whole groups, closest to target row count."""
    groups = table.iloc[available].scaffold.to_numpy()
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(unique)
    if target >= len(available):
        return np.asarray(available)
    chunks = [available[groups==group] for group in order]
    counts = np.cumsum([len(c) for c in chunks])
    stop = int(np.argmin(np.abs(counts-target)))+1
    return np.concatenate(chunks[:stop])


def evaluate(root):
    frozen_path = root / "artifacts/frozen_selection.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen["protocol_sha256"] != sha256(root / "PROTOCOL.md") or frozen["data_sha256"] != sha256(root / "data/processed/eligible_compounds.csv"):
        raise RuntimeError("Frozen protocol or data has changed")
    if frozen["implementation_hashes"] != implementation_hashes(root):
        raise RuntimeError("Frozen training/evaluation implementation has changed")
    table, x = load(root)
    # Candidate labels enter only this evaluation routine, after choices are frozen.
    y = labels(table)
    reservoir, calibration, test = (indexes(table, p) for p in ["replay_train", "replay_calibration", "candidate"])
    config = Config(**frozen["config"])
    runs, predictions = [], []
    folder = root / "artifacts/final"
    folder.mkdir(parents=True, exist_ok=True)
    for size in frozen["training_sizes"]:
        for seed in frozen["seeds"]:
            tag = f"n{size}_seed{seed}"
            result_path, pred_path = folder / f"{tag}.json", folder / f"{tag}.npz"
            if result_path.exists() and pred_path.exists():
                result = json.loads(result_path.read_text())
                if result["frozen_sha256"] != sha256(frozen_path):
                    raise RuntimeError("Evaluation cache no longer matches frozen settings")
                with np.load(pred_path) as stored:
                    run_predictions = {k: stored[k] for k in stored.files}
            else:
                start = time.perf_counter()
                tr = scaffold_subset(table, reservoir, len(reservoir) if size=="all" else size, seed)
                ca = scaffold_subset(table, calibration, frozen["calibration_target"], seed+17)
                model, _ = train_ensemble(x[tr], y[tr], config, [seed+o for o in frozen["model_member_offsets"]])
                model.calibrate(x[ca], y[ca,0]*2+y[ca,1])
                run_predictions = {"neural": model.predict(x[test]), "neural_uncalibrated": model.raw_predict(x[test]), "prior": prior_predict(y[tr], len(test))}
                for kind, strength in frozen["baseline_settings"].items():
                    baseline = IndependentBaseline(kind, strength, seed).fit(x[tr], y[tr], np.ones_like(y[tr], dtype=bool))
                    baseline.calibrate(x[ca], y[ca,0]*2+y[ca,1])
                    run_predictions[kind] = baseline.predict(x[test])
                metadata = {"training_ids": table.iloc[tr].compound_id.tolist(), "calibration_ids": table.iloc[ca].compound_id.tolist(), "training_count": len(tr), "calibration_count": len(ca), "seed": seed, "frozen_sha256": sha256(frozen_path), "thresholds": {"potency": 6.3, "dd2": 80., "hepg2": 20.}}
                model.save(folder / f"{tag}.pt", metadata=metadata)
                result = {"size": size, "seed": seed, "training_count": len(tr), "calibration_count": len(ca), "seed_assay_cost": 2*(len(tr)+len(ca)), "candidate_count": len(test), "temperature": model.temperature, "seconds": time.perf_counter()-start, "frozen_sha256": sha256(frozen_path), "metrics": {k: metrics(y[test], p) for k,p in run_predictions.items()}}
                np.savez_compressed(pred_path, **run_predictions)
                write_json(result_path, result)
            for method, scores in result["metrics"].items():
                runs.append({k: result[k] for k in ["size","seed","training_count","calibration_count","seed_assay_cost","candidate_count","seconds"]} | {"method": method} | scores)
            predictions.append((size, seed, run_predictions))
            print(f"{tag}: neural Brier {result['metrics']['neural']['joint_brier']:.4f}, AUC Dd2 {result['metrics']['neural']['dd2_auc']:.3f}, HepG2 {result['metrics']['neural']['hepg2_auc']:.3f}", flush=True)
    frame = pd.DataFrame(runs)
    frame.to_csv(root / "reports/final_metrics.csv", index=False)
    mature = [p for size, _, p in predictions if size=="all"]
    mean_predictions = {method: np.mean([p[method] for p in mature], axis=0) for method in mature[0]}
    final_metrics = {k: metrics(y[test], p) for k,p in mean_predictions.items()}
    strongest = min(["rf","logistic"], key=lambda k: final_metrics[k]["joint_brier"])
    print("Computing scaffold-cluster bootstrap intervals...", flush=True)
    confidence = bootstrap_comparison(y[test], mean_predictions["neural"], mean_predictions[strongest], table.iloc[test].scaffold.to_numpy())
    # Each stored model retains its own calibration. Average calibrated probabilities.
    write_json(root / "artifacts/model_manifest.json", {"format_version": 1, "checkpoints": [f"final/nall_seed{s}.pt" for s in SEEDS], "aggregation": "mean_of_individually_calibrated_probabilities", "frozen_sha256": sha256(frozen_path), "intended_population": "TCAMS-like initial hits with known pXC50_3D7 >= 6.3"})
    final = {"metrics_of_mean_prediction": final_metrics, "strongest_baseline": strongest, "scaffold_bootstrap_95_percent": confidence, "acceptance": acceptance(final_metrics["neural"], final_metrics["prior"], final_metrics[strongest]), "test_count": len(test), "test_scaffolds": int(table.iloc[test].scaffold.nunique()), "training_reservoir_count": len(reservoir), "five_seed_variability": {method: {key: {"mean": float(frame.loc[(frame['size']=="all") & (frame.method==method), key].mean()), "std": float(frame.loc[(frame['size']=="all") & (frame.method==method), key].std())} for key in ["joint_brier","dd2_auc","hepg2_auc","both_top20_enrichment"]} for method in ["neural","rf","logistic"]}}
    write_json(root / "reports/final_summary.json", final)
    export = table.iloc[test][["compound_id","scaffold","dd2_pass","hepg2_pass"]].copy()
    for method, p in mean_predictions.items():
        for i in range(4):
            export[f"{method}_p{i}"] = p[:,i]
    export.to_csv(root / "reports/heldout_predictions.csv", index=False)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tune","partial","freeze","evaluate"])
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    {"tune": tune, "partial": partial_diagnostic, "freeze": freeze, "evaluate": evaluate}[args.stage](args.root)
