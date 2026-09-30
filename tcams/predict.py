"""Batch inference with the frozen checkpoint manifest."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import canonicalize, features
from .model import Ensemble, conditional_probabilities


def predict(input_path: Path, output_path: Path, manifest_path: Path):
    torch.set_num_threads(2)
    frame = pd.read_csv(input_path)
    if not {"SMILES", "pXC50_3D7"}.issubset(frame.columns):
        raise ValueError("Input CSV requires SMILES and pXC50_3D7 columns")
    if frame.empty:
        raise ValueError("Input CSV contains no compounds")
    potency = pd.to_numeric(frame.pXC50_3D7, errors="raise").to_numpy()
    if not np.isfinite(potency).all():
        raise ValueError("Known potency must be finite")
    canonical = [canonicalize(s) for s in frame.SMILES]
    x = features(canonical, potency)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported manifest")
    predictions = [Ensemble.load(manifest_path.parent / name).predict(x) for name in manifest["checkpoints"]]
    p = np.mean(predictions, axis=0)
    output = frame.copy()
    output["canonical_smiles"] = canonical
    output["eligible_by_potency"] = potency >= 6.3
    for i, name in enumerate(["p_both_fail", "p_hepg2_only", "p_dd2_only", "p_both_pass"]):
        output[name] = p[:,i]
    output["p_dd2_pass"] = p[:,2]+p[:,3]
    output["p_hepg2_pass"] = p[:,1]+p[:,3]
    output["p_hepg2_pass_given_dd2_pass"] = conditional_probabilities(p, dd2=1)
    output["p_dd2_pass_given_hepg2_pass"] = conditional_probabilities(p, hepg2=1)
    output["both_pass_between_seed_std"] = np.std([v[:,3] for v in predictions], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("artifacts/model_manifest.json"), type=Path)
    args = parser.parse_args()
    result = predict(args.input, args.output, args.manifest)
    print(f"Wrote {len(result)} predictions to {args.output}")
    if not result.eligible_by_potency.all():
        print("Some inputs are below the eligibility cutoff; their predictions are outside the evaluated population.")
