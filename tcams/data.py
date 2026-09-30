"""Versioned source acquisition, deterministic chemistry, and scaffold partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

SOURCE = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLNTD/set1_gsk/chemblntd_gsk.txt.gz"
ENDPOINTS = ["pXC50_3D7", "PCT_INHB_DD2", "PCT_INHIB_HEPG2"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    # Deterministic tie breaking; retain charge, tautomer, and stereochemistry.
    fragment = sorted(fragments, key=lambda m: (-m.GetNumHeavyAtoms(), -m.GetNumAtoms(), Chem.MolToSmiles(m)))[0]
    return Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)


def scaffold(smiles: str) -> str:
    # Ignore scaffold stereochemistry so stereoisomeric families cannot cross splits.
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "ACYCLIC"


def features(smiles: list[str], potency: np.ndarray) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fp = np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles])
    return np.column_stack([fp.astype(np.float32), np.asarray(potency, dtype=np.float32)])


def group_partition(groups, fractions, seed):
    """Whole-scaffold assignment, balanced by row count, without using labels."""
    groups = np.asarray(groups)
    names = list(fractions)
    if not np.isclose(sum(fractions.values()), 1):
        raise ValueError("Partition fractions must sum to one")
    keys, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    order = order[np.argsort(-counts[order], kind="stable")]
    targets = np.array(list(fractions.values())) * len(groups)
    totals = np.zeros(len(names), dtype=int)
    assignment = {}
    for i in order:
        # Largest remaining relative deficit, random tie-break fixed by seed.
        bucket = int(np.argmax((targets - totals) / np.maximum(targets, 1)))
        assignment[keys[i]] = names[bucket]
        totals[bucket] += counts[i]
    return np.array([assignment[g] for g in groups])


def prepare(root: Path) -> dict:
    raw = root / "data/raw/chemblntd_gsk.txt.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        with urllib.request.urlopen(SOURCE, timeout=120) as response:
            payload = response.read()
        raw.write_bytes(payload)
    provenance = raw.with_suffix(".provenance.json")
    if provenance.exists():
        if json.loads(provenance.read_text())["sha256"] != sha256(raw):
            raise ValueError("Raw source differs from its recorded checksum")
    else:
        write_json(provenance, {"source_url": SOURCE, "sha256": sha256(raw), "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "bytes": raw.stat().st_size, "license": "CC0 (GSK deposition)"})
    df = pd.read_csv(raw, sep="\t", compression="gzip")
    required = ["COMPOUND_ID", "SMILES", *ENDPOINTS]
    if not set(required).issubset(df.columns):
        raise ValueError("TCAMS source schema changed")
    raw_complete = int(df[ENDPOINTS].notna().all(axis=1).sum())
    df = df[required].copy()
    for col in ENDPOINTS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    canonical, invalid = [], []
    for idx, s in enumerate(df.SMILES):
        try:
            canonical.append(canonicalize(s))
        except (ValueError, TypeError):
            canonical.append(None)
            invalid.append(str(df.iloc[idx].COMPOUND_ID))
    df["canonical_smiles"] = canonical
    valid = df.dropna(subset=["canonical_smiles"])
    duplicate_rows = valid[valid.duplicated("canonical_smiles", keep=False)].copy()
    clean = valid.groupby("canonical_smiles", sort=True)[ENDPOINTS].median().reset_index()
    ids = valid.groupby("canonical_smiles").COMPOUND_ID.agg(lambda s: "|".join(sorted(s.astype(str))))
    clean["source_ids"] = clean.canonical_smiles.map(ids)
    clean["compound_id"] = clean.canonical_smiles.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:20])
    clean["scaffold"] = clean.canonical_smiles.map(scaffold)
    clean["eligible"] = clean["pXC50_3D7"].ge(6.3)
    complete = clean[ENDPOINTS].notna().all(axis=1)
    eligible = clean[clean.eligible & complete].copy().reset_index(drop=True)
    eligible["dd2_pass"] = eligible.PCT_INHB_DD2.ge(80).astype(int)
    eligible["hepg2_pass"] = eligible.PCT_INHIB_HEPG2.le(20).astype(int)
    eligible["joint_class"] = 2 * eligible.dd2_pass + eligible.hepg2_pass
    eligible["partition"] = group_partition(eligible.scaffold, {"development": .4, "replay_train": .35, "replay_calibration": .05, "candidate": .2}, 20260930)
    dev = eligible.partition.eq("development")
    eligible["development_partition"] = ""
    eligible.loc[dev, "development_partition"] = group_partition(eligible.loc[dev, "scaffold"], {"train": .65, "validation": .25, "calibration": .1}, 409)
    output = root / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output / "cleaned_compounds.csv", index=False)
    duplicate_rows.to_csv(output / "duplicate_audit.csv", index=False)
    eligible.to_csv(output / "eligible_compounds.csv", index=False)
    x = features(eligible.canonical_smiles.tolist(), eligible.pXC50_3D7.to_numpy())
    np.savez_compressed(output / "features.npz", x=x, compound_id=eligible.compound_id.to_numpy(dtype=str))
    conflicts = {}
    for col, cutoff, greater in [("PCT_INHB_DD2", 80, True), ("PCT_INHIB_HEPG2", 20, False)]:
        temp = valid.dropna(subset=[col]).copy()
        temp["pass"] = temp[col].ge(cutoff) if greater else temp[col].le(cutoff)
        conflicts[col] = int(temp.groupby("canonical_smiles")["pass"].nunique().gt(1).sum())
    audit = {"raw_rows": len(df), "raw_complete_rows": raw_complete, "invalid_smiles": invalid, "unique_structures": len(clean), "complete_unique_structures": int(complete.sum()), "eligible_complete": len(eligible), "both_pass": int(eligible.joint_class.eq(3).sum()), "scaffolds": int(eligible.scaffold.nunique()), "acyclic_eligible": int(eligible.scaffold.eq("ACYCLIC").sum()), "duplicate_extra_rows": len(valid)-len(clean), "duplicate_threshold_conflicts": conflicts, "partitions": eligible.partition.value_counts().to_dict(), "development_partitions": eligible.loc[dev, "development_partition"].value_counts().to_dict(), "source_sha256": sha256(raw), "eligible_sha256": sha256(output / "eligible_compounds.csv"), "rdkit_version": rdBase.rdkitVersion, "feature_count": x.shape[1], "endpoint_ranges": {c: [float(clean[c].min()), float(clean[c].max())] for c in ENDPOINTS}}
    write_json(root / "reports/data_audit.json", audit)
    return audit


def load(root: Path):
    table = pd.read_csv(root / "data/processed/eligible_compounds.csv", keep_default_na=False)
    with np.load(root / "data/processed/features.npz") as packed:
        if not np.array_equal(packed["compound_id"], table.compound_id.astype(str)):
            raise ValueError("Feature/compound alignment mismatch")
        x = packed["x"]
    return table, x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    print(json.dumps(prepare(parser.parse_args().root), indent=2))
