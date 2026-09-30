"""Verify delivered data, split provenance and end-to-end saved-model inference."""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import rdBase

from .data import features, load, sha256, write_json
from .experiment import implementation_hashes
from .predict import predict


def verify(root: Path):
    table,x=load(root)
    frozen_path=root / "artifacts/frozen_selection.json"
    frozen=json.loads(frozen_path.read_text())
    assert frozen["implementation_hashes"]==implementation_hashes(root)
    assert frozen["protocol_sha256"]==sha256(root / "PROTOCOL.md")
    assert frozen["data_sha256"]==sha256(root / "data/processed/eligible_compounds.csv")
    assert table.groupby("scaffold").partition.nunique().max()==1
    np.testing.assert_array_equal(x,features(table.canonical_smiles.tolist(),table.pXC50_3D7.to_numpy()))
    candidate=table[table.partition.eq("candidate")]
    by_id=table.set_index("compound_id")
    manifest_path=root / "artifacts/model_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest["checkpoints"])==5
    checkpoint_hashes={}
    for filename in manifest["checkpoints"]:
        path=manifest_path.parent / filename
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        assert len(checkpoint["members"])==3
        info=checkpoint["metadata"]
        assert info["frozen_sha256"]==sha256(frozen_path)
        train_ids,cal_ids=info["training_ids"],info["calibration_ids"]
        assert len(set(train_ids))==info["training_count"]
        assert len(set(cal_ids))==info["calibration_count"]
        assert by_id.loc[train_ids].partition.eq("replay_train").all()
        assert by_id.loc[cal_ids].partition.eq("replay_calibration").all()
        assert set(train_ids).isdisjoint(cal_ids)
        assert set(train_ids+cal_ids).isdisjoint(candidate.compound_id)
        assert set(by_id.loc[train_ids].scaffold).isdisjoint(by_id.loc[cal_ids].scaffold)
        for member in checkpoint["members"]:
            assert member["epochs"]==frozen["config"]["epochs"]
            assert member["potency_mean"]==float(x[table.compound_id.isin(train_ids),-1].mean())
        checkpoint_hashes[filename]=sha256(path)
    tmp=root / "tmp/verification"
    tmp.mkdir(parents=True,exist_ok=True)
    input_path=tmp / "heldout_inputs.csv"
    candidate[["compound_id","canonical_smiles","pXC50_3D7"]].rename(columns={"canonical_smiles":"SMILES"}).to_csv(input_path,index=False)
    actual=predict(input_path,tmp / "heldout_outputs.csv",manifest_path)
    expected=pd.read_csv(root / "reports/heldout_predictions.csv")
    assert actual.compound_id.tolist()==expected.compound_id.tolist()
    probability_columns=["p_both_fail","p_hepg2_only","p_dd2_only","p_both_pass"]
    np.testing.assert_allclose(actual[probability_columns],expected[[f"neural_p{i}" for i in range(4)]],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(actual[probability_columns].sum(axis=1),1,atol=1e-7)
    assert np.isfinite(actual[probability_columns]).all().all()
    result={"passed":True,"candidate_predictions_verified":len(actual),"scaffold_separation_verified":True,"features_recomputed_and_matched":True,"checkpoint_training_and_calibration_provenance_verified":True,"checkpoints_sha256":checkpoint_hashes,"protocol_sha256":sha256(root / "PROTOCOL.md"),"frozen_selection_sha256":sha256(frozen_path),"source_sha256":sha256(root / "data/raw/chemblntd_gsk.txt.gz"),"python":platform.python_version(),"torch":torch.__version__,"rdkit":rdBase.rdkitVersion}
    write_json(root / "reports/artifact_verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    verify(Path("."))
