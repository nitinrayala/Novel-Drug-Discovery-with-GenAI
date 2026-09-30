from pathlib import Path

import numpy as np
import pytest

from tcams.data import canonicalize, features, group_partition, load, scaffold
from tcams.experiment import scaffold_subset


def test_fragment_canonicalization_and_acyclic_family():
    assert canonicalize("CCO.[Na+]") == "CCO"
    assert canonicalize("[Na+].OCC") == "CCO"
    assert scaffold("CCO") == scaffold("CCC") == "ACYCLIC"
    assert canonicalize("C[NH3+]") == "C[NH3+]"
    with pytest.raises(ValueError):
        canonicalize("not a molecule")


def test_scaffold_groups_never_cross_partition():
    groups = np.array(["a"]*8+["b"]*5+["c"]*4+["d"]*3+["e"]*2+["f"])
    split = group_partition(groups,{"train":.6,"test":.4},12)
    for group in np.unique(groups):
        assert len(set(split[groups==group]))==1
    np.testing.assert_array_equal(split,group_partition(groups,{"train":.6,"test":.4},12))


def test_input_features_are_binary_plus_known_potency():
    x = features(["CCO","c1ccccc1"],np.array([6.3,7.]))
    assert x.shape==(2,2049)
    assert set(np.unique(x[:,:-1])).issubset({0,1})
    np.testing.assert_allclose(x[:,-1],[6.3,7.])


@pytest.mark.skipif(not Path("data/processed/features.npz").exists(), reason="Run data preparation first")
def test_real_data_partitions_alignment_and_nested_seed_groups():
    table,x = load(Path("."))
    assert x.shape==(len(table),2049)
    assert table.compound_id.is_unique
    assert table.groupby("scaffold").partition.nunique().max()==1
    dev=table[table.partition=="development"]
    assert dev.groupby("scaffold").development_partition.nunique().max()==1
    reservoir=np.flatnonzero(table.partition.eq("replay_train"))
    a=scaffold_subset(table,reservoir,200,11)
    b=scaffold_subset(table,reservoir,600,11)
    assert set(a).issubset(b)
    for g in table.iloc[a].scaffold.unique():
        assert set(table.index[table.scaffold==g]).issubset(a)
