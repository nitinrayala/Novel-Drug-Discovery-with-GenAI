"""Metrics calculated only by evaluation code, never by candidate training."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    joint = 2*y[:, 0] + y[:, 1]
    result = {"joint_log_loss": float(log_loss(joint, p, labels=[0,1,2,3])), "joint_brier": float(np.mean(np.sum((p-np.eye(4)[joint])**2, axis=1)))}
    for name, target, prediction in [("dd2", y[:, 0], p[:, 2]+p[:, 3]), ("hepg2", y[:, 1], p[:, 1]+p[:, 3]), ("both", (joint==3).astype(int), p[:, 3])]:
        result[name+"_brier"] = float(brier_score_loss(target, prediction))
        result[name+"_prevalence"] = float(target.mean())
        result[name+"_auc"] = float(roc_auc_score(target, prediction)) if len(np.unique(target))==2 else None
        result[name+"_ap"] = float(average_precision_score(target, prediction)) if target.sum() else None
    count = max(1, int(np.ceil(len(y)*.2)))
    # Stable ties preserve original ID order, which has no outcome-based ordering.
    selected = np.argsort(-p[:, 3], kind="stable")[:count]
    precision = float((joint[selected]==3).mean())
    result["both_top20_precision"] = precision
    result["both_top20_enrichment"] = precision / result["both_prevalence"] if result["both_prevalence"] else None
    return result


def prior_predict(y, n):
    counts = np.bincount(y[:,0]*2+y[:,1], minlength=4) + 1
    return np.tile(counts/counts.sum(), (n,1))


def bootstrap_comparison(y, p, baseline, groups, repeats=1000, seed=8401):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups==g) for g in unique]
    values = {k: [] for k in ["dd2_auc", "hepg2_auc", "both_auc", "both_top20_enrichment", "joint_brier_difference", "dd2_brier_difference", "hepg2_brier_difference"]}
    for _ in range(repeats):
        idx = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        a, b = metrics(y[idx], p[idx]), metrics(y[idx], baseline[idx])
        for k in values:
            value = a[k[:-11]]-b[k[:-11]] if k.endswith("_difference") else a[k]
            if value is not None:
                values[k].append(value)
    return {k: [float(v) for v in np.quantile(values[k], [.025,.975])] for k in values}
