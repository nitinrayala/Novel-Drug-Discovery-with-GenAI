import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tcams.model import Config, Ensemble, conditional_probabilities, fit_member, partial_nll, temperature_scale


def test_complete_loss_equals_cross_entropy():
    torch.manual_seed(3)
    logits = torch.randn(4,4, requires_grad=True)
    labels = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    actual = partial_nll(logits, labels, torch.ones_like(labels, dtype=torch.bool))
    expected = F.cross_entropy(logits, torch.arange(4))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("endpoint,outcome,compatible", [(0,0,[0,1]),(0,1,[2,3]),(1,0,[0,2]),(1,1,[1,3])])
def test_partial_loss_marginalizes_exactly(endpoint, outcome, compatible):
    p = torch.tensor([[.1,.2,.3,.4]])
    labels = torch.full((1,2), float("nan"))
    labels[0,endpoint] = outcome
    mask = torch.zeros(1,2,dtype=torch.bool)
    mask[0,endpoint] = True
    loss = partial_nll(p.log(), labels, mask)
    torch.testing.assert_close(loss, -p[0,compatible].sum().log())


def test_hidden_values_cannot_change_loss_or_gradients():
    logits = torch.tensor([[3., -2., 0., 1.], [-1.,2.,0.,1.]], requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])
    a = partial_nll(logits, torch.tensor([[1.,0.],[0.,1.]]), mask)
    grad_a, = torch.autograd.grad(a, logits)
    b = partial_nll(logits, torch.tensor([[1.,float("nan")],[float("nan"),1.]]), mask)
    grad_b, = torch.autograd.grad(b, logits)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(grad_a,grad_b)


def test_unobserved_rows_contribute_no_gradient():
    logits = torch.randn(3,4,requires_grad=True)
    mask = torch.tensor([[True, True],[False,False],[False,False]])
    loss = partial_nll(logits, torch.tensor([[1,0],[-1,-1],[-1,-1]]), mask)
    loss.backward()
    assert torch.count_nonzero(logits.grad[1:]) == 0
    torch.testing.assert_close(loss, F.cross_entropy(logits[:1],torch.tensor([2])))
    empty = partial_nll(logits, torch.full((3,2),-1), torch.zeros(3,2,dtype=torch.bool))
    assert empty.item() == 0


def test_extreme_logits_stay_finite():
    logits = torch.tensor([[1000.,-1000.,-2000.,3000.]],requires_grad=True)
    loss = partial_nll(logits,torch.tensor([[0,-1]]),torch.tensor([[True,False]]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_reject_nonbinary_observed_labels():
    with pytest.raises(ValueError, match="binary|0 or 1"):
        partial_nll(torch.zeros(1,4), torch.tensor([[2,-1]]),torch.tensor([[True,False]]))


def test_conditionals_and_temperature_preserve_probability_contract():
    p = np.array([[.1,.2,.3,.4]])
    np.testing.assert_allclose(conditional_probabilities(p,dd2=1),[4/7])
    np.testing.assert_allclose(conditional_probabilities(p,hepg2=1),[2/3])
    np.testing.assert_allclose(conditional_probabilities(p,dd2=0),[2/3])
    np.testing.assert_allclose(conditional_probabilities(p,hepg2=0),[.75])
    assert np.isnan(conditional_probabilities(np.array([[1.,0.,0.,0.]]),dd2=1)[0])
    for t in [.5,1.,5.]:
        np.testing.assert_allclose(temperature_scale(p,t).sum(axis=1),1)


def test_train_reproducibility_hidden_row_isolation_and_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(32)
    x = rng.integers(0,2,(32,2049)).astype(np.float32)
    x[:,-1] = np.linspace(6.3,8,32)
    y = np.column_stack([x[:,0],x[:,1]]).astype(int)
    mask = np.ones_like(y,dtype=bool)
    config = Config(width=8,epochs=3,dropout=.1)
    a, _ = fit_member(x,y,mask,config,2)
    b, _ = fit_member(np.vstack([x,np.full((1,2049),99)]),np.vstack([y,[-1,-1]]),np.vstack([mask,[False,False]]),config,2)
    np.testing.assert_array_equal(a.predict(x),b.predict(x))
    ensemble = Ensemble([a],temperature=1.3)
    path = tmp_path / "model.pt"
    ensemble.save(path)
    np.testing.assert_array_equal(ensemble.predict(x),Ensemble.load(path).predict(x))
    assert a.potency_mean == pytest.approx(x[:,-1].mean())
