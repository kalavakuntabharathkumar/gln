from gradelens.metrics import compute_metrics

def test_perfect_metrics():
    result = compute_metrics([0, 1, 2, 1], [0, 1, 2, 1])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["quadratic_kappa"] == 1.0
