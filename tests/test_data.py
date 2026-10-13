import pandas as pd
import pytest
from gradelens.data import load_answers

def test_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["answer"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_answers(path)
