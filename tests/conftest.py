import pytest

from src.model_utils import load_model, set_seed


@pytest.fixture(scope="session")
def model_tok():
    set_seed(0)
    return load_model("gpt2")
