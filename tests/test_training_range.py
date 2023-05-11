import pytest
import pandas as pd
from main import calculate_training_range


@pytest.fixture(scope="module")
def rpe_data():
    global DF_RPE
    DF_RPE = pd.read_csv("source/rpe-calculator.csv").set_index("RPE")
    return DF_RPE


@pytest.fixture
def training_data():
    return {"one_rm": 50, "reps": 8, "rpe_schema": [6, 7, 8, 9, 9.5]}


def test_calculate_training_range(training_data):
    assert calculate_training_range(**training_data) == [35, 37.5, 40, 42.5, 45]
