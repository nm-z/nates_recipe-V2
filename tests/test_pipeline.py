import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

from config import CONFIG, DATASET_FILES
from utils import load_dataset, validate_dataset_files
from optimizer import SystematicOptimizer, BattleTestedOptimizer

@pytest.mark.parametrize("dataset_id", list(CONFIG["DATASET_ID_MAP"].keys()))
def test_load_dataset(dataset_id):
    dataset_key = CONFIG["DATASET_ID_MAP"][dataset_id]
    assert validate_dataset_files(dataset_key)
    X, y = load_dataset(dataset_key)
    assert X.size > 0
    assert y.size > 0


def test_systematic_optimizer_single_trial():
    dataset_key = CONFIG["DATASET_ID_MAP"][1]
    X, y = load_dataset(dataset_key)
    opt = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)
    results = opt.run_systematic_optimization(X, y)
    assert "test_r2" in results


def test_battle_tested_optimizer_single_trial():
    dataset_key = CONFIG["DATASET_ID_MAP"][1]
    X, y = load_dataset(dataset_key)
    opt = BattleTestedOptimizer(dataset_num=1, max_trials=1)
    results = opt.run_optimization(X, y)
    assert "test_r2" in results


def test_load_dataset_invalid_id():
    """Ensure invalid dataset identifiers raise a ValueError."""
    with pytest.raises(ValueError):
        load_dataset(999)
