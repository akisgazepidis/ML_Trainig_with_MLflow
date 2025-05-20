import numpy as np
import pytest
from utils.utils import load_datasets


@pytest.mark.unit
def test_load_datasets():
    # Test with default parameters
    X_train, X_test, y_train, y_test = load_datasets()
    
    # Check if the splits are numpy arrays
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    
    # Check if the splits have correct shapes
    assert len(X_train.shape) == 2  # 2D array for features
    assert len(y_train.shape) == 1  # 1D array for target
    
    # Test split ratio (default test_size=0.2)
    total_samples = len(X_train) + len(X_test)
    ratio = len(X_test) / total_samples
    np.testing.assert_allclose(ratio, 0.2, rtol=0.05)  # 5% tolerance
    
    # Test with custom parameters
    X_train, X_test, y_train, y_test = load_datasets(test_size=0.3, random_state=123)
    total_samples = len(X_train) + len(X_test)
    ratio = len(X_test) / total_samples
    np.testing.assert_allclose(ratio, 0.3, rtol=0.05)  # 5% tolerance