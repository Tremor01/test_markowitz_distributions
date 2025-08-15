import numpy as np
from pandas import DataFrame
from sklearn.covariance import LedoitWolf


def calculate_log_returns(data: DataFrame) -> DataFrame:
    return np.log(data / data.shift(1)).dropna()

def calculate_returns(data: DataFrame) -> DataFrame:
    return data.pct_change().dropna()
    # return (data / data.shift(1)).dropna()


def calculate_simple_covariance_matrix(returns: DataFrame) -> np.ndarray:
    return (returns.cov() * 365).values

def calculate_ledoit_wolf_covariance_matrix(returns: DataFrame) -> np.ndarray:
    lw = LedoitWolf()
    lw.fit(returns)
    cov_matrix = lw.covariance_ * 365
    return cov_matrix
