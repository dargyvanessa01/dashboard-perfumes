from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np

# Modelo ARIMA para pronóstico de ventas
def entrenar_modelo_arima(serie_temporal):
    model = ARIMA(serie_temporal, order=(1,1,1))
    model_fit = model.fit()
    return model_fit

# Regularización con Ridge Regression
def modelo_ridge(X, y):
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(Ridge(), param_grid, cv=tscv)
    grid.fit(X, y)
    return grid.best_estimator_
