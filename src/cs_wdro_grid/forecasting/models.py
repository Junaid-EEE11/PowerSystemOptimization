"""Gradient-boosted point and quantile forecast bundle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor


@dataclass
class ForecastBundle:
    """F3 point model plus lower/median/upper quantile regressors."""

    quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)
    seed: int = 0
    point_parameters: dict[str, float | int] = field(default_factory=dict)
    quantile_parameters: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        point_parameters = {
            "max_iter": 150,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.1,
            **self.point_parameters,
        }
        quantile_parameters = {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.05,
            **self.quantile_parameters,
        }
        self.point_model = HistGradientBoostingRegressor(
            loss="squared_error", random_state=self.seed, **point_parameters
        )
        self.quantile_models = {
            q: GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                random_state=self.seed,
                **quantile_parameters,
            )
            for q in self.quantiles
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ForecastBundle:
        self.point_model.fit(x, y)
        for model in self.quantile_models.values():
            model.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=x.index)
        result["point"] = self.point_model.predict(x)
        for q, model in self.quantile_models.items():
            result[f"q_{q:.2f}"] = model.predict(x)
        quantile_columns = [f"q_{q:.2f}" for q in self.quantiles]
        ordered = np.sort(result[quantile_columns].to_numpy(), axis=1)
        result.loc[:, quantile_columns] = ordered
        return result
