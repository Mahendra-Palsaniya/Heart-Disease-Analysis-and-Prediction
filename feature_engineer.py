# ===== FeatureEngineer (must be defined before loading pipeline) =====
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X['age_thalach_ratio']    = X['age']      / (X['thalach'] + 1)
        X['chol_age_interaction'] = X['chol']     *  X['age']
        X['bp_chol_ratio']        = X['trestbps'] / (X['chol'] + 1)
        return X
