import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from abc import ABC, abstractmethod

# ================================================================
# BASE CLASS
# ================================================================

class ModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.
    Every subclass must implement fit, predict, predict_dist, and summary.
    """

    def __init__(self):
        self._model      = None   # stores the fitted model object
        self._is_fitted  = False
        self.name        = "BaseModel"

    @abstractmethod
    def fit(self, X_train, y_train):
        """Fit the model on training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """
        Return predicted mean counts for X.
        Returns a np.array of shape (n,)
        """
        pass

    @abstractmethod
    def predict_dist(self, X, max_k=60):
        """
        Return full PMF for each row in X.
        Returns a np.array of shape (n, max_k+1) where
        entry [i, k] = P(Y=k | X_i)
        """
        pass

    @abstractmethod
    def get_summary(self):
        """Return a printable model summary."""
        pass

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet. Call .fit() first."
            )

    def score(self, X, y_true, metric="mae"):
        """
        Convenience scoring method.

        Parameters
        ----------
        metric : str — "mae" or "rmse"
        """
        self._check_fitted()
        y_pred = self.predict(X)
        if metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric == "rmse":
            return np.sqrt(np.mean((np.array(y_true) - y_pred) ** 2))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'mae' or 'rmse'.")


# ================================================================
# OLS WRAPPER
# ================================================================

class OLSWrapper(ModelWrapper):

    def __init__(self):
        super().__init__()
        self.name = "OLSWrapper"

    def fit(self, X_train, y_train):
        self._model     = sm.OLS(y_train, X_train).fit()
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return self._model.predict(X).values \
               if hasattr(self._model.predict(X), "values") \
               else np.array(self._model.predict(X))

    def predict_dist(self, X, max_k=60):
        """
        OLS doesn't produce a count PMF natively.
        Approximates using a discretised Normal distribution
        centred on the predicted mean with std = residual std.
        """
        self._check_fitted()
        from scipy.stats import norm

        means   = self.predict(X)
        sigma   = np.sqrt(self._model.mse_resid)
        k_vals  = np.arange(0, max_k + 1)

        pmfs = np.zeros((len(means), max_k + 1))
        for i, mu in enumerate(means):
            # P(k-0.5 < Y < k+0.5) for each k — discretised Normal
            pmfs[i] = norm.cdf(k_vals + 0.5, mu, sigma) - \
                      norm.cdf(k_vals - 0.5, mu, sigma)
            pmfs[i] = np.clip(pmfs[i], 0, None)
            pmfs[i] /= pmfs[i].sum()   # normalise to sum to 1

        return pmfs

    def get_summary(self):
        self._check_fitted()
        return self._model.summary()


# ================================================================
# POISSON WRAPPER
# ================================================================

class PoissonWrapper(ModelWrapper):

    def __init__(self):
        super().__init__()
        self.name = "PoissonWrapper"

    def fit(self, X_train, y_train):
        self._model = sm.GLM(
            y_train, X_train,
            family=sm.families.Poisson()
        ).fit()
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        preds = self._model.predict(X)
        return preds.values if hasattr(preds, "values") else np.array(preds)

    def predict_dist(self, X, max_k=60):
        """
        Uses the Poisson PMF with λ = predicted mean.
        P(Y=k) = e^{-λ} * λ^k / k!
        """
        self._check_fitted()
        from scipy.stats import poisson

        means  = self.predict(X)
        k_vals = np.arange(0, max_k + 1)

        pmfs = np.zeros((len(means), max_k + 1))
        for i, mu in enumerate(means):
            pmfs[i] = poisson.pmf(k_vals, mu)

        return pmfs

    def get_summary(self):
        self._check_fitted()
        return self._model.summary()


# ================================================================
# NEGATIVE BINOMIAL WRAPPER
# ================================================================

class NegBinWrapper(ModelWrapper):

    def __init__(self):
        super().__init__()
        self.name = "NegBinWrapper"
        self._alpha = None   # dispersion parameter — estimated during fit

    def fit(self, X_train, y_train):
        self._model = sm.GLM(
            y_train, X_train,
            family=sm.families.NegativeBinomial()
        ).fit()
        # Extract dispersion parameter alpha = 1/r
        self._alpha     = self._model.scale
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        preds = self._model.predict(X)
        return preds.values if hasattr(preds, "values") else np.array(preds)

    def predict_dist(self, X, max_k=60):
        """
        Uses the Negative Binomial PMF with:
            mean  = μ (predicted)
            r     = 1 / alpha  (estimated dispersion)
            p     = r / (r + μ)
        """
        self._check_fitted()
        from scipy.stats import nbinom

        means  = self.predict(X)
        r      = 1.0 / self._alpha
        k_vals = np.arange(0, max_k + 1)

        pmfs = np.zeros((len(means), max_k + 1))
        for i, mu in enumerate(means):
            p_param  = r / (r + mu)
            pmfs[i]  = nbinom.pmf(k_vals, n=r, p=p_param)

        return pmfs

    def get_summary(self):
        self._check_fitted()
        return self._model.summary()