import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm
from sklearn.linear_model import LassoCV, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from model_wrapper import ModelWrapper, OLSWrapper, PoissonWrapper, NegBinWrapper

class CountRegressor:
    def __init__(self, X, y, min_train, retrain_every):

        # Core data
        self.X = X
        self.y = y
        self.min_train = min_train
        self.retrain_every = retrain_every

        # --- Active features: starts as all columns ---
        self.active_features = list(X.columns)

        # --- Internal state ---
        self._fitted_model  = None   # stores the last fitted ModelWrapper
        self._wf_results    = None   # stores walk-forward results once run

    # ================================================================
    # INTERNAL HELPER
    # ================================================================
    @property
    def _X_active(self):
        """Always returns X with only the currently active features."""
        return self.X[self.active_features]

    # ================================================================
    # INSPECT
    # ================================================================
    def info(self):
        """
        Prints a high-level summary of X and y:
        shape, active features, dtypes, and basic y stats.
        """
        print("=" * 55)
        print("  DATASET INFO")
        print("=" * 55)

        print(f"\n{'Observations:':<25} {len(self.X)}")
        print(f"{'Total features in X:':<25} {len(self.X.columns)}")
        print(f"{'Active features:':<25} {len(self.active_features)}")
        print(f"{'Min train size:':<25} {self.min_train}")
        print(f"{'Retrain every:':<25} {self.retrain_every}")

        # Walk-forward fold estimate
        remaining = len(self.X) - self.min_train
        n_folds   = max(0, int(np.ceil(remaining / self.retrain_every)))
        print(f"{'Estimated WF folds:':<25} {n_folds}")

        print(f"\n--- Active Features ---")
        for col in self.active_features:
            print(f"  {col:<35} {self.X[col].dtype}")

        inactive = [c for c in self.X.columns if c not in self.active_features]
        if inactive:
            print(f"\n--- Inactive Features (excluded from modelling) ---")
            for col in inactive:
                print(f"  {col}")

        print(f"\n--- Target (y) ---")
        print(f"  Name:      {self.y.name if hasattr(self.y, 'name') else 'y'}")
        print(f"  Length:    {len(self.y)}")
        print(f"  Dtype:     {self.y.dtype}")

    def missing(self):
        """
        Reports missing values in X (active features only) and y.
        Flags any columns with missing values.
        """
        print("=" * 55)
        print("  MISSING VALUES")
        print("=" * 55)

        # --- X ---
        X_active   = self._X_active
        na_counts  = X_active.isnull().sum()
        na_pct     = (na_counts / len(X_active) * 100).round(2)

        x_summary = pd.DataFrame({
            "Missing Count": na_counts,
            "Missing %":     na_pct
        })

        print(f"\n--- X (active features) ---")
        if na_counts.sum() == 0:
            print("  No missing values found.")
        else:
            print(x_summary[x_summary["Missing Count"] > 0].to_string())
            print(f"\n  Rows with any missing value: "
                  f"{X_active.isnull().any(axis=1).sum()}")

        # --- y ---
        y_missing = self.y.isnull().sum()
        print(f"\n--- y ---")
        if y_missing == 0:
            print("  No missing values found.")
        else:
            print(f"  Missing count: {y_missing} "
                  f"({y_missing / len(self.y) * 100:.2f}%)")

    def describe_y(self):
        """
        Descriptive statistics for y with count-model specific diagnostics:
        dispersion ratio, zero percentage, and a histogram.
        """
        print("=" * 55)
        print("  TARGET (y) DESCRIPTION")
        print("=" * 55)

        mean       = self.y.mean()
        var        = self.y.var()
        dispersion = var / mean

        print(f"\n{'Mean:':<25} {mean:.4f}")
        print(f"{'Variance:':<25} {var:.4f}")
        print(f"{'Std Dev:':<25} {self.y.std():.4f}")
        print(f"{'Min:':<25} {self.y.min()}")
        print(f"{'Max:':<25} {self.y.max()}")
        print(f"{'Skewness:':<25} {self.y.skew():.4f}")
        print(f"{'Zero count:':<25} {(self.y == 0).sum()} "
              f"({(self.y == 0).mean() * 100:.2f}%)")

        print(f"\n--- Dispersion Check ---")
        print(f"  Var / Mean = {dispersion:.4f}")

        # Histogram

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self.y, bins=range(int(self.y.min()), int(self.y.max()) + 2),
                edgecolor="white", color="steelblue", align="left")
        ax.axvline(mean,           color="red",    linestyle="--",
                   linewidth=1.5,  label=f"Mean ({mean:.1f})")
        ax.axvline(self.y.median(), color="orange", linestyle=":",
                   linewidth=1.5,  label=f"Median ({self.y.median():.1f})")
        ax.set_xlabel("y (count)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of y")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def describe_X(self):
        """
        Descriptive statistics for all active features.
        Numeric columns: standard stats + skew.
        Non-numeric columns: unique count + value frequencies.
        Flags any features with very low variance.
        """
        print("=" * 55)
        print("  FEATURE (X) DESCRIPTION")
        print("=" * 55)
    
        X_active     = self._X_active
        numeric_cols = X_active.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = X_active.select_dtypes(exclude=np.number).columns.tolist()
    
        # --- Numeric ---
        if numeric_cols:
            print("\n--- Numeric Features ---")
            stats = X_active[numeric_cols].describe().T
            stats["skew"] = X_active[numeric_cols].skew()
            print(stats.round(4).to_string())
    
            # Flag near-constant features
            low_var = [col for col in numeric_cols
                       if X_active[col].std() < 0.01]
            if low_var:
                print(f"\n  Warning — Low variance features (std < 0.01): {low_var}")
                print("  These may cause issues in GLM fitting.")
    
        # --- Categorical ---
        if cat_cols:
            print("\n--- Categorical Features ---")
            for col in cat_cols:
                n_unique = X_active[col].nunique()
                n_missing = X_active[col].isnull().sum()
                print(f"\n  {col}")
                print(f"    Unique values:  {n_unique}")
                print(f"    Missing:        {n_missing}")
                print(f"    Value counts:")
                counts = X_active[col].value_counts()
                for val, cnt in counts.items():
                    pct = cnt / len(X_active) * 100
                    print(f"      {str(val):<20} {cnt:>5}  ({pct:.1f}%)")
    
        if not numeric_cols and not cat_cols:
            print("\n  No active features found.")

    def describe_temporal(self):
        """
        Shows how observations are distributed over time using
        row position as a proxy for time (since X is ordered past → present).
        Useful for checking whether min_train is reasonable.
        """
        print("=" * 55)
        print("  TEMPORAL STRUCTURE")
        print("=" * 55)

        n          = len(self.X)
        train_end  = self.min_train
        remaining  = n - train_end
        n_folds    = max(0, int(np.ceil(remaining / self.retrain_every)))

        print(f"\n  Total observations:      {n}")
        print(f"  Min train window:        row 0  → row {train_end - 1} "
              f"({train_end} obs)")
        print(f"  Test window:             row {train_end} → row {n - 1} "
              f"({remaining} obs)")
        print(f"  Retrain every:           {self.retrain_every} obs")
        print(f"  Estimated folds:         {n_folds}")

        print(f"\n--- Walk-Forward Fold Breakdown ---")
        fold_start = train_end
        for i in range(n_folds):
            test_start = fold_start
            test_end   = min(fold_start + self.retrain_every - 1, n - 1)
            train_size = test_start  # expanding window: train on 0 → test_start-1
            print(f"  Fold {i+1:>2}: "
                  f"train rows 0–{test_start - 1:>4} ({train_size} obs) | "
                  f"test rows {test_start:>4}–{test_end:>4} "
                  f"({test_end - test_start + 1} obs)")
            fold_start += self.retrain_every

    # ================================================================
    # FEATURE MANAGEMENT
    # ================================================================
    
    def activate_features(self, features):
        """
        Add one or more features back into active_features.
        Accepts a single string or a list of strings.
        """
        if isinstance(features, str):
            features = [features]
    
        for feature in features:
            if feature not in self.X.columns:
                print(f"  Error: '{feature}' not found in X.")
            elif feature in self.active_features:
                print(f"  '{feature}' is already active.")
            else:
                self.active_features.append(feature)
                print(f"  '{feature}' activated.")
    
    def deactivate_features(self, features):
        """
        Remove one or more features from active_features.
        Accepts a single string or a list of strings.
        """
        if isinstance(features, str):
            features = [features]
    
        for feature in features:
            if feature not in self.active_features:
                print(f"  '{feature}' is not in active features.")
            else:
                self.active_features.remove(feature)
                print(f"  '{feature}' deactivated.")
    
    def set_active_features(self, features):
        """
        Explicitly set active_features to a given list.
        Validates that all supplied names exist in X.
        """
        invalid = [f for f in features if f not in self.X.columns]
        if invalid:
            print(f"  Error: these columns are not in X: {invalid}")
            return
        self.active_features = list(features)
        print(f"  Active features set to: {self.active_features}")
    
    def reset_features(self):
        """Restore active_features to all columns in X."""
        self.active_features = list(self.X.columns)
        print(f"  Active features reset to all {len(self.active_features)} columns.")
    
    def add_feature(self, name, values):
        """
        Add a new column to X and activate it immediately.
        values must be array-like with the same length as X.
        """
        if len(values) != len(self.X):
            print(f"  Error: length of values ({len(values)}) "
                  f"doesn't match X ({len(self.X)}).")
            return
        if name in self.X.columns:
            print(f"  Warning: '{name}' already exists in X and will be overwritten.")
        self.X[name] = values
        if name not in self.active_features:
            self.active_features.append(name)
        print(f"  '{name}' added to X and activated.")
    
    def transform_feature(self, name, func, new_name=None):
        """
        Apply a function to an existing column.
        If new_name is given, saves as a new column and activates it.
        If new_name is None, overwrites the existing column in place.
    
        Example:
            pipe.transform_feature("home_adj_shots", np.log, new_name="log_home_adj_shots")
            pipe.transform_feature("home_adj_shots", np.log)   # overwrites
        """
        if name not in self.X.columns:
            print(f"  Error: '{name}' not found in X.")
            return
        transformed = func(self.X[name])
        if new_name:
            self.X[new_name] = transformed
            if new_name not in self.active_features:
                self.active_features.append(new_name)
            print(f"  '{name}' transformed and saved as '{new_name}', activated.")
        else:
            self.X[name] = transformed
            print(f"  '{name}' transformed in place.")
    
    def add_interaction(self, name_a, name_b, new_name=None):
        """
        Creates a product interaction term between two numeric columns.
        If new_name is not supplied, defaults to 'name_a_x_name_b'.
    
        Example:
            pipe.add_interaction("prob_h", "home_adj_shots")
        """
        for name in [name_a, name_b]:
            if name not in self.X.columns:
                print(f"  Error: '{name}' not found in X.")
                return
        col_name = new_name if new_name else f"{name_a}_x_{name_b}"
        self.X[col_name] = self.X[name_a] * self.X[name_b]
        if col_name not in self.active_features:
            self.active_features.append(col_name)
        print(f"  Interaction '{col_name}' created and activated.")

    def set_feature_dtype(self, features, dtype):
        """
        Convert one or more features to a specified dtype.
        Accepts a single string or a list of strings.
        
        Common uses:
            pipe.set_feature_dtype("league", "category")
            pipe.set_feature_dtype(["season", "league"], "category")
            pipe.set_feature_dtype("season", "int")
        
        Parameters
        ----------
        features : str or list of str
        dtype    : str — any valid pandas dtype string
                   e.g. "category", "int", "float", "str", "bool"
        """
        if isinstance(features, str):
            features = [features]
    
        for feature in features:
            if feature not in self.X.columns:
                print(f"  Error: '{feature}' not found in X.")
                continue
            try:
                old_dtype = self.X[feature].dtype
                self.X[feature] = self.X[feature].astype(dtype)
                print(f"  '{feature}' converted from {old_dtype} → {dtype}.")
            except Exception as e:
                print(f"  Error converting '{feature}' to {dtype}: {e}")

    # ================================================================
    # DIAGNOSE
    # ================================================================
    
    def plot_distributions(self):
        """
        Plots histograms for numeric features and bar charts for
        categorical features.
        """
        X_active     = self._X_active
        numeric_cols = X_active.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = X_active.select_dtypes(exclude=np.number).columns.tolist()
        all_cols     = numeric_cols + cat_cols
    
        if not all_cols:
            print("  No active features to plot.")
            return
    
        n_cols = 3
        n_rows = int(np.ceil(len(all_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
    
        for i, col in enumerate(all_cols):
            ax = axes[i]
            if col in numeric_cols:
                ax.hist(X_active[col].dropna(), bins=30,
                        edgecolor="white", color="steelblue")
                ax.axvline(X_active[col].mean(),   color="red",
                           linestyle="--", linewidth=1.5, label="Mean")
                ax.axvline(X_active[col].median(), color="orange",
                           linestyle=":",  linewidth=1.5, label="Median")
                ax.legend(fontsize=8)
            else:
                counts = X_active[col].value_counts()
                ax.bar(counts.index.astype(str), counts.values, color="steelblue")
                ax.tick_params(axis="x", rotation=45)
    
            ax.set_title(col, fontsize=11)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
    
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
    
        fig.suptitle("Feature Distributions", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_correlations(self, threshold=0.7):
        """
        Plots a correlation heatmap for numeric active features.
        Prints pairs exceeding the threshold.
    
        Parameters
        ----------
        threshold : float — flag pairs with |r| above this value (default 0.7)
        """
        X_active     = self._X_active
        numeric_cols = X_active.select_dtypes(include=np.number).columns.tolist()
    
        if len(numeric_cols) < 2:
            print("  Need at least 2 numeric active features for correlation.")
            return
    
        corr = X_active[numeric_cols].corr()
    
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, square=True, ax=ax
        )
        ax.set_title("Correlation Heatmap (Active Numeric Features)", fontsize=13)
        plt.tight_layout()
        plt.show()
    
        # Flag high correlation pairs
        high_corr = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
        )
        high_corr.columns = ["Feature A", "Feature B", "Correlation"]
        high_corr = high_corr[high_corr["Correlation"].abs() > threshold]
    
        print(f"\n--- Pairs with |r| > {threshold} ---")
        if high_corr.empty:
            print(f"  None found.")
        else:
            print(high_corr.to_string(index=False))
    
    def vif(self, use_active_only=True):
        """
        Computes Variance Inflation Factors for numeric features.
        Flags features above VIF thresholds.
    
        Parameters
        ----------
        use_active_only : bool — if True, runs on active features only (default)
                                 if False, runs on all numeric columns in X
        """
        if use_active_only:
            X_eval = self._X_active.select_dtypes(include=np.number).copy()
            label  = "active features"
        else:
            X_eval = self.X.select_dtypes(include=np.number).copy()
            label  = "all features"
    
        # Drop near-constant columns — these break VIF
        X_eval = X_eval.loc[:, X_eval.std() > 0]
    
        if X_eval.shape[1] < 2:
            print("  Need at least 2 numeric features to compute VIF.")
            return
    
        vif_data = pd.DataFrame({
            "Feature": X_eval.columns,
            "VIF": [
                variance_inflation_factor(X_eval.values, i)
                for i in range(X_eval.shape[1])
            ]
        }).sort_values("VIF", ascending=False).reset_index(drop=True)
    
        print("=" * 55)
        print(f"  VIF SCORES ({label})")
        print("=" * 55)
        print(f"\n{vif_data.round(3).to_string(index=False)}")
        print(f"\n  Rule of thumb: VIF > 5 = moderate, VIF > 10 = serious")
    
        moderate = vif_data[vif_data["VIF"] > 5]
        serious  = vif_data[vif_data["VIF"] > 10]
    
        if serious.empty and moderate.empty:
            print("  → No multicollinearity concerns.")
        elif serious.empty:
            print(f"  → Moderate concern: {moderate['Feature'].tolist()}")
        else:
            print(f"  → Serious concern:  {serious['Feature'].tolist()}")
    
    def plot_feature_vs_y(self):
        """
        Scatter plots for numeric features vs y.
        Box plots for categorical features vs y.
        """
        X_active     = self._X_active
        numeric_cols = X_active.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = X_active.select_dtypes(exclude=np.number).columns.tolist()
        all_cols     = numeric_cols + cat_cols
    
        if not all_cols:
            print("  No active features to plot.")
            return
    
        n_cols = 3
        n_rows = int(np.ceil(len(all_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
    
        for i, col in enumerate(all_cols):
            ax = axes[i]
            if col in numeric_cols:
                ax.scatter(X_active[col], self.y, alpha=0.3,
                           color="steelblue", edgecolors="none", s=15)
                # Add a simple trend line
                m, b = np.polyfit(X_active[col].fillna(0), self.y, 1)
                x_line = np.linspace(X_active[col].min(), X_active[col].max(), 100)
                ax.plot(x_line, m * x_line + b, color="red",
                        linewidth=1.5, linestyle="--", label="Trend")
                ax.legend(fontsize=8)
            else:
                categories = X_active[col].dropna().unique()
                data_by_cat = [
                    self.y[X_active[col] == cat].dropna().values
                    for cat in categories
                ]
                ax.boxplot(data_by_cat, labels=[str(c) for c in categories])
                ax.tick_params(axis="x", rotation=45)
    
            ax.set_title(col, fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel("y")
    
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
    
        fig.suptitle("Features vs y", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def check_feature(self, feature):
        """
        Deep dive on a single feature. Combines:
          - Missing value count
          - Distribution plot
          - Temporal plot (value over row index)
          - Correlation with y
          - Ranked correlations with other active numeric features
          - VIF contribution
        
        Parameters
        ----------
        feature : str — name of the feature to inspect
        """
        if feature not in self.X.columns:
            print(f"  Error: '{feature}' not found in X.")
            return
    
        X_active     = self._X_active
        numeric_cols = X_active.select_dtypes(include=np.number).columns.tolist()
        is_numeric   = feature in numeric_cols
    
        print("=" * 55)
        print(f"  FEATURE CHECK — {feature}")
        print("=" * 55)
    
        col = self.X[feature]
    
        # --- Missing ---
        n_missing = col.isnull().sum()
        print(f"\n  Dtype:          {col.dtype}")
        print(f"  Missing:        {n_missing} ({n_missing / len(col) * 100:.2f}%)")
        print(f"  Active:         {feature in self.active_features}")
    
        if is_numeric:
            print(f"  Mean:           {col.mean():.4f}")
            print(f"  Std:            {col.std():.4f}")
            print(f"  Min / Max:      {col.min():.4f} / {col.max():.4f}")
            print(f"  Skew:           {col.skew():.4f}")
    
        fig, axes = plt.subplots(1, 3 if is_numeric else 2,
                                 figsize=(18 if is_numeric else 12, 4))
    
        # --- Distribution ---
        ax = axes[0]
        if is_numeric:
            ax.hist(col.dropna(), bins=30, edgecolor="white", color="steelblue")
            ax.axvline(col.mean(),   color="red",    linestyle="--",
                       linewidth=1.5, label="Mean")
            ax.axvline(col.median(), color="orange", linestyle=":",
                       linewidth=1.5, label="Median")
            ax.legend(fontsize=8)
        else:
            counts = col.value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="steelblue")
            ax.tick_params(axis="x", rotation=45)
        ax.set_title("Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
    
        # --- vs y ---
        ax = axes[1]
        if is_numeric:
            ax.scatter(col, self.y, alpha=0.3, color="steelblue",
                       edgecolors="none", s=15)
            m, b = np.polyfit(col.fillna(0), self.y, 1)
            x_line = np.linspace(col.min(), col.max(), 100)
            ax.plot(x_line, m * x_line + b, color="red",
                    linewidth=1.5, linestyle="--")
            corr_with_y = col.corr(self.y)
            ax.set_title(f"vs y  (r = {corr_with_y:.3f})")
        else:
            categories  = col.dropna().unique()
            data_by_cat = [self.y[col == cat].dropna().values for cat in categories]
            ax.boxplot(data_by_cat, labels=[str(c) for c in categories])
            ax.tick_params(axis="x", rotation=45)
            ax.set_title("vs y (by category)")
        ax.set_xlabel(feature)
        ax.set_ylabel("y")
    
        # --- Temporal plot (numeric only) ---
        if is_numeric:
            ax = axes[2]
            ax.plot(col.values, color="steelblue", linewidth=0.8, alpha=0.8)
            ax.set_title("Over Time (row index)")
            ax.set_xlabel("Row index")
            ax.set_ylabel(feature)
    
        plt.suptitle(f"Feature Check — {feature}", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.show()
    
        # --- Correlation with other active numeric features ---
        if is_numeric and len(numeric_cols) > 1:
            other_cols = [c for c in numeric_cols if c != feature]
            corrs = (
                X_active[other_cols]
                .corrwith(X_active[feature])
                .sort_values(key=abs, ascending=False)
            )
            print(f"\n--- Correlation with other active numeric features ---")
            for fname, r in corrs.items():
                flag = " ← high" if abs(r) > 0.7 else ""
                print(f"  {fname:<35} r = {r:>7.4f}{flag}")
    
        # --- VIF contribution ---
        if is_numeric:
            X_vif = X_active.select_dtypes(include=np.number).copy()
            X_vif = X_vif.loc[:, X_vif.std() > 0]
            if feature in X_vif.columns and X_vif.shape[1] >= 2:
                idx          = list(X_vif.columns).index(feature)
                vif_val      = variance_inflation_factor(X_vif.values, idx)
                vif_flag     = ("← serious" if vif_val > 10
                                else "← moderate" if vif_val > 5 else "")
                print(f"\n--- VIF ---")
                print(f"  {feature}: {vif_val:.3f} {vif_flag}")

    # ================================================================
    # FEATURE SELECTION
    # ================================================================
    
    def _check_model_fitted(self):
        """Internal helper — raises clear error if no model has been fitted yet."""
        if self._fitted_model is None:
            raise RuntimeError(
                "No model fitted yet. Call .fit(model) before running "
                "feature selection methods."
            )
    
    def partial_ftest(self, feature):
        """
        Tests whether removing a feature significantly worsens model fit
        using a likelihood ratio test (LRT) for GLMs, or F-test for OLS.
        
        Only valid for OLS, Poisson, and NegBin models.
        Raises a warning for other model types.
    
        Parameters
        ----------
        feature : str — feature to test for removal
        """
        self._check_model_fitted()
    
        model_type = type(self._fitted_model).__name__
    
        if model_type not in ("OLSWrapper", "PoissonWrapper", "NegBinWrapper"):
            print(f"  Warning: partial_ftest is not valid for {model_type}. "
                  f"Use permutation_importance instead.")
            return
    
        if feature not in self.active_features:
            print(f"  Error: '{feature}' is not in active features.")
            return
    
        from scipy.stats import chi2
    
        X_active  = self._X_active
        X_full    = sm.add_constant(X_active)
        X_reduced = sm.add_constant(X_active.drop(columns=[feature]))
    
        if model_type == "OLSWrapper":
            full    = sm.OLS(self.y, X_full).fit()
            reduced = sm.OLS(self.y, X_reduced).fit()
            # F-test for OLS
            f_stat  = ((reduced.ssr - full.ssr) / 1) / (full.ssr / full.df_resid)
            p_value = 1 - chi2.cdf(f_stat, df=1)
            stat_label = f"F-stat = {f_stat:.4f}"
        else:
            family  = (sm.families.Poisson() if model_type == "PoissonWrapper"
                       else sm.families.NegativeBinomial())
            full    = sm.GLM(self.y, X_full,    family=family).fit()
            reduced = sm.GLM(self.y, X_reduced, family=family).fit()
            # LRT for GLMs
            lrt     = 2 * (full.llf - reduced.llf)
            p_value = chi2.sf(lrt, df=1)
            stat_label = f"LRT = {lrt:.4f}"
    
        print("=" * 55)
        print(f"  PARTIAL TEST — {feature}")
        print("=" * 55)
        print(f"\n  Model type:  {model_type}")
        print(f"  Test stat:   {stat_label}")
        print(f"  p-value:     {p_value:.4f}")
    
        if p_value < 0.05:
            print(f"  → '{feature}' is significant (p < 0.05). Keep it.")
        else:
            print(f"  → '{feature}' is NOT significant (p >= 0.05). "
                  f"Consider removing it.")
    
    def aic_bic_comparison(self):
        """
        Fits the current model with and without each active feature,
        recording AIC and BIC each time.
        Returns a DataFrame ranked by AIC.
    
        Only valid for OLS, Poisson, and NegBin models.
        """
        self._check_model_fitted()
    
        model_type = type(self._fitted_model).__name__
    
        if model_type not in ("OLSWrapper", "PoissonWrapper", "NegBinWrapper"):
            print(f"  Warning: aic_bic_comparison is not valid for {model_type}. "
                  f"Use permutation_importance instead.")
            return
    
        X_active = self._X_active
        X_full   = sm.add_constant(X_active)
    
        # Fit full model first
        if model_type == "OLSWrapper":
            full_fit  = sm.OLS(self.y, X_full).fit()
            def _fit(X): return sm.OLS(self.y, X).fit()
        else:
            family = (sm.families.Poisson() if model_type == "PoissonWrapper"
                      else sm.families.NegativeBinomial())
            full_fit  = sm.GLM(self.y, X_full, family=family).fit()
            def _fit(X): return sm.GLM(self.y, X, family=family).fit()
    
        print("=" * 55)
        print(f"  AIC / BIC COMPARISON — {model_type}")
        print("=" * 55)
        print(f"\n  Full model — AIC: {full_fit.aic:.2f}  BIC: {full_fit.bic:.2f}")
        print(f"\n  Impact of removing each feature:")
    
        results = []
        for feature in self.active_features:
            X_reduced = sm.add_constant(
                X_active.drop(columns=[feature]),
                has_constant="add"
            )
            reduced_fit = _fit(X_reduced)
            results.append({
                "Dropped Feature": feature,
                "AIC":             round(reduced_fit.aic, 2),
                "BIC":             round(reduced_fit.bic, 2),
                "ΔAIC":            round(reduced_fit.aic - full_fit.aic, 2),
                "ΔBIC":            round(reduced_fit.bic - full_fit.bic, 2),
            })
    
        results_df = (pd.DataFrame(results)
                        .sort_values("ΔAIC")
                        .reset_index(drop=True))
    
        print(f"\n{results_df.to_string(index=False)}")
        print(f"\n  Interpretation:")
        print(f"  ΔAIC > 0 → dropping this feature worsens fit (keep it)")
        print(f"  ΔAIC < 0 → dropping this feature improves fit (consider removing)")
    
        return results_df
    
    def lasso_selection(self, cv=5, scale=True):
        """
        Fits a LASSO model to identify which features survive regularisation.
        
        - OLSWrapper:     uses sklearn LassoCV (proper L1 OLS)
        - PoissonWrapper: uses sklearn PoissonRegressor with L1 penalty
        - NegBinWrapper:  Warning — no native LASSO support. 
                          Falls back to OLS LASSO as a guide.
        - Others:         Warning raised, method exits.
    
        Parameters
        ----------
        cv    : int   — number of cross-validation folds for LassoCV (default 5)
        scale : bool  — whether to standardise X before fitting (default True,
                        required for LASSO to be meaningful)
        """
        self._check_model_fitted()
    
        model_type = type(self._fitted_model).__name__
    
        if model_type not in ("OLSWrapper", "PoissonWrapper", "NegBinWrapper"):
            print(f"  Warning: lasso_selection is not supported for {model_type}.")
            return
    
        if model_type == "NegBinWrapper":
            print(f"  Warning: LASSO is not natively available for Negative Binomial.")
            print(f"  Falling back to OLS LASSO as a feature guide only.")
            print(f"  Treat surviving features as indicative, not definitive.\n")
    
        # Prepare X — numeric active features only
        X_active     = self._X_active
        numeric_cols = X_active.select_dtypes(include=np.number).columns.tolist()
        X_num        = X_active[numeric_cols].fillna(0).values
        y_vals       = self.y.values
    
        if scale:
            scaler = StandardScaler()
            X_num  = scaler.fit_transform(X_num)
    
        print("=" * 55)
        print(f"  LASSO FEATURE SELECTION — {model_type}")
        print("=" * 55)
        print(f"  Scaled: {scale}  |  CV folds: {cv}")
    
        if model_type == "OLSWrapper" or model_type == "NegBinWrapper":
            # LassoCV selects optimal alpha via cross-validation
            lasso = LassoCV(cv=cv, max_iter=10000).fit(X_num, y_vals)
            coefs = lasso.coef_
            alpha = lasso.alpha_
            print(f"  Optimal alpha (λ): {alpha:.6f}")
    
        elif model_type == "PoissonWrapper":
            # Try a range of alphas and pick best by cross-validated deviance
            alphas     = np.logspace(-4, 1, 30)
            kf         = KFold(n_splits=cv, shuffle=False)
            best_alpha = None
            best_score = -np.inf
    
            for alpha in alphas:
                fold_scores = []
                for train_idx, val_idx in kf.split(X_num):
                    m = PoissonRegressor(alpha=alpha, max_iter=1000)
                    m.fit(X_num[train_idx], y_vals[train_idx])
                    fold_scores.append(m.score(X_num[val_idx], y_vals[val_idx]))
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
    
            lasso = PoissonRegressor(alpha=best_alpha, max_iter=1000)
            lasso.fit(X_num, y_vals)
            coefs = lasso.coef_
            print(f"  Optimal alpha (λ): {best_alpha:.6f}")
    
        # Build results table
        results = pd.DataFrame({
            "Feature":   numeric_cols,
            "Coefficient": coefs.round(6)
        }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)
    
        results["Survived"] = results["Coefficient"].abs() > 0
    
        print(f"\n{results.to_string(index=False)}")
    
        survived = results[results["Survived"]]["Feature"].tolist()
        zeroed   = results[~results["Survived"]]["Feature"].tolist()
    
        print(f"\n  Survived regularisation ({len(survived)}): {survived}")
        if zeroed:
            print(f"  Zeroed out ({len(zeroed)}):               {zeroed}")
            print(f"\n  Tip: consider calling .deactivate_features({zeroed}) "
                  f"and re-evaluating.")
    
        return results
    
    def permutation_importance(self, n_repeats=10, metric="mae", random_state=42):
        """
        Model-agnostic feature importance. Shuffles each active numeric
        feature n_repeats times and measures the degradation in the
        chosen metric. Works for all model types.
    
        Parameters
        ----------
        n_repeats    : int — how many times to shuffle each feature (default 10)
        metric       : str — "mae" or "rmse" (default "mae")
        random_state : int — for reproducibility
        """
        self._check_model_fitted()
    
        rng      = np.random.default_rng(random_state)
        X_active = self._X_active
        num_cols = X_active.select_dtypes(include=np.number).columns.tolist()
    
        if not num_cols:
            print("  No numeric active features found.")
            return
    
        # Baseline score on unshuffled data
        X_base    = sm.add_constant(X_active, has_constant="add")
        y_pred    = self._fitted_model.predict(X_base)
    
        def _score(y_true, y_pred):
            if metric == "mae":
                return np.mean(np.abs(y_true - y_pred))
            elif metric == "rmse":
                return np.sqrt(np.mean((y_true - y_pred) ** 2))
            else:
                raise ValueError(f"Unknown metric '{metric}'. Use 'mae' or 'rmse'.")
    
        baseline = _score(self.y.values, y_pred)
    
        print("=" * 55)
        print(f"  PERMUTATION IMPORTANCE ({metric.upper()})")
        print("=" * 55)
        print(f"  Baseline {metric.upper()}: {baseline:.4f}")
        print(f"  Repeats per feature: {n_repeats}\n")
    
        results = []
        for col in num_cols:
            deltas = []
            for _ in range(n_repeats):
                X_permuted       = X_active.copy()
                X_permuted[col]  = rng.permutation(X_permuted[col].values)
                X_perm_const     = sm.add_constant(X_permuted, has_constant="add")
                y_perm_pred      = self._fitted_model.predict(X_perm_const)
                permuted_score   = _score(self.y.values, y_perm_pred)
                deltas.append(permuted_score - baseline)
    
            results.append({
                "Feature":       col,
                f"Mean Δ{metric.upper()}": round(np.mean(deltas), 4),
                "Std":           round(np.std(deltas), 4),
            })
    
        results_df = (pd.DataFrame(results)
                        .sort_values(f"Mean Δ{metric.upper()}", ascending=False)
                        .reset_index(drop=True))
    
        print(results_df.to_string(index=False))
        print(f"\n  Interpretation:")
        print(f"  Higher Δ{metric.upper()} → shuffling this feature hurts more "
              f"→ more important")
        print(f"  Near 0 or negative → feature adds little predictive value")
    
        return results_df

        # ================================================================
    # MODELLING
    # ================================================================
    
    def fit(self, model, add_constant=True):
        """
        Fits a ModelWrapper on the full active feature set.
        Stores the fitted model for use in summary, evaluation,
        and feature selection methods.
    
        Parameters
        ----------
        model        : ModelWrapper subclass instance
        add_constant : bool — whether to add intercept (default True)
        """
        X = self._X_active.select_dtypes(include=np.number)
        if add_constant:
            X = sm.add_constant(X, has_constant="add")
    
        model.fit(X, self.y)
        self._fitted_model      = model
        self._fitted_add_const  = add_constant   # store for reuse in other methods
        print(f"  {model.name} fitted on {X.shape[0]} observations, "
              f"{X.shape[1]} features "
              f"({'with' if add_constant else 'without'} constant).")
    
    def summary(self):
        """
        Prints the summary of the currently fitted model.
        """
        self._check_model_fitted()
        print(self._fitted_model.get_summary())
    
    def walk_forward(self, model, add_constant=True, metric="mae"):
        """
        Runs a walk-forward expanding window backtest for a given model.
        Stores results in self._wf_results keyed by model name so that
        compare_models can reuse them without refitting.
    
        Parameters
        ----------
        model        : ModelWrapper subclass instance
        add_constant : bool   — whether to add intercept (default True)
        metric       : str    — "mae" or "rmse" (default "mae")
    
        Stores
        ------
        self._wf_results : dict keyed by model.name, each value is a dict:
            {
                "fold_df"  : DataFrame with per-fold results,
                "all_preds": Series of all out-of-sample predictions,
                "all_actuals": Series of all out-of-sample actuals,
                "metric"   : str,
                "mean_score": float
            }
        """
        if self._wf_results is None:
            self._wf_results = {}
    
        X_num = self._X_active.select_dtypes(include=np.number)
        n     = len(X_num)
    
        fold_records = []
        all_preds    = []
        all_actuals  = []
        all_indices  = []
    
        fold_num   = 0
        test_start = self.min_train
    
        print(f"  Running walk-forward: {model.name}")
        print(f"  {'Fold':<6} {'Train size':<12} {'Test size':<12} "
              f"{'Test rows':<20} {metric.upper()}")
        print(f"  {'-'*60}")
    
        while test_start < n:
            test_end = min(test_start + self.retrain_every, n)
            fold_num += 1
    
            # --- Split ---
            X_train = X_num.iloc[:test_start]
            y_train = self.y.iloc[:test_start]
            X_test  = X_num.iloc[test_start:test_end]
            y_test  = self.y.iloc[test_start:test_end]
    
            # --- Add constant ---
            if add_constant:
                X_train = sm.add_constant(X_train, has_constant="add")
                X_test  = sm.add_constant(X_test,  has_constant="add")
    
            # --- Fit and predict ---
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
    
            # --- Score ---
            if metric == "mae":
                score = np.mean(np.abs(y_test.values - y_pred))
            elif metric == "rmse":
                score = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    
            fold_records.append({
                "fold":       fold_num,
                "train_size": test_start,
                "test_size":  test_end - test_start,
                "test_start": test_start,
                "test_end":   test_end - 1,
                metric:       round(score, 4)
            })
    
            all_preds.extend(y_pred)
            all_actuals.extend(y_test.values)
            all_indices.extend(y_test.index.tolist())
    
            print(f"  {fold_num:<6} {test_start:<12} {test_end - test_start:<12} "
                  f"{str(test_start) + '-' + str(test_end-1):<20} {score:.4f}")
    
            test_start += self.retrain_every
    
        fold_df     = pd.DataFrame(fold_records)
        mean_score  = fold_df[metric].mean()
        all_preds   = pd.Series(all_preds,   index=all_indices, name="predicted")
        all_actuals = pd.Series(all_actuals, index=all_indices, name="actual")
    
        # --- Store results keyed by model name ---
        self._wf_results[model.name] = {
            "fold_df":     fold_df,
            "all_preds":   all_preds,
            "all_actuals": all_actuals,
            "metric":      metric,
            "mean_score":  mean_score
        }
    
        print(f"\n  Mean {metric.upper()} across folds: {mean_score:.4f}")
        print(f"  Results stored under key: '{model.name}'")
    
        return fold_df
    
    def compare_models(self, models, add_constant=True, metric="mae"):
        """
        Compares multiple models using walk-forward backtesting.
        Reuses stored results if a model has already been evaluated,
        otherwise runs walk_forward automatically.
    
        Parameters
        ----------
        models       : list of ModelWrapper instances
        add_constant : bool — whether to add intercept (default True)
        metric       : str  — "mae" or "rmse" (default "mae")
    
        Returns
        -------
        pd.DataFrame — comparison table ranked by mean metric score
        """
        if self._wf_results is None:
            self._wf_results = {}
    
        comparison = []
    
        for model in models:
            # Reuse stored results if available, otherwise run walk_forward
            if model.name in self._wf_results:
                print(f"  Reusing stored results for {model.name}.")
                result = self._wf_results[model.name]
            else:
                print(f"  No stored results for {model.name} — running walk_forward.")
                self.walk_forward(model, add_constant=add_constant, metric=metric)
                result = self._wf_results[model.name]
    
            comparison.append({
                "Model":              model.name,
                f"Mean {metric.upper()}": round(result["mean_score"], 4),
                f"Best fold {metric.upper()}":  round(result["fold_df"][metric].min(), 4),
                f"Worst fold {metric.upper()}": round(result["fold_df"][metric].max(), 4),
                "Folds":              len(result["fold_df"])
            })
    
        comparison_df = (pd.DataFrame(comparison)
                           .sort_values(f"Mean {metric.upper()}")
                           .reset_index(drop=True))
    
        print("\n" + "=" * 55)
        print("  MODEL COMPARISON")
        print("=" * 55)
        print(f"\n{comparison_df.to_string(index=False)}")
        print(f"\n  Ranked by mean {metric.upper()} (lower is better).")
    
        return comparison_df

    # ================================================================
    # EVALUATE
    # ================================================================
    
    def _get_wf_results(self, model):
        """
        Internal helper — retrieves stored walk-forward results for a model.
        Raises a clear error if walk_forward hasn't been run yet.
        """
        if self._wf_results is None or model.name not in self._wf_results:
            raise RuntimeError(
                f"No walk-forward results found for {model.name}. "
                f"Call .walk_forward(model) first."
            )
        return self._wf_results[model.name]
    
    def plot_walk_forward_mae(self, model):
        """
        Plots fold-level MAE over walk-forward folds for a single model.
    
        Parameters
        ----------
        model : ModelWrapper — must have walk_forward results stored
        """
        result  = self._get_wf_results(model)
        fold_df = result["fold_df"]
        metric  = result["metric"]
        mean_score = result["mean_score"]
    
        fig, ax = plt.subplots(figsize=(10, 4))
    
        ax.plot(fold_df["fold"], fold_df[metric],
                marker="o", linewidth=1.5, color="steelblue", label=model.name)
        ax.axhline(mean_score, color="red", linestyle="--",
                   linewidth=1.2, label=f"Mean {metric.upper()} ({mean_score:.4f})")
    
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Walk-Forward {metric.upper()} — {model.name}")
        ax.legend()
        plt.tight_layout()
        plt.show()
    
        print(f"  Mean {metric.upper()}: {mean_score:.4f}")
        print(f"  Best fold:  {fold_df[metric].min():.4f} "
              f"(fold {fold_df.loc[fold_df[metric].idxmin(), 'fold']})")
        print(f"  Worst fold: {fold_df[metric].max():.4f} "
              f"(fold {fold_df.loc[fold_df[metric].idxmax(), 'fold']})")
    
    def residual_plot(self, model):
        """
        Plots out-of-sample residuals (actual - predicted) against
        predicted values. Uses stored walk-forward predictions.
    
        Parameters
        ----------
        model : ModelWrapper — must have walk_forward results stored
        """
        result   = self._get_wf_results(model)
        preds    = result["all_preds"].values
        actuals  = result["all_actuals"].values
        residuals = actuals - preds
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # --- Residuals vs fitted ---
        axes[0].scatter(preds, residuals, alpha=0.3, color="steelblue",
                        edgecolors="none", s=15)
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual (actual - predicted)")
        axes[0].set_title(f"Residuals vs Fitted — {model.name}")
    
        # --- Residual distribution ---
        axes[1].hist(residuals, bins=30, edgecolor="white", color="steelblue")
        axes[1].axvline(0,                color="red",    linestyle="--",
                        linewidth=1.2,    label="Zero")
        axes[1].axvline(residuals.mean(), color="orange", linestyle=":",
                        linewidth=1.2,
                        label=f"Mean ({residuals.mean():.2f})")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Residual Distribution — {model.name}")
        axes[1].legend()
    
        plt.tight_layout()
        plt.show()
    
        print(f"  Mean residual:   {residuals.mean():.4f}  "
              f"(close to 0 = unbiased)")
        print(f"  Std residual:    {residuals.std():.4f}")
        print(f"  Max over-pred:   {residuals.min():.4f}")
        print(f"  Max under-pred:  {residuals.max():.4f}")
    
    def dispersion_check(self, model):
        """
        Computes Pearson chi² / df on out-of-sample predictions.
        Indicates whether Poisson's mean=variance assumption holds.
        Only meaningful for Poisson — raises a warning for other models.
    
        Parameters
        ----------
        model : ModelWrapper — must have walk_forward results stored
        """
        if type(model).__name__ not in ("PoissonWrapper", "OLSWrapper"):
            print(f"  Note: dispersion_check is most meaningful for "
                  f"PoissonWrapper. Interpreting with caution for "
                  f"{type(model).__name__}.")
    
        result  = self._get_wf_results(model)
        preds   = result["all_preds"].values
        actuals = result["all_actuals"].values
    
        # Pearson chi² = sum((y - mu)^2 / mu)
        pearson_chi2 = np.sum((actuals - preds) ** 2 / np.clip(preds, 1e-8, None))
        df           = len(actuals) - self._X_active.shape[1] - 1
        dispersion   = pearson_chi2 / df
    
        print("=" * 55)
        print(f"  DISPERSION CHECK — {model.name}")
        print("=" * 55)
        print(f"\n  Pearson chi²:       {pearson_chi2:.4f}")
        print(f"  Degrees of freedom: {df}")
        print(f"  Dispersion ratio:   {dispersion:.4f}")
    
        if dispersion < 1.2:
            print("  → Near-equidispersed. Poisson assumption holds reasonably.")
        elif dispersion < 2.0:
            print("  → Moderate overdispersion. NegBin worth fitting.")
        else:
            print("  → Strong overdispersion. NegBin strongly recommended.")
    
    def plot_predicted_vs_actual(self, model):
        """
        Scatter plot of out-of-sample predicted vs actual values.
        Includes a 45-degree perfect prediction line.
    
        Parameters
        ----------
        model : ModelWrapper — must have walk_forward results stored
        """
        result  = self._get_wf_results(model)
        preds   = result["all_preds"].values
        actuals = result["all_actuals"].values
    
        min_val = min(preds.min(), actuals.min())
        max_val = max(preds.max(), actuals.max())
    
        fig, ax = plt.subplots(figsize=(7, 7))
    
        ax.scatter(actuals, preds, alpha=0.3, color="steelblue",
                   edgecolors="none", s=15)
        ax.plot([min_val, max_val], [min_val, max_val],
                color="red", linestyle="--", linewidth=1.5, label="Perfect prediction")
    
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predicted vs Actual — {model.name}")
        ax.legend()
        plt.tight_layout()
        plt.show()
    
        corr = np.corrcoef(actuals, preds)[0, 1]
        mae  = np.mean(np.abs(actuals - preds))
        print(f"  Correlation (actual vs predicted): {corr:.4f}")
        print(f"  MAE:                               {mae:.4f}")
    
    def distribution_check(self, model, max_k=60):
        """
        Overlays the empirical distribution of out-of-sample actuals
        against the average predicted PMF across all test observations.
        Shows whether the model's distributional shape matches reality.
    
        Parameters
        ----------
        model : ModelWrapper — must have walk_forward results stored
        max_k : int          — upper bound of shot counts to evaluate (default 60)
        """
        result   = self._get_wf_results(model)
        preds    = result["all_preds"].values
        actuals  = result["all_actuals"].values
    
        # --- Empirical distribution ---
        k_vals       = np.arange(0, max_k + 1)
        actual_freq  = np.array([(actuals == k).mean() for k in k_vals])
    
        # --- Average predicted PMF ---
        # Rebuild X for test observations using stored indices
        test_idx  = result["all_preds"].index
        X_test    = self._X_active.select_dtypes(include=np.number).iloc[test_idx]
        if self._fitted_add_const:
            X_test = sm.add_constant(X_test, has_constant="add")
    
        pmfs         = model.predict_dist(X_test, max_k=max_k)
        avg_pmf      = pmfs.mean(axis=0)
    
        fig, ax = plt.subplots(figsize=(12, 5))
        width = 0.4
        ax.bar(k_vals - width/2, actual_freq, width=width,
               label="Actual",  color="steelblue", alpha=0.8)
        ax.bar(k_vals + width/2, avg_pmf,     width=width,
               label="Predicted PMF", color="orange", alpha=0.8)
    
        ax.set_xlabel("Shot count (k)")
        ax.set_ylabel("Probability")
        ax.set_title(f"Empirical vs Predicted Distribution — {model.name}")
        ax.legend()
        plt.tight_layout()
        plt.show()
    
        # Highlight where model over/underestimates
        diff = avg_pmf - actual_freq
        over  = k_vals[diff >  0.01]
        under = k_vals[diff < -0.01]
        if len(over):
            print(f"  Model overestimates probability at k = {over.tolist()}")
        if len(under):
            print(f"  Model underestimates probability at k = {under.tolist()}")
    
    def temporal_stability(self, model, add_constant=True):
        """
        Refits the model on each walk-forward training window and records
        coefficients at each fold. Plots coefficient trajectories over time
        to check for instability or drift.
    
        Only meaningful for GLM-based models with interpretable coefficients.
        Raises a warning for XGBoost.
    
        Parameters
        ----------
        model        : ModelWrapper — any fitted wrapper
        add_constant : bool         — should match what was used in walk_forward
        """
        model_type = type(model).__name__
    
        if model_type == "XGBoostWrapper":
            print(f"  Warning: temporal_stability is not meaningful for "
                  f"XGBoostWrapper — coefficients are not directly interpretable.")
            return
    
        if model_type == "NegBinWrapper":
            print(f"  Warning: temporal_stability refits NegBin on every fold. "
                  f"This may be slow for large datasets.")
    
        self._check_model_fitted()
    
        X_num = self._X_active.select_dtypes(include=np.number)
        n     = len(X_num)
    
        coef_records = []
        test_start   = self.min_train
    
        while test_start < n:
            X_train = X_num.iloc[:test_start]
            y_train = self.y.iloc[:test_start]
    
            if add_constant:
                X_train = sm.add_constant(X_train, has_constant="add")
    
            model.fit(X_train, y_train)
    
            # Extract coefficients from statsmodels fitted model
            coefs = model._model.params
            coefs["fold_train_size"] = test_start
            coef_records.append(coefs)
    
            test_start += self.retrain_every
    
        coef_df = pd.DataFrame(coef_records).set_index("fold_train_size")
    
        # Plot each coefficient's trajectory
        feature_cols = [c for c in coef_df.columns]
        n_cols       = 3
        n_rows       = int(np.ceil(len(feature_cols) / n_cols))
    
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
    
        for i, col in enumerate(feature_cols):
            axes[i].plot(coef_df.index, coef_df[col],
                         marker="o", linewidth=1.5, color="steelblue")
            axes[i].axhline(0, color="red", linestyle="--",
                            linewidth=1, alpha=0.5)
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("Train size (rows)")
            axes[i].set_ylabel("Coefficient")
    
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
    
        fig.suptitle(f"Coefficient Stability — {model.name}", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.show()
    
        print(f"  Coefficient ranges across folds:")
        print(coef_df.describe().round(4).to_string())
    
        return coef_df
        
    # ================================================================
    # DISTRIBUTE
    # ================================================================
    
    def predict_mean(self, X_input, add_constant=None):
        """
        Returns predicted mean shot count for one or more observations.
    
        Parameters
        ----------
        X_input     : pd.DataFrame — must contain the active numeric features
        add_constant: bool or None — if None, uses setting from fit()
    
        Returns
        -------
        np.array of predicted means
        """
        self._check_model_fitted()
    
        add_const = self._fitted_add_const if add_constant is None else add_constant
    
        X = X_input[self.active_features].select_dtypes(include=np.number)
        if add_const:
            X = sm.add_constant(X, has_constant="add")
    
        return self._fitted_model.predict(X)
    
    
    def predict_pmf(self, X_input, max_k=60, add_constant=None):
        """
        Returns full PMF over k = 0, 1, ..., max_k for one or more observations.
    
        Parameters
        ----------
        X_input     : pd.DataFrame — must contain the active numeric features
        max_k       : int          — upper bound of shot counts (default 60)
        add_constant: bool or None — if None, uses setting from fit()
    
        Returns
        -------
        np.array of shape (n_rows, max_k + 1)
        where entry [i, k] = P(Y = k | X_i)
        """
        self._check_model_fitted()
    
        add_const = self._fitted_add_const if add_constant is None else add_constant
    
        X = X_input[self.active_features].select_dtypes(include=np.number)
        if add_const:
            X = sm.add_constant(X, has_constant="add")
    
        return self._fitted_model.predict_dist(X, max_k=max_k)
    
    
    def plot_pmf(self, X_input, max_k=60, add_constant=None,
                 actual=None, label=None):
        """
        Plots the predicted PMF for a single observation.
    
        Parameters
        ----------
        X_input     : pd.DataFrame — single row feature input
        max_k       : int          — upper bound of shot counts (default 60)
        add_constant: bool or None — if None, uses setting from fit()
        actual      : int or None  — if provided, marks the actual value on plot
        label       : str or None  — optional title label (e.g. "Arsenal vs Chelsea")
        """
        self._check_model_fitted()
    
        if len(X_input) > 1:
            print("  Warning: plot_pmf expects a single row. "
                  "Using first row only.")
            X_input = X_input.iloc[[0]]
    
        pmf    = self.predict_pmf(X_input, max_k=max_k,
                                   add_constant=add_constant)[0]
        k_vals = np.arange(0, max_k + 1)
        mean   = self.predict_mean(X_input, add_constant=add_constant)[0]
    
        fig, ax = plt.subplots(figsize=(12, 5))
    
        ax.bar(k_vals, pmf, color="steelblue", alpha=0.8, label="Predicted PMF")
        ax.axvline(mean, color="red", linestyle="--",
                   linewidth=1.5, label=f"Predicted mean ({mean:.1f})")
    
        if actual is not None:
            ax.axvline(actual, color="green", linestyle="-",
                       linewidth=2, label=f"Actual ({actual})")
    
        # Shade 80% credible interval
        cumulative = np.cumsum(pmf)
        lower = k_vals[cumulative >= 0.10][0]
        upper = k_vals[cumulative >= 0.90][0]
        ax.axvspan(lower, upper, alpha=0.1, color="steelblue",
                   label=f"80% interval ({lower}–{upper})")
    
        title = f"Predicted PMF — {self._fitted_model.name}"
        if label:
            title += f" | {label}"
        ax.set_xlabel("Shot count (k)")
        ax.set_ylabel("P(Y = k)")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
    
        print(f"  Predicted mean:      {mean:.2f}")
        print(f"  80% interval:        {lower} – {upper}")
        print(f"  P(Y < 10):           {pmf[:10].sum():.4f}")
        print(f"  P(Y > 25):           {pmf[26:].sum():.4f}")
    
    
    def predict_match(self, X_home, X_away, max_k=60,
                      add_constant=None, label=None):
        """
        Generates PMFs for home shots, away shots, and total shots
        for a single match by convolving the two team PMFs.
    
        Parameters
        ----------
        X_home      : pd.DataFrame — single row, home team features
        X_away      : pd.DataFrame — single row, away team features
        max_k       : int          — upper bound per team (default 60)
        add_constant: bool or None — if None, uses setting from fit()
        label       : str or None  — optional match label for plot title
    
        Returns
        -------
        dict with keys: pmf_home, pmf_away, pmf_total, mean_home,
                        mean_away, mean_total, k_vals_total
        """
        self._check_model_fitted()
    
        pmf_home = self.predict_pmf(X_home, max_k=max_k,
                                     add_constant=add_constant)[0]
        pmf_away = self.predict_pmf(X_away, max_k=max_k,
                                     add_constant=add_constant)[0]
    
        pmf_total = self._convolve_pmfs(pmf_home, pmf_away)
    
        mean_home  = float(np.sum(np.arange(len(pmf_home))  * pmf_home))
        mean_away  = float(np.sum(np.arange(len(pmf_away))  * pmf_away))
        mean_total = float(np.sum(np.arange(len(pmf_total)) * pmf_total))
    
        k_vals_total = np.arange(len(pmf_total))
    
        # --- Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        for ax, pmf, mean, title_suffix in zip(
            axes,
            [pmf_home, pmf_away, pmf_total],
            [mean_home, mean_away, mean_total],
            ["Home Shots", "Away Shots", "Total Shots"]
        ):
            k = np.arange(len(pmf))
            ax.bar(k, pmf, color="steelblue", alpha=0.8)
            ax.axvline(mean, color="red", linestyle="--",
                       linewidth=1.5, label=f"Mean ({mean:.1f})")
    
            # 80% interval
            cumulative = np.cumsum(pmf)
            lower = k[cumulative >= 0.10][0]
            upper = k[cumulative >= 0.90][0]
            ax.axvspan(lower, upper, alpha=0.1, color="steelblue",
                       label=f"80% ({lower}–{upper})")
    
            ax.set_xlabel("Shot count (k)")
            ax.set_ylabel("Probability")
            ax.set_title(title_suffix)
            ax.legend(fontsize=8)
    
        match_label = label if label else "Match Prediction"
        fig.suptitle(f"{match_label} — {self._fitted_model.name}",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        plt.show()
    
        print(f"  {'':20} {'Mean':>8}  {'80% interval':>15}")
        print(f"  {'Home shots':<20} {mean_home:>8.2f}")
        print(f"  {'Away shots':<20} {mean_away:>8.2f}")
        print(f"  {'Total shots':<20} {mean_total:>8.2f}")
    
        return {
            "pmf_home":    pmf_home,
            "pmf_away":    pmf_away,
            "pmf_total":   pmf_total,
            "mean_home":   mean_home,
            "mean_away":   mean_away,
            "mean_total":  mean_total,
            "k_vals_total": k_vals_total
        }
    
    
    def _convolve_pmfs(self, pmf_a, pmf_b):
        """
        Convolves two PMFs to produce the distribution of their sum.
        P(A + B = t) = sum_k P(A=k) * P(B=t-k)
    
        Parameters
        ----------
        pmf_a : np.array — PMF of first variable
        pmf_b : np.array — PMF of second variable
    
        Returns
        -------
        np.array — PMF of the sum
        """
        max_t  = len(pmf_a) + len(pmf_b) - 2
        result = np.zeros(max_t + 1)
        for k in range(len(pmf_a)):
            result[k: k + len(pmf_b)] += pmf_a[k] * pmf_b
        return result


    # ================================================================
    # PREDICT
    # ================================================================
    
    def set_future_data(self, X_future, labels=None):
        """
        Stores future data for use by all PREDICT methods.
        Validates that X_future contains all active features.
        Extra columns beyond active features are ignored.
    
        Parameters
        ----------
        X_future : pd.DataFrame — one row per future observation
        labels   : list of str or None — optional labels per row
                   e.g. ["Match 1", "Match 2"]
        """
        # Validate active features are present
        missing_cols = [f for f in self.active_features
                        if f not in X_future.columns]
        if missing_cols:
            raise ValueError(
                f"X_future is missing the following active features: "
                f"{missing_cols}"
            )
    
        extra_cols = [c for c in X_future.columns
                      if c not in self.active_features]
        if extra_cols:
            print(f"  Note: {len(extra_cols)} extra column(s) in X_future "
                  f"will be ignored: {extra_cols}")
    
        if labels is not None and len(labels) != len(X_future):
            raise ValueError(
                f"labels length ({len(labels)}) must match "
                f"number of rows in X_future ({len(X_future)})."
            )
    
        self._X_future = X_future.reset_index(drop=True)
        self._labels   = labels if labels is not None else \
                         [f"Observation {i+1}" for i in range(len(X_future))]
    
        print(f"  Future data set: {len(X_future)} observation(s), "
              f"{len(self.active_features)} active feature(s) used.")
        if extra_cols:
            print(f"  Ignored columns: {extra_cols}")
    
    def _check_future_data(self):
        """Internal helper — raises clear error if future data not set."""
        if not hasattr(self, "_X_future") or self._X_future is None:
            raise RuntimeError(
                "No future data set. Call .set_future_data(X_future) first."
            )
    
    def _prepare_future_X(self, add_constant):
        """
        Internal helper — returns the future X matrix with only
        active numeric features, with optional constant added.
        """
        add_const = self._fitted_add_const if add_constant is None else add_constant
        X = self._X_future[self.active_features].select_dtypes(include=np.number)
        if add_const:
            X = sm.add_constant(X, has_constant="add")
        return X
    
    def predict_mean(self, add_constant=None):
        """
        Returns predicted mean count for each row in X_future.
    
        Parameters
        ----------
        add_constant : bool or None — if None, uses setting from fit()
    
        Returns
        -------
        np.array of shape (n,) — predicted mean per observation
        """
        self._check_model_fitted()
        self._check_future_data()
    
        X = self._prepare_future_X(add_constant)
        means = self._fitted_model.predict(X)
    
        results = pd.DataFrame({
            "label":         self._labels,
            "predicted_mean": means.round(3)
        })
    
        print(results.to_string(index=False))
        return means
    
    def predict_pmf(self, max_k=60, add_constant=None):
        """
        Returns full PMF over k = 0, 1, ..., max_k for each row
        in X_future.
    
        Parameters
        ----------
        max_k        : int          — upper bound of counts (default 60)
        add_constant : bool or None — if None, uses setting from fit()
    
        Returns
        -------
        np.array of shape (n, max_k + 1)
        where entry [i, k] = P(Y = k | X_future[i])
        """
        self._check_model_fitted()
        self._check_future_data()
    
        X    = self._prepare_future_X(add_constant)
        pmfs = self._fitted_model.predict_dist(X, max_k=max_k)
    
        # Store for reuse by plot_pmf and predict_future
        self._last_pmfs = pmfs
        return pmfs
    
    def plot_pmf(self, index=None, max_k=60, add_constant=None, actual=None):
        """
        Plots the predicted PMF for future observations.
        If index is provided, plots that single observation.
        If index is None, plots all observations in a grid.
    
        Parameters
        ----------
        index        : int or None — row index to plot. If None, plots all
        max_k        : int         — upper bound of counts (default 60)
        add_constant : bool or None
        actual       : int or list of int or None
                       actual observed value(s) to mark on the plot
        """
        self._check_model_fitted()
        self._check_future_data()
    
        # Reuse stored PMFs if available, else recompute
        if hasattr(self, "_last_pmfs") and self._last_pmfs is not None and \
           self._last_pmfs.shape[1] == max_k + 1:
            pmfs = self._last_pmfs
        else:
            pmfs = self.predict_pmf(max_k=max_k, add_constant=add_constant)
    
        k_vals = np.arange(0, max_k + 1)
    
        def _single_pmf_plot(ax, pmf, label, actual_val=None):
            mean = float(np.sum(k_vals * pmf))
            ax.bar(k_vals, pmf, color="steelblue", alpha=0.8)
            ax.axvline(mean, color="red", linestyle="--",
                       linewidth=1.5, label=f"Mean ({mean:.1f})")
    
            # 80% interval
            cumulative = np.cumsum(pmf)
            lower = k_vals[cumulative >= 0.10][0]
            upper = k_vals[cumulative >= 0.90][0]
            ax.axvspan(lower, upper, alpha=0.1, color="steelblue",
                       label=f"80% ({lower}–{upper})")
    
            if actual_val is not None:
                ax.axvline(actual_val, color="green", linestyle="-",
                           linewidth=2, label=f"Actual ({actual_val})")
    
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Count (k)")
            ax.set_ylabel("P(Y = k)")
            ax.legend(fontsize=7)
    
        # --- Single observation ---
        if index is not None:
            if index >= len(pmfs):
                raise ValueError(
                    f"index {index} out of range. "
                    f"X_future has {len(pmfs)} rows."
                )
            actual_val = actual[index] if isinstance(actual, list) else actual
            fig, ax = plt.subplots(figsize=(12, 5))
            _single_pmf_plot(ax, pmfs[index], self._labels[index], actual_val)
            fig.suptitle(f"Predicted PMF — {self._fitted_model.name}",
                         fontsize=13)
            plt.tight_layout()
            plt.show()
    
            mean = float(np.sum(k_vals * pmfs[index]))
            cumulative = np.cumsum(pmfs[index])
            lower = k_vals[cumulative >= 0.10][0]
            upper = k_vals[cumulative >= 0.90][0]
            print(f"  Label:          {self._labels[index]}")
            print(f"  Predicted mean: {mean:.2f}")
            print(f"  80% interval:   {lower} – {upper}")
    
        # --- All observations in a grid ---
        else:
            n      = len(pmfs)
            n_cols = 3
            n_rows = int(np.ceil(n / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten()
    
            for i in range(n):
                actual_val = (actual[i] if isinstance(actual, list)
                              else actual)
                _single_pmf_plot(axes[i], pmfs[i],
                                 self._labels[i], actual_val)
    
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
    
            fig.suptitle(f"Predicted PMFs — {self._fitted_model.name}",
                         fontsize=13, y=1.02)
            plt.tight_layout()
            plt.show()
    
    def predict_future(self, max_k=60, add_constant=None):
        """
        Runs predictions for all rows in X_future.
        Returns a summary DataFrame and stores full PMFs.
    
        Parameters
        ----------
        max_k        : int          — upper bound of counts (default 60)
        add_constant : bool or None — if None, uses setting from fit()
    
        Returns
        -------
        summary_df : pd.DataFrame — one row per observation with
                                    mean, 80% interval
        pmfs       : np.array of shape (n, max_k + 1)
        """
        self._check_model_fitted()
        self._check_future_data()
    
        pmfs   = self.predict_pmf(max_k=max_k, add_constant=add_constant)
        k_vals = np.arange(0, max_k + 1)
    
        summary_records = []
    
        for i in range(len(pmfs)):
            pmf  = pmfs[i]
            mean = float(np.sum(k_vals * pmf))
    
            cumulative = np.cumsum(pmf)
            lower = k_vals[cumulative >= 0.10][0]
            upper = k_vals[cumulative >= 0.90][0]
    
            summary_records.append({
                "label":          self._labels[i],
                "predicted_mean": round(mean, 2),
                "80_interval":    f"{lower}–{upper}",
                "p_lower_10":     round(float(pmf[:10].sum()), 4),
                "p_upper_30":     round(float(pmf[31:].sum()), 4),
            })
    
        summary_df = pd.DataFrame(summary_records)
    
        # Store on instance for external use
        self._last_predictions = {
            "summary": summary_df,
            "pmfs":    pmfs
        }
    
        print("=" * 55)
        print("  PREDICTION SUMMARY")
        print("=" * 55)
        print(f"\n{summary_df.to_string(index=False)}")
        print(f"\n  Full PMFs stored in self._last_predictions['pmfs']")
    
        return summary_df, pmfs