import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from . import config

class UHIModeler:
    def __init__(self, csv_path=config.CSV_PATH):
        self.csv_path = csv_path
        self.df = None
        self.results = {}

    def load_and_preprocess(self):
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        # Aggregation
        print("Aggregating data...")
        self.df = df.groupby([config.GROUP_COL, 'year'])[config.FEATURES_TO_AGG].median().reset_index()
        self.df = self.df.dropna()
        print(f"Processed data shape: {self.df.shape}")

    def train_and_evaluate(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return rmse, r2

    def run_stability_check(self, n_splits=config.N_SPLITS):
        if self.df is None:
            self.load_and_preprocess()

        X = self.df[config.FEATURE_COLS]
        y = self.df[config.TARGET_COL]
        groups = self.df[config.GROUP_COL]

        gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=config.RANDOM_STATE)
        
        models = {
            'LR': Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())]),
            'RF': RandomForestRegressor(n_estimators=100, n_jobs=-1),
            'XGB': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, n_jobs=-1)
        }

        metrics = {name: {'rmse': [], 'r2': []} for name in models}
        importances = {name: {'shap': [], 'perm': []} for name in models if name in ['RF', 'XGB']}

        print(f"Performing Stability Check ({n_splits} splits)...")
        
        for i, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for name, model in models.items():
                # Set random state for reproducibility within splits
                if hasattr(model, 'random_state'):
                    model.random_state = config.RANDOM_STATE + i
                elif isinstance(model, Pipeline) and hasattr(model.named_steps['regressor'], 'random_state'):
                     model.named_steps['regressor'].random_state = config.RANDOM_STATE + i

                rmse, r2 = self.train_and_evaluate(model, X_train, X_test, y_train, y_test)
                metrics[name]['rmse'].append(rmse)
                metrics[name]['r2'].append(r2)

                # Importance Analysis for Tree-based models
                if name in importances:
                    # SHAP
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_test)
                    importances[name]['shap'].append(np.abs(shap_values.values).mean(axis=0))

                    # Permutation
                    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=config.RANDOM_STATE + i)
                    importances[name]['perm'].append(perm.importances_mean)

        self._print_results(metrics, importances, X.columns)

    def _print_results(self, metrics, importances, columns):
        print("\n=== Stability Check Results (Mean ± Std) ===")
        for name, m in metrics.items():
            print(f"{name}: RMSE={np.mean(m['rmse']):.4f}±{np.std(m['rmse']):.4f}, R2={np.mean(m['r2']):.4f}±{np.std(m['r2']):.4f}")

        for name, imp in importances.items():
            print(f"\n=== {name} Importance Analysis ===")
            shap_df = pd.DataFrame(imp['shap'], columns=columns).drop(columns=['year'], errors='ignore')
            perm_df = pd.DataFrame(imp['perm'], columns=columns).drop(columns=['year'], errors='ignore')
            
            print(f"SHAP (Mean):\n{shap_df.mean().sort_values(ascending=False)}")
            print(f"Permutation (Mean):\n{perm_df.mean().sort_values(ascending=False)}")
