import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap
import dill
import matplotlib.pyplot as plt


class InsightAIMLModel(BaseEstimator, ClassifierMixin):
    """
    A custom fraud detection model that combines LightGBM and XGBoost using a stacking
    approach with a logistic regression meta-classifier. It includes custom thresholding
    and business rules for fraud detection, as well as explainability using SHAP.
    """

    def __init__(
        self,
        preprocessor,
        val_size=0.2,
        allowed_types=("TRANSFER", "CASH_OUT"),
        recall_min=0.995,
        thr_start=0.50,
        thr_end=0.999,
        thr_points=100,
        cast_float32=True,
        random_state=42,
        lgbm_params=None,
        xgb_params=None,
        meta_params=None,
        reason_labels=None,
    ):
        """
        Initializes the InsightAIMLModel.

        Args:
            preprocessor (ColumnTransformer): The preprocessing pipeline for the features.
            val_size (float): The proportion of the training data to use for validation.
            allowed_types (tuple): The transaction types that can be flagged as fraudulent.
            recall_min (float): The minimum recall to achieve when selecting the threshold.
            thr_start (float): The starting point for the threshold search.
            thr_end (float): The ending point for the threshold search.
            thr_points (int): The number of points to check in the threshold search.
            cast_float32 (bool): Whether to cast the preprocessed data to float32.
            random_state (int): The random state for reproducibility.
            lgbm_params (dict): Parameters for the LGBMClassifier.
            xgb_params (dict): Parameters for the XGBClassifier.
            meta_params (dict): Parameters for the LogisticRegression meta-classifier.
            reason_labels (dict): A dictionary to map feature names to more readable labels.
        """
        self.preprocessor = preprocessor
        self.val_size = val_size
        self.allowed_types = tuple(allowed_types)
        self.recall_min = recall_min
        self.thr_start = thr_start
        self.thr_end = thr_end
        self.thr_points = thr_points
        self.cast_float32 = cast_float32
        self.random_state = random_state
        self.lgbm_params = lgbm_params or dict(
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.xgb_params = xgb_params or dict(
            n_estimators=100,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            grow_policy="lossguide",
            random_state=random_state,
            n_jobs=-1,
        )
        self.meta_params = meta_params or dict(max_iter=1000)
        self.reason_labels = reason_labels or {}

    def _make_pipes(self, y_tr):
        """
        Creates the base model pipelines.
        """
        to_float32 = (
            FunctionTransformer(lambda x: x.astype(np.float32))
            if self.cast_float32
            else FunctionTransformer(lambda x: x)
        )
        if "scale_pos_weight" not in self.xgb_params:
            pos_weight = y_tr.value_counts()[0] / y_tr.value_counts()[1]
            self.xgb_params = {**self.xgb_params, "scale_pos_weight": float(pos_weight)}
        self.lgbm_pipe_ = Pipeline(
            [
                ("pre", self.preprocessor),
                ("cast", to_float32),
                ("clf", LGBMClassifier(**self.lgbm_params)),
            ]
        )
        self.xgb_pipe_ = Pipeline(
            [
                ("pre", self.preprocessor),
                ("cast", to_float32),
                ("clf", XGBClassifier(**self.xgb_params)),
            ]
        )

    def _pick_threshold(self, y_true, proba, allowed_mask):
        """
        Selects the best threshold based on the validation set.
        """
        best = None
        for t in np.linspace(self.thr_start, self.thr_end, self.thr_points):
            preds = ((proba >= t) & allowed_mask).astype(int)
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            fp, fn = cm[0, 1], cm[1, 0]
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            if rec >= self.recall_min:
                key = (fp, -prec, fn)
                if best is None or key < best[0]:
                    best = (key, t)
        return 0.5 if best is None else best[1]

    @staticmethod
    def _to_numpy(Xt):
        if hasattr(Xt, "toarray"):
            return Xt.toarray()
        return np.asarray(Xt)

    @staticmethod
    def _get_feature_names(pre):
        try:
            return pre.get_feature_names_out()
        except Exception:
            return None

    @staticmethod
    def _extract_pos_shap(shap_values):
        if isinstance(shap_values, (list, tuple)) and len(shap_values) > 1:
            return shap_values[1]
        return shap_values

    def _tree_shap_single(self, pipe, Xt_np, feature_names):
        try:
            import shap

            clf = pipe.named_steps["clf"]
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xt_np)
            sv_pos = self._extract_pos_shap(sv).reshape(-1)
            expected = explainer.expected_value
            if isinstance(expected, (list, tuple, np.ndarray)):
                expected = expected[-1]
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(Xt_np.shape[1])]
            df = pd.DataFrame({"feature": feature_names, "shap": sv_pos})
            df["abs_shap"] = df["shap"].abs()
            df = df.sort_values("abs_shap", ascending=False)
            return df, float(expected), True, None
        except Exception as e:
            return pd.DataFrame(), None, False, str(e)

    @staticmethod
    def _logit_to_prob(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _label(self, feat):
        return self.reason_labels.get(feat, feat)

    def fit(self, X, y):
        """
        Fits the model to the training data.
        """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.random_state
        )
        self.classes_ = np.unique(y)
        self._make_pipes(y_tr)
        self.lgbm_pipe_.fit(X_tr, y_tr)
        self.xgb_pipe_.fit(X_tr, y_tr)
        lgbm_val = self.lgbm_pipe_.predict_proba(X_val)[:, 1]
        xgb_val = self.xgb_pipe_.predict_proba(X_val)[:, 1]
        meta_X_val = np.vstack([lgbm_val, xgb_val]).T
        self.meta_model_ = LogisticRegression(**self.meta_params)
        self.meta_model_.fit(meta_X_val, y_val)
        types_val = (
            X_val["type"].values
            if "type" in X_val.columns
            else np.array([None] * len(X_val))
        )
        allowed_val = np.isin(types_val, self.allowed_types)
        meta_val_proba = self.meta_model_.predict_proba(meta_X_val)[:, 1]
        self.threshold_ = self._pick_threshold(y_val, meta_val_proba, allowed_val)
        return self

    def predict_proba(self, X):
        """
        Predicts the class probabilities for the input data.
        """
        lgbm_p = self.lgbm_pipe_.predict_proba(X)[:, 1]
        xgb_p = self.xgb_pipe_.predict_proba(X)[:, 1]
        meta_X = np.vstack([lgbm_p, xgb_p]).T
        p1 = self.meta_model_.predict_proba(meta_X)[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """
        Predicts the class labels for the input data.
        """
        proba = self.predict_proba(X)[:, 1]
        types_arr = (
            X["type"].values if "type" in X.columns else np.array([None] * len(X))
        )
        allowed = np.isin(types_arr, self.allowed_types)
        return ((proba >= self.threshold_) & allowed).astype(int)

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

    def set_threshold(self, X_val, y_val):
        p = self.predict_proba(X_val)[:, 1]
        types_arr = (
            X_val["type"].values
            if "type" in X_val.columns
            else np.array([None] * len(X_val))
        )
        allowed = np.isin(types_arr, self.allowed_types)
        self.threshold_ = self._pick_threshold(y_val, p, allowed)
        return self.threshold_

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def evaluate(self, X, y):
        p = self.predict_proba(X)[:, 1]
        preds = self.predict(X)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, p)
        return {
            "confusion_matrix": cm,
            "precision": prec,
            "recall": rec,
            "roc_auc": auc,
            "threshold": getattr(self, "threshold_", None),
        }

    def explain(self, X_row: pd.DataFrame, top_k: int = 12, plot: bool = False):
        if not isinstance(X_row, pd.DataFrame):
            raise ValueError("X_row must be a pandas DataFrame.")
        X_row = X_row.iloc[:1].copy()
        if "type" in X_row.columns:
            type_val = X_row["type"].iloc[0]
            allowed = bool(type_val in self.allowed_types)
        else:
            type_val = None
            allowed = False
        p_lgbm = float(self.lgbm_pipe_.predict_proba(X_row)[:, 1][0])
        p_xgb = float(self.xgb_pipe_.predict_proba(X_row)[:, 1][0])
        coef = self.meta_model_.coef_.ravel()
        intercept = float(self.meta_model_.intercept_.ravel()[0])
        z = intercept + coef[0] * p_lgbm + coef[1] * p_xgb
        p_meta = float(self._logit_to_prob(z))
        final_pred = int(allowed and (p_meta >= self.threshold_))
        pre = self.lgbm_pipe_.named_steps["pre"]
        Xt = pre.transform(X_row)
        if self.cast_float32 and hasattr(Xt, "astype"):
            try:
                Xt = Xt.astype(np.float32)
            except Exception:
                pass
        Xt_np = self._to_numpy(Xt)
        feat_names = self._get_feature_names(pre)
        lgbm_df, lgbm_base, lgbm_ok, lgbm_err = self._tree_shap_single(
            self.lgbm_pipe_, Xt_np, feat_names
        )
        xgb_df, xgb_base, xgb_ok, xgb_err = self._tree_shap_single(
            self.xgb_pipe_, Xt_np, feat_names
        )
        if plot and lgbm_ok and not lgbm_df.empty:
            import matplotlib.pyplot as plt

            top = lgbm_df.head(top_k).iloc[::-1]
            plt.figure(figsize=(6, max(2, 0.35 * len(top))))
            plt.barh(top["feature"], top["shap"])
            plt.title("LGBM SHAP contributions (sample)")
            plt.tight_layout()
            plt.show()
        if plot and xgb_ok and not xgb_df.empty:
            import matplotlib.pyplot as plt

            top = xgb_df.head(top_k).iloc[::-1]
            plt.figure(figsize=(6, max(2, 0.35 * len(top))))
            plt.barh(top["feature"], top["shap"])
            plt.title("XGB SHAP contributions (sample)")
            plt.tight_layout()
            plt.show()
        return {
            "type_value": type_val,
            "allowed_type": allowed,
            "p_lgbm": p_lgbm,
            "p_xgb": p_xgb,
            "meta_intercept": intercept,
            "meta_coef_lgbm": float(coef[0]),
            "meta_coef_xgb": float(coef[1]),
            "meta_logit": float(z),
            "meta_proba": p_meta,
            "threshold": float(getattr(self, "threshold_", 0.5)),
            "final_pred": final_pred,
            "lgbm_shap_available": lgbm_ok,
            "xgb_shap_available": xgb_ok,
            "lgbm_shap_error": lgbm_err,
            "xgb_shap_error": xgb_err,
            "lgbm_top_features": lgbm_df.head(top_k).reset_index(drop=True),
            "xgb_top_features": xgb_df.head(top_k).reset_index(drop=True),
        }

    def explain_text(self, X_row: pd.DataFrame, top_k: int = 5):
        if not isinstance(X_row, pd.DataFrame):
            raise ValueError("X_row must be a pandas DataFrame.")
        X_row = X_row.iloc[:1].copy()
        exp = self.explain(X_row, top_k=max(top_k, 10), plot=False)
        p_lgbm = exp["p_lgbm"]
        p_xgb = exp["p_xgb"]
        p_meta = exp["meta_proba"]
        thr = exp["threshold"]
        allowed = exp["allowed_type"]
        tval = exp["type_value"]
        final_pred = exp["final_pred"]
        w_l = exp["meta_coef_lgbm"]
        w_x = exp["meta_coef_xgb"]
        reasons = []
        counter_reasons = []
        if exp["lgbm_shap_available"] and exp["xgb_shap_available"]:
            ldf = exp["lgbm_top_features"][["feature", "shap"]].copy()
            xdf = exp["xgb_top_features"][["feature", "shap"]].copy()
            ldf = ldf.groupby("feature", as_index=False)["shap"].sum()
            xdf = xdf.groupby("feature", as_index=False)["shap"].sum()
            merged = ldf.merge(xdf, on="feature", how="outer", suffixes=("_lgbm", "_xgb")).fillna(
                0.0
            )
            aw_l, aw_x = abs(w_l), abs(w_x)
            denom = aw_l + aw_x if (aw_l + aw_x) > 0 else 1.0
            wl, wx = aw_l / denom, aw_x / denom
            merged["combined_shap"] = (
                wl * merged["shap_lgbm"] + wx * merged["shap_xgb"]
            )
            merged["abs_combined"] = merged["combined_shap"].abs()
            merged = merged.sort_values("abs_combined", ascending=False)
            pos_df = merged[merged["combined_shap"] > 0].head(top_k)
            for _, r in pos_df.iterrows():
                reasons.append(
                    {
                        "feature": self._label(r["feature"]),
                        "raw_feature": r["feature"],
                        "impact": float(r["combined_shap"]),
                        "direction": "increases_fraud_risk",
                    }
                )
            neg_df = merged[merged["combined_shap"] < 0].head(top_k)
            for _, r in neg_df.iterrows():
                counter_reasons.append(
                    {
                        "feature": self._label(r["feature"]),
                        "raw_feature": r["feature"],
                        "impact": float(r["combined_shap"]),
                        "direction": "reduces_fraud_risk",
                    }
                )
        else:
            reasons = [
                {
                    "feature": "Model signals",
                    "raw_feature": "probabilities",
                    "impact": float(max(p_lgbm, p_xgb) - thr),
                    "direction": "increases_fraud_risk"
                    if (p_meta >= thr and allowed)
                    else "reduces_fraud_risk",
                }
            ]
        if not allowed:
            rule_text = f"Transaction type={tval!r} is outside monitored types {list(self.allowed_types)}; treated as NOT FRAUD."
        else:
            rule_text = f"Transaction type={tval!r} is within monitored types {list(self.allowed_types)}."
        decision_text = "Flagged as FRAUD." if final_pred == 1 else "Predicted as NOT FRAUD."
        details_text = f"{rule_text} Combined probability={p_meta:.6f} vs threshold={thr:.6f} " f"(LightGBM={p_lgbm:.6f}, XGBoost={p_xgb:.6f})."
        return {
            "decision": int(final_pred),
            "decision_text": decision_text,
            "details_text": details_text,
            "probabilities": {
                "p_meta": float(p_meta),
                "p_lgbm": float(p_lgbm),
                "p_xgb": float(p_xgb),
                "threshold": float(thr),
            },
            "business_rule": {
                "allowed": bool(allowed),
                "type_value": tval,
                "allowed_types": list(self.allowed_types),
            },
            "reasons": reasons,
            "counter_reasons": counter_reasons,
        }
