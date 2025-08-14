# app.py
# Streamlit KDD99 IDS — CSV input + row-level SHAP for RF, ISO, AE
# Fixed model paths under /mnt/data. No emojis in UI.

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

# Optional (only for AE)
try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

# ----------------
# Fixed file paths
# ----------------
RF_PIPE_PATH    = "rf_pipeline_kdd.pkl"
ISO_PIPE_PATH   = "iso_pipeline_kdd.pkl"
AE_MODEL_PATH   = "ae_model_kdd.h5"
AE_PREPROC_PATH = "ae_preprocessor_kdd.pkl"

# -----------
# Page setup
# ----------st.set_page_config(page_title="KDD99 IDS — SHAP", layout="wide")
st.title("NSL-KDD Intrusion Detection System")

# ---------------
# Load the assets
# ---------------
@st.cache_resource(show_spinner=False)
def load_assets():
    rf_pipe = joblib.load(RF_PIPE_PATH) if os.path.exists(RF_PIPE_PATH) else None
    iso_pipe = joblib.load(ISO_PIPE_PATH) if os.path.exists(ISO_PIPE_PATH) else None

    ae_pre = joblib.load(AE_PREPROC_PATH) if os.path.exists(AE_PREPROC_PATH) else None
    ae_model = None
    if keras_load_model and os.path.exists(AE_MODEL_PATH):
        try:
            ae_model = keras_load_model(AE_MODEL_PATH, compile=False)
        except Exception:
            ae_model = None

    return rf_pipe, iso_pipe, ae_model, ae_pre

rf_pipe, iso_pipe, ae_model, ae_pre = load_assets()

# ----------------
# Utility helpers
# ----------------
def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def render_current_fig():
    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)
    plt.clf(); plt.close("all")

def get_feature_names_out(preproc, input_cols=None):
    if preproc is None:
        return None
    try:
        return list(preproc.get_feature_names_out())
    except Exception:
        pass
    names = []
    try:
        for name, trans, cols in preproc.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            est = trans
            if hasattr(trans, "steps"):
                est = trans.steps[-1][1]
            if hasattr(est, "get_feature_names_out"):
                try:
                    sub = est.get_feature_names_out(cols)
                except Exception:
                    sub = est.get_feature_names_out()
                names.extend([str(s) for s in sub])
            else:
                if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                    names.extend([str(c) for c in cols])
                else:
                    names.append(str(cols))
    except Exception:
        return None
    return names

def split_pipeline(pipe):
    """Return (preprocessor, estimator) from an sklearn Pipeline if possible."""
    pre, est = None, pipe
    try:
        if hasattr(pipe, "named_steps"):
            steps = list(pipe.named_steps.items())
            if steps:
                est = steps[-1][1]
            for _, obj in steps:
                if isinstance(obj, ColumnTransformer):
                    pre = obj; break
    except Exception:
        pass
    return pre, est

def make_single_output_exp(exp, class_idx=None):
    """
    Collapse a row-level shap.Explanation to single output so the waterfall
    always receives 1D values and a scalar base_value.
    """
    import numpy as np, shap as _shap
    vals = exp.values
    base = exp.base_values
    data = exp.data
    names = getattr(exp, "feature_names", None)

    if np.ndim(vals) == 1:
        b = float(np.ravel(base)[0]) if np.size(base) else float(base)
        return _shap.Explanation(values=vals, base_values=b, data=data, feature_names=names)

    if np.ndim(vals) == 2:
        if class_idx is None:
            out_names = getattr(exp, "output_names", None)
            class_idx = out_names.index("1") if (out_names and "1" in out_names) else vals.shape[0] - 1
        v = vals[class_idx]
        b_arr = np.ravel(base)
        b = float(b_arr[class_idx]) if b_arr.size > 1 else float(b_arr[0])
        return _shap.Explanation(values=v, base_values=b, data=data, feature_names=names)

    return exp

def background_sample(X_dense, k=50, seed=0):
    """
    Return a numpy array of background rows (never a DenseData).
    If only one row is available, synthesize a small noisy cloud.
    """
    rng = np.random.default_rng(seed)
    n = X_dense.shape[0]
    if n <= 1:
        reps = max(32, k)
        X_aug = np.repeat(X_dense, reps, axis=0)
        std = X_aug.std(axis=0, keepdims=True) + 1e-6
        noise = rng.normal(0, 0.02 * std, size=X_aug.shape)
        return X_aug + noise
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    return X_dense[idx]

# ---------------------
# Upload CSV of samples
# ---------------------
st.subheader("Upload data (CSV)")
csv_file = st.file_uploader("KDD99 raw feature CSV (columns must match training schema)", type=["csv"])
if csv_file is None:
    st.info("Upload your CSV to continue.")
    st.stop()

try:
    input_df = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.markdown("Preview")
st.dataframe(input_df.head(), use_container_width=True)

# --------------
# Run Prediction
# --------------
st.divider()
run_clicked = st.button("Run Prediction", type="primary")
if "_results" not in st.session_state:
    st.session_state._results = None

if run_clicked:
    results = input_df.copy()

    # Random Forest
    if rf_pipe is not None:
        try:
            rf_pred = rf_pipe.predict(input_df)
            results["RandomForest"] = ["Attack" if p == 1 else "Normal" for p in rf_pred]
            if hasattr(rf_pipe, "predict_proba"):
                try:
                    results["RF_Prob"] = rf_pipe.predict_proba(input_df)[:, 1]
                except Exception:
                    pass
        except Exception as e:
            st.error(f"RF prediction failed: {e}")

    # Isolation Forest
    if iso_pipe is not None:
        try:
            iso_raw = iso_pipe.predict(input_df)  # -1 anomaly, 1 normal
            iso_pred = np.where(iso_raw == -1, 1, 0)
            results["IsolationForest"] = ["Attack" if p == 1 else "Normal" for p in iso_pred]
        except Exception as e:
            st.error(f"Isolation Forest prediction failed: {e}")

    # Autoencoder
    if ae_model is not None and ae_pre is not None:
        try:
            X_ae = ae_pre.transform(input_df); X_ae = to_dense(X_ae)
            recon = ae_model.predict(X_ae, verbose=0)
            mse = np.mean((X_ae - recon) ** 2, axis=1)
            med = np.median(mse); mad = np.median(np.abs(mse - med)) + 1e-9
            thr = med + 3.0 * mad
            results["Autoencoder"] = ["Attack" if v > thr else "Normal" for v in mse]
            results["AE_MSE"] = mse
            st.caption(f"AE threshold (median+3*MAD): {thr:.6f}")
        except Exception as e:
            st.error(f"Autoencoder scoring failed: {e}")

    st.session_state._results = results

# Results + download
if st.session_state._results is not None:
    st.subheader("Results")
    st.dataframe(st.session_state._results, use_container_width=True)
    st.download_button(
        "Download results CSV",
        st.session_state._results.to_csv(index=False).encode("utf-8"),
        file_name="kdd99_predictions.csv",
        mime="text/csv",
    )

# -------------------------------
# Explain a single flow (SHAP UI)
# -------------------------------
st.divider()
st.header("Explain a single flow (SHAP)")
row_idx = st.number_input(
    "Row index to explain",
    min_value=0,
    max_value=max(len(input_df) - 1, 0),
    value=0,
    step=1
)

if st.button("Compute SHAP for selected row"):
    # -------- Random Forest (TreeExplainer, tree_path_dependent, raw) --------
    if rf_pipe is not None:
        st.subheader("Random Forest — SHAP (raw, tree_path_dependent)")
        rf_pre, rf_est = split_pipeline(rf_pipe)

        try:
            if rf_pre is not None and rf_est is not None:
                X_rf = rf_pre.transform(input_df)
                X_rf_dense = to_dense(X_rf)
                feat_names = get_feature_names_out(rf_pre, input_df.columns)

                # SHAP requirement for models with categorical splits:
                # use feature_perturbation="tree_path_dependent" and DO NOT pass background
                expl = shap.TreeExplainer(
                    rf_est,
                    model_output="raw",  # required with tree_path_dependent
                    feature_perturbation="tree_path_dependent",
                )
                row = X_rf_dense[row_idx:row_idx+1]
                row_exp = expl(row)[0]

                # attach feature names for readability (optional)
                if getattr(row_exp, "feature_names", None) is None and feat_names is not None:
                    row_exp = shap.Explanation(values=row_exp.values,
                                               base_values=row_exp.base_values,
                                               data=row_exp.data,
                                               feature_names=feat_names)
            else:
                # Fallback: generic explainer on raw inputs with explicit masker on probability
                raw_bg = background_sample(input_df.values, k=min(50, len(input_df)))
                masker = shap.maskers.Independent(raw_bg)
                def f_rf(A):
                    df = pd.DataFrame(A, columns=input_df.columns)
                    return rf_pipe.predict_proba(df)[:, 1]
                expl = shap.Explainer(f_rf, masker, algorithm="auto")
                row_exp = expl(input_df.iloc[row_idx:row_idx+1])[0]

            # choose class index if multi-output (usually not needed here)
            class_idx = None
            try:
                classes = getattr(rf_est, "classes_", getattr(rf_pipe, "classes_", None))
                if classes is not None and len(classes) > 1:
                    class_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else len(classes) - 1
            except Exception:
                pass

            single = make_single_output_exp(row_exp, class_idx)

            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(single, max_display=12, show=False)
            render_current_fig()

        except Exception as e:
            st.error(f"RF SHAP failed: {e}")

    # -------- Isolation Forest (anomaly score) --------
    if iso_pipe is not None:
        st.subheader("Isolation Forest — SHAP (anomaly score)")
        iso_pre, iso_est = split_pipeline(iso_pipe)
        try:
            if iso_pre is not None and iso_est is not None:
                X_iso = iso_pre.transform(input_df); X_iso_dense = to_dense(X_iso)
                feat_names_iso = get_feature_names_out(iso_pre, input_df.columns)

                def f_iso(A):
                    return (-iso_est.decision_function(A)).astype(float)

                bg_iso = background_sample(X_iso_dense, k=50)
                masker_iso = shap.maskers.Independent(bg_iso)
                expl_iso = shap.Explainer(f_iso, masker_iso, algorithm="auto", feature_names=feat_names_iso)
                exp_iso = expl_iso(X_iso_dense[row_idx:row_idx+1])[0]
            else:
                cols = list(input_df.columns)
                def f_iso_raw(A):
                    df = pd.DataFrame(A, columns=cols)
                    return (-iso_pipe.decision_function(df)).astype(float)

                raw_bg = background_sample(input_df.values, k=min(50, len(input_df)))
                masker_raw = shap.maskers.Independent(raw_bg)
                expl_iso = shap.Explainer(f_iso_raw, masker_raw, algorithm="auto")
                exp_iso = expl_iso(input_df.iloc[row_idx:row_idx+1])[0]

            single_iso = make_single_output_exp(exp_iso, None)
            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(single_iso, max_display=12, show=False)
            render_current_fig()
        except Exception as e:
            st.error(f"Isolation Forest SHAP failed: {e}")

    # -------- Autoencoder (MSE) --------
    if ae_model is not None and ae_pre is not None:
        st.subheader("Autoencoder — SHAP (reconstruction MSE)")
        try:
            X_ae = ae_pre.transform(input_df); X_ae_dense = to_dense(X_ae)
            feat_names_ae = get_feature_names_out(ae_pre, input_df.columns)

            def f_ae(A):
                recon = ae_model.predict(A, verbose=0)
                return np.mean((A - recon) ** 2, axis=1)

            bg_ae = background_sample(X_ae_dense, k=50)
            masker_ae = shap.maskers.Independent(bg_ae)
            expl_ae = shap.Explainer(f_ae, masker_ae, algorithm="auto", feature_names=feat_names_ae)
            exp_ae = expl_ae(X_ae_dense[row_idx:row_idx+1])[0]

            single_ae = make_single_output_exp(exp_ae, None)
            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(single_ae, max_display=12, show=False)
            render_current_fig()
        except Exception as e:
            st.error(f"Autoencoder SHAP failed: {e}")
