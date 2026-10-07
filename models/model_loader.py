import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from .model_definitions import FraudMLP, create_ft_transformer

BASE_DIR = Path(__file__).resolve().parent.parent


def find_artifact(*relative_candidates: str) -> Path:
    """Search for artifact in deployment_artifacts/ or root."""
    for rel in relative_candidates:
        candidate = BASE_DIR / rel
        if candidate.exists():
            return candidate
    # Default to first candidate for error message
    expected = BASE_DIR / relative_candidates[0]
    raise FileNotFoundError(f"Model artifact not found.\nExpected:\n{expected}")


@st.cache_resource
def load_tabnet() -> TabNetClassifier:
    """Load the saved TabNet model from directory or zip."""
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    # Check zip candidates first
    for rel_zip in [
        "deployment_artifacts/models/tabnet.zip",
        "tabnet_model.zip",
        "models/tabnet.zip",
    ]:
        zip_path = BASE_DIR / rel_zip
        if zip_path.exists():
            model = TabNetClassifier(device_name=device_name)
            model.load_model(str(zip_path))
            return model

    # Check directory candidates
    for rel_dir in [
        "deployment_artifacts/models/tabnet",
        "tabnet",
        "models/tabnet",
    ]:
        dir_path = BASE_DIR / rel_dir
        if dir_path.is_dir() and (dir_path / "model_params.json").exists() and (dir_path / "network.pt").exists():
            with open(dir_path / "model_params.json") as f:
                loaded_params = json.load(f)
            loaded_params["init_params"]["device_name"] = device_name
            saved_state_dict = torch.load(dir_path / "network.pt", map_location=torch.device(device_name))

            model = TabNetClassifier(**loaded_params["init_params"])
            model._set_network()
            model.network.load_state_dict(saved_state_dict)
            model.network.eval()
            model.load_class_attrs(loaded_params["class_attrs"])
            return model

    raise FileNotFoundError(
        f"Model artifact not found.\nExpected:\ndeployment_artifacts/models/tabnet or tabnet/"
    )


@st.cache_resource
def load_mlp() -> FraudMLP:
    """Load the saved MLP model."""
    weights_path = find_artifact("deployment_artifacts/models/mlp.pth", "mlp.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudMLP(input_dim=30)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_ft_transformer():
    """Load the saved FT-Transformer model."""
    weights_path = find_artifact("deployment_artifacts/models/ft_transformer.pth", "ft_transformer.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_ft_transformer()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler."""
    scaler_path = find_artifact("deployment_artifacts/preprocessing/scaler.pkl", "scaler.pkl")
    return joblib.load(scaler_path)


@st.cache_data
def load_feature_names() -> list:
    """Load the ordered list of feature names."""
    path = find_artifact("deployment_artifacts/preprocessing/feature_names.json", "feature_names.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_thresholds() -> dict:
    """Load the saved decision thresholds."""
    path = find_artifact("deployment_artifacts/metrics/thresholds.json", "thresholds.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict:
    """Load the saved model evaluation results."""
    path = find_artifact("deployment_artifacts/metrics/model_results.json", "model_results.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_test_data() -> pd.DataFrame:
    """Load the test results CSV."""
    path = find_artifact("deployment_artifacts/data/test_results.csv", "test_results.csv")
    return pd.read_csv(path)


@st.cache_data
def load_shap_global() -> pd.DataFrame:
    """Load TabNet SHAP global importance."""
    path = find_artifact(
        "deployment_artifacts/xai/tabnet_shap_global_importance.csv",
        "tabnet_shap_global_importance.csv",
    )
    return pd.read_csv(path)


@st.cache_data
def load_shap_local() -> pd.DataFrame:
    """Load TabNet SHAP local explanations."""
    path = find_artifact(
        "deployment_artifacts/xai/tabnet_shap_local.csv",
        "tabnet_shap_local.csv",
    )
    return pd.read_csv(path)


@st.cache_data
def load_lime_local() -> pd.DataFrame:
    """Load TabNet LIME local explanations."""
    path = find_artifact(
        "deployment_artifacts/xai/tabnet_lime_local.csv",
        "tabnet_lime_local.csv",
    )
    return pd.read_csv(path)


@st.cache_data
def load_shap_lime_comparison() -> pd.DataFrame:
    """Load SHAP and LIME comparison table."""
    path = find_artifact(
        "deployment_artifacts/xai/shap_lime_comparison.csv",
        "shap_lime_comparison.csv",
    )
    return pd.read_csv(path)


@st.cache_data
def load_explained_transaction() -> dict:
    """Load the specific transaction explained by SHAP and LIME."""
    path = find_artifact(
        "deployment_artifacts/xai/explained_transaction.json",
        "explained_transaction.json",
    )
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_shap_sample_values() -> pd.DataFrame:
    """Load the precomputed SHAP values for the 20 test sample transactions."""
    path = find_artifact(
        "deployment_artifacts/xai/tabnet_shap_values.csv",
        "tabnet_shap_values.csv",
    )
    return pd.read_csv(path)
