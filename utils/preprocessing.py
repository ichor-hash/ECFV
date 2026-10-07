import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple


def get_device() -> torch.device:
    """Get the available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_single_transaction(
    features_scaled: np.ndarray,
    tabnet_model,
    mlp_model,
    ft_model,
    thresholds: dict,
    device: torch.device
) -> Dict[str, dict]:
    """Get predictions from all three models for a single scaled transaction.
    
    Args:
        features_scaled: Scaled feature array of shape (1, 30)
        tabnet_model: Loaded TabNet model
        mlp_model: Loaded MLP model
        ft_model: Loaded FT-Transformer model
        thresholds: Dict of model name -> threshold value
        device: torch device
    
    Returns:
        Dict with model name -> {probability, threshold, prediction}
    """
    results = {}
    
    # TabNet prediction
    tabnet_proba = tabnet_model.predict_proba(features_scaled)[:, 1][0]
    tabnet_thresh = thresholds["TabNet"]
    results["TabNet"] = {
        "probability": float(tabnet_proba),
        "threshold": tabnet_thresh,
        "prediction": "Fraud" if tabnet_proba >= tabnet_thresh else "Legitimate"
    }
    
    # MLP prediction
    with torch.no_grad():
        x_tensor = torch.FloatTensor(features_scaled).to(device)
        mlp_logit = mlp_model(x_tensor)
        mlp_proba = torch.sigmoid(mlp_logit).cpu().numpy()[0]
    mlp_thresh = thresholds["MLP"]
    results["MLP"] = {
        "probability": float(mlp_proba),
        "threshold": mlp_thresh,
        "prediction": "Fraud" if mlp_proba >= mlp_thresh else "Legitimate"
    }
    
    # FT-Transformer prediction
    with torch.no_grad():
        x_tensor = torch.FloatTensor(features_scaled).to(device)
        ft_logit = ft_model(x_tensor, None)
        ft_proba = torch.sigmoid(ft_logit).cpu().numpy()[0][0]
    ft_thresh = thresholds["FT-Transformer"]
    results["FT-Transformer"] = {
        "probability": float(ft_proba),
        "threshold": ft_thresh,
        "prediction": "Fraud" if ft_proba >= ft_thresh else "Legitimate"
    }
    
    return results


def get_model_agreement(predictions: Dict[str, dict]) -> Tuple[int, str]:
    """Compute model agreement level.
    
    Returns:
        (count_fraud, description) where count_fraud is number of models predicting fraud
    """
    fraud_count = sum(1 for v in predictions.values() if v["prediction"] == "Fraud")
    total = len(predictions)
    
    if fraud_count == total:
        return fraud_count, "Unanimous: All models predict Fraud"
    elif fraud_count == 0:
        return fraud_count, "Unanimous: All models predict Legitimate"
    elif fraud_count >= total / 2:
        return fraud_count, f"Majority Agreement: {fraud_count}/{total} models predict Fraud"
    else:
        return fraud_count, f"Disagreement: Only {fraud_count}/{total} models predict Fraud"
