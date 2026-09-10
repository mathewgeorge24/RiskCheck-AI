from __future__ import annotations
import requests
import pandas as pd
from .features import FEATURES, REQUIRED_COLUMNS


def validate_invoice_frame(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def risk_band(score: float) -> tuple[str, int, str, str]:
    if score < 20:
        return "Low Risk", 5, "Priority 5 - Routine", "Routine processing with continued monitoring"
    if score < 40:
        return "Moderate Risk", 4, "Priority 4 - Low", "Monitor and review if additional risk signals emerge"
    if score < 60:
        return "Elevated Risk", 3, "Priority 3 - Medium", "Standard manual review"
    if score < 80:
        return "High Risk", 2, "Priority 2 - High", "High-priority manual review"
    return "Critical Risk", 1, "Priority 1 - Immediate", "Immediate fraud investigation"


def investigation_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if int(row.get("split_invoice_flag", 0)) == 1:
        reasons.append("Split-invoice behaviour detected")
    if int(row.get("blacklisted_flag", 0)) == 1:
        reasons.append("Supplier is blacklisted")
    ratio = float(row.get("amount_to_supplier_90d_avg_ratio", 1.0))
    if ratio < 0.60 or ratio > 1.50:
        reasons.append("Invoice amount differs substantially from recent supplier pattern")
    if float(row.get("amount_diff_supplier_90d_avg", 0.0)) > 5000:
        reasons.append("Large deviation from supplier 90-day average")
    if float(row.get("supplier_risk_score", 0.0)) >= 0.60:
        reasons.append("Elevated supplier risk score")
    if abs(float(row.get("invoice_amount_zscore", 0.0))) >= 2:
        reasons.append("Unusual invoice amount relative to historical distribution")
    return reasons


def _extract_probabilities(payload: dict, expected_rows: int) -> list[float]:
    """Accept common Databricks/MLflow response shapes.

    Preferred endpoint output is one fraud probability per row. If your served
    model returns a different shape, adapt only this function.
    """
    preds = payload.get("predictions", payload.get("outputs", payload))
    if isinstance(preds, dict) and "fraud_probability" in preds:
        preds = preds["fraud_probability"]
    if not isinstance(preds, list):
        raise ValueError(f"Unexpected serving response: {type(preds).__name__}")

    out: list[float] = []
    for p in preds:
        if isinstance(p, dict):
            if "fraud_probability" in p:
                out.append(float(p["fraud_probability"]))
            elif "probability" in p:
                prob = p["probability"]
                out.append(float(prob[1] if isinstance(prob, (list, tuple)) else prob))
            else:
                raise ValueError(f"Prediction object has no probability field: {p}")
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append(float(p[1]))
        else:
            out.append(float(p))

    if len(out) != expected_rows:
        raise ValueError(f"Endpoint returned {len(out)} predictions for {expected_rows} rows")
    return out


def score_via_databricks(df: pd.DataFrame, host: str, token: str, endpoint: str, chunk_size: int = 1000) -> pd.DataFrame:
    missing = validate_invoice_frame(df)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    request_df = df[FEATURES].copy()
    records = request_df.to_dict(orient="records")
    url = f"{host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    all_probabilities: list[float] = []
    for i in range(0, len(records), chunk_size):
        chunk_records = records[i : i + chunk_size]
        response = requests.post(
            url,
            headers=headers,
            json={"dataframe_records": chunk_records},
            timeout=120,
        )
        response.raise_for_status()
        chunk_probs = _extract_probabilities(response.json(), len(chunk_records))
        all_probabilities.extend(chunk_probs)

    scored = df.copy()
    scored["fraud_probability"] = all_probabilities
    scored["riskcheck_score"] = (scored["fraud_probability"] * 100).round(2)
    mapped = scored["riskcheck_score"].apply(risk_band)
    scored[["risk_category", "priority_rank", "priority_level", "recommended_action"]] = pd.DataFrame(
        mapped.tolist(), index=scored.index
    )
    scored["fraud_prediction"] = (scored["fraud_probability"] >= 0.5).astype(int)
    scored["investigation_reasons"] = scored.apply(investigation_reasons, axis=1)
    return scored.sort_values(["priority_rank", "riskcheck_score"], ascending=[True, False])

