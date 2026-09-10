from __future__ import annotations

import json
import time
import uuid

import pandas as pd
import requests
from databricks import sql


# =========================================================
# DATABRICKS SQL CONNECTION
# =========================================================

def _connect(sql_cfg):
    return sql.connect(
        server_hostname=sql_cfg["server_hostname"],
        http_path=sql_cfg["http_path"],
        access_token=sql_cfg["access_token"],
    )


# =========================================================
# ENSURE REQUIRED TABLES EXIST
# =========================================================

def ensure_tables(sql_cfg) -> None:
    upload_table = sql_cfg["upload_table"]
    feedback_table = sql_cfg["feedback_table"]

    with _connect(sql_cfg) as conn, conn.cursor() as cur:

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {upload_table} (
            invoice_id STRING,

            invoice_amount DOUBLE,
            submission_hour DOUBLE,
            supplier_invoice_count_30d DOUBLE,
            supplier_avg_amount_90d DOUBLE,
            invoice_amount_zscore DOUBLE,
            duplicate_invoice_flag DOUBLE,
            split_invoice_flag DOUBLE,
            late_night_submission_flag DOUBLE,
            supplier_age_days DOUBLE,
            supplier_risk_score DOUBLE,
            blacklisted_flag DOUBLE,
            avg_invoice_amount DOUBLE,
            annual_budget DOUBLE,
            ocr_total_extracted DOUBLE,
            image_tamper_flag DOUBLE,
            amount_to_supplier_90d_avg_ratio DOUBLE,
            amount_to_supplier_avg_ratio DOUBLE,
            amount_diff_supplier_90d_avg DOUBLE,
            supplier_age_years DOUBLE,
            invoice_day_of_week DOUBLE,
            weekend_invoice_flag DOUBLE,
            outside_business_hours_flag DOUBLE,
            invoice_to_department_budget_ratio DOUBLE,
            log_invoice_amount DOUBLE,
            image_metadata_available_flag DOUBLE,
            invoice_ocr_amount_diff DOUBLE,
            invoice_ocr_relative_diff DOUBLE,
            payment_terms_NET30 DOUBLE,
            payment_terms_NET60 DOUBLE,
            payment_terms_NET90 DOUBLE,
            invoice_type_GOODS DOUBLE,
            invoice_type_SERVICES DOUBLE,

            riskcheck_score DOUBLE,
            risk_category STRING,
            priority_rank INT,
            priority_level STRING,
            recommended_action STRING,
            fraud_prediction INT,
            fraud_probability DOUBLE,
            investigation_reasons STRING,

            batch_id STRING,
            source_file STRING,
            scored_at TIMESTAMP

        ) USING DELTA
        """)

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {feedback_table} (
            invoice_id STRING,
            verified_label INT,
            investigator_note STRING,
            verified_at TIMESTAMP,
            consumed_by_training BOOLEAN
        ) USING DELTA
        """)


# =========================================================
# SAVE SCORED INVOICES
# =========================================================

def persist_scored_results(
    scored: pd.DataFrame,
    sql_cfg,
    batch_id: str,
    source_file: str,
    chunk_size: int = 200,
) -> None:
    ensure_tables(sql_cfg)
    table = sql_cfg["upload_table"]

    feature_columns = [
        "invoice_amount",
        "submission_hour",
        "supplier_invoice_count_30d",
        "supplier_avg_amount_90d",
        "invoice_amount_zscore",
        "duplicate_invoice_flag",
        "split_invoice_flag",
        "late_night_submission_flag",
        "supplier_age_days",
        "supplier_risk_score",
        "blacklisted_flag",
        "avg_invoice_amount",
        "annual_budget",
        "ocr_total_extracted",
        "image_tamper_flag",
        "amount_to_supplier_90d_avg_ratio",
        "amount_to_supplier_avg_ratio",
        "amount_diff_supplier_90d_avg",
        "supplier_age_years",
        "invoice_day_of_week",
        "weekend_invoice_flag",
        "outside_business_hours_flag",
        "invoice_to_department_budget_ratio",
        "log_invoice_amount",
        "image_metadata_available_flag",
        "invoice_ocr_amount_diff",
        "invoice_ocr_relative_diff",
        "payment_terms_NET30",
        "payment_terms_NET60",
        "payment_terms_NET90",
        "invoice_type_GOODS",
        "invoice_type_SERVICES",
    ]

    insert_columns = (
        ["invoice_id"]
        + feature_columns
        + [
            "riskcheck_score",
            "risk_category",
            "priority_rank",
            "priority_level",
            "recommended_action",
            "fraud_prediction",
            "fraud_probability",
            "investigation_reasons",
            "batch_id",
            "source_file",
        ]
    )

    rows = []

    for _, row in scored.iterrows():
        values = [str(row["invoice_id"])]

        values.extend(
            float(row[col])
            if pd.notna(row[col])
            else None
            for col in feature_columns
        )

        reasons = row["investigation_reasons"]
        if isinstance(reasons, (list, tuple)):
            reasons_json = json.dumps(reasons)
        elif isinstance(reasons, str):
            reasons_json = reasons
        else:
            reasons_json = json.dumps([str(reasons)])

        values.extend(
            [
                float(row["riskcheck_score"]),
                str(row["risk_category"]),
                int(row["priority_rank"]),
                str(row["priority_level"]),
                str(row["recommended_action"]),
                int(row["fraud_prediction"]),
                float(row["fraud_probability"]),
                reasons_json,
                str(batch_id),
                str(source_file),
            ]
        )

        rows.append(tuple(values))

    if not rows:
        return

    # Databricks SQL has a hard limit of 10,000 parameters per parameterized query.
    # With 43 parameters per row, chunk_size=200 creates ~8,600 parameters per statement.
    connect_start = time.perf_counter()

    with _connect(sql_cfg) as conn:
        connect_seconds = (
            time.perf_counter()
            - connect_start
        )

        insert_start = time.perf_counter()

        with conn.cursor() as cur:
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i : i + chunk_size]
                row_placeholder = (
                    "("
                    + ",".join(["?"] * len(insert_columns))
                    + ", current_timestamp())"
                )
                values_clause = ",".join([row_placeholder] * len(chunk))
                sql_statement = f"""
                    INSERT INTO {table}
                    ({",".join(insert_columns)}, scored_at)
                    VALUES {values_clause}
                """
                parameters = [
                    value
                    for row_values in chunk
                    for value in row_values
                ]
                cur.execute(
                    sql_statement,
                    parameters,
                )

        insert_seconds = (
            time.perf_counter()
            - insert_start
        )

    print(
        f"Databricks SQL connection: "
        f"{connect_seconds:.2f} seconds"
    )

    print(
        f"Databricks INSERT ({len(rows)} rows): "
        f"{insert_seconds:.2f} seconds"
    )

# =========================================================
# LOAD ONE SCORED BATCH
# Used when reviewer opens link from email
# =========================================================

def load_scored_batch(
    batch_id: str,
    sql_cfg,
) -> pd.DataFrame:

    table = sql_cfg["upload_table"]

    with _connect(sql_cfg) as conn, conn.cursor() as cur:

        cur.execute(
            f"""
            SELECT *
            FROM {table}
            WHERE batch_id = ?
            ORDER BY riskcheck_score DESC
            """,
            (batch_id,),
        )

        columns = [
            desc[0]
            for desc in cur.description
        ]

        rows = cur.fetchall()

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# =========================================================
# SAVE INVESTIGATOR FEEDBACK
# =========================================================

def save_feedback(
    invoice_id: str,
    verified_label: int,
    note: str,
    sql_cfg,
) -> None:

    table = sql_cfg["feedback_table"]

    with _connect(sql_cfg) as conn, conn.cursor() as cur:

        cur.execute(
            f"""
            INSERT INTO {table}
            (
                invoice_id,
                verified_label,
                investigator_note,
                verified_at,
                consumed_by_training
            )
            VALUES (
                ?, ?, ?, current_timestamp(), false
            )
            """,
            (
                invoice_id,
                int(verified_label),
                note,
            ),
        )


# =========================================================
# COUNT NEW VERIFIED LABELS
# =========================================================

def count_unconsumed_labels(
    sql_cfg,
) -> int:

    table = sql_cfg["feedback_table"]

    with _connect(sql_cfg) as conn, conn.cursor() as cur:

        cur.execute(
            f"""
            SELECT count(*)
            FROM {table}
            WHERE consumed_by_training = false
            """
        )

        return int(
            cur.fetchone()[0]
        )


# =========================================================
# TRIGGER DATABRICKS RETRAINING JOB
# =========================================================

def trigger_retraining_job(
    db_cfg,
    retraining_cfg=None,
) -> int:

    url = (
        f"{db_cfg['host'].rstrip('/')}"
        "/api/2.2/jobs/run-now"
    )

    token = str(
        uuid.uuid4()
    )

    job_id = None
    if retraining_cfg and "retraining_job_id" in retraining_cfg:
        job_id = retraining_cfg["retraining_job_id"]
    elif "retraining_job_id" in db_cfg:
        job_id = db_cfg["retraining_job_id"]
    else:
        raise ValueError("Missing 'retraining_job_id' in retraining or databricks configuration")

    response = requests.post(
        url,
        headers={
            "Authorization":
                f"Bearer {db_cfg['token']}",
            "Content-Type":
                "application/json",
        },
        json={
            "job_id":
                int(job_id),
            "idempotency_token":
                token,
            "job_parameters": {
                "trigger":
                    "riskcheck_streamlit_feedback_threshold"
            },
        },
        timeout=60,
    )

    response.raise_for_status()

    return int(
        response.json()["run_id"]
    )