# RiskCheck AI - Streamlit Application

This starter app is based on the final RiskCheck AI Databricks notebook.

## What it does

1. Accepts CSV or Parquet invoice batches.
2. Validates `invoice_id` + the exact 32 RiskCheck AI model features.
3. Sends the 32 features to a Databricks Model Serving endpoint.
4. Converts fraud probability to the notebook's 0-100 RiskCheck Score.
5. Uses the notebook's five risk bands and Priority 1-5 mapping.
6. Shows a dashboard and priority queue.
7. Emails Priority 1 and Priority 2 alerts.
8. Captures investigator-verified Fraud / Legitimate outcomes in a Delta table.
9. Triggers a Databricks retraining Job when enough new verified labels accumulate.

## Important: "auto-training" must use verified labels

Do **not** retrain on uploaded invoices merely because the model predicted them as fraud/legitimate. That creates a self-reinforcing feedback loop. Uploaded invoices are scored immediately, but they become training data only after a trusted human/process records the true outcome.

Recommended retraining flow:

`new verified labels -> retraining job -> train candidate -> evaluate on untouched validation/test window -> compare against champion -> register candidate -> promote only if gates pass -> serving endpoint update`

Suggested promotion gates include minimum ROC-AUC, recall, precision, false-positive-rate, and Priority 1/2 fraud-capture metrics.

## Step 1 - Package/register your Databricks model

Your notebook currently creates:

- `assembler` (VectorAssembler)
- `rf_model` (RandomForestClassificationModel)
- `model_feature_columns` (32 inputs)

Before the Streamlit app can call it, package the assembler and Random Forest together and register the pipeline with MLflow / Unity Catalog. A simple Databricks notebook pattern is:

```python
from pyspark.ml import PipelineModel
import mlflow
import mlflow.spark

riskcheck_pipeline = PipelineModel(stages=[assembler, rf_model])

with mlflow.start_run() as run:
    mlflow.spark.log_model(
        spark_model=riskcheck_pipeline,
        artifact_path="riskcheck_model",
        registered_model_name="main.riskcheck.riskcheck_ai"
    )
```

For production, wrap the model so the served endpoint returns `fraud_probability` rather than only a class prediction. The Streamlit file `riskcheck/scoring.py` intentionally isolates response parsing in `_extract_probabilities()` so you only need to adapt one function if your endpoint response differs.

## Step 2 - Create a Databricks Model Serving endpoint

Create an endpoint for the registered model and put its name in `.streamlit/secrets.toml` as `serving_endpoint`.

## Step 3 - Create the feedback/retraining notebook

Create a Databricks notebook/job that:

1. Reads `main.riskcheck.investigator_feedback` where `consumed_by_training = false`.
2. Joins those invoice IDs back to the 32 production features.
3. Appends verified labels to your controlled training dataset.
4. Retrains the same feature pipeline / Random Forest (or selected candidate models).
5. Evaluates against a fixed or time-based holdout set.
6. Registers a new model version only if validation gates pass.
7. Updates the serving endpoint to the approved version.
8. Marks feedback rows as consumed only after successful training.

Then create a Databricks Job for that notebook and place its Job ID in the secrets file.

## Step 4 - Configure Streamlit secrets

Copy:

`.streamlit/secrets.toml.example` -> `.streamlit/secrets.toml`

Fill in Databricks workspace credentials, SQL Warehouse details, model endpoint, retraining Job ID, and SMTP credentials.

Never hard-code tokens/passwords in `app.py`.

## Step 5 - Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Email policy used by this starter

- Priority 1: immediate fraud-team email
- Priority 2: immediate AP-review email
- Priority 3-5: dashboard only

You can change this in `riskcheck/emailer.py`.

## Production additions still recommended

- SSO / role-based access
- tenant/client isolation
- encrypted secrets
- audit log of scoring and user actions
- idempotent file processing
- schema/version checks
- malware scanning for uploads
- PII controls and retention policies
- model/data drift monitoring
- approval gate before model promotion
- email deduplication / cooldowns
- retry queue for failed email and Databricks calls
