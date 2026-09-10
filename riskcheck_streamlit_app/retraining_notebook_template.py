# Databricks notebook source
# RiskCheck AI - retraining job template
# This is intentionally a template: wire it to your production feature table and model registry.

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

FEEDBACK_TABLE = "workspace.default.riskcheck_investigator_feedback"
FEATURE_TABLE = "workspace.default.riskcheck_scored_invoices"  # contains invoice_id + the 32 model features

FEATURES = [
    "invoice_amount","submission_hour","supplier_invoice_count_30d","supplier_avg_amount_90d",
    "invoice_amount_zscore","duplicate_invoice_flag","split_invoice_flag","late_night_submission_flag",
    "supplier_age_days","supplier_risk_score","blacklisted_flag","avg_invoice_amount","annual_budget",
    "ocr_total_extracted","image_tamper_flag","amount_to_supplier_90d_avg_ratio","amount_to_supplier_avg_ratio",
    "amount_diff_supplier_90d_avg","supplier_age_years","invoice_day_of_week","weekend_invoice_flag",
    "outside_business_hours_flag","invoice_to_department_budget_ratio","log_invoice_amount",
    "image_metadata_available_flag","invoice_ocr_amount_diff","invoice_ocr_relative_diff",
    "payment_terms_NET30","payment_terms_NET60","payment_terms_NET90","invoice_type_GOODS","invoice_type_SERVICES"
]

feedback = spark.table(FEEDBACK_TABLE).filter(F.col("consumed_by_training") == F.lit(False))
features = spark.table(FEATURE_TABLE)
new_labeled = features.join(feedback.select("invoice_id", F.col("verified_label").alias("is_fraud")), "invoice_id")

if new_labeled.count() == 0:
    dbutils.notebook.exit("No new verified labels")

# TODO: append new_labeled into a versioned controlled training table.
# TODO: create time-aware train/validation/test splits to avoid leakage.

assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="error")
ml_df = assembler.transform(new_labeled).select(F.col("is_fraud").cast("double").alias("label"), "features")
train_df, val_df = ml_df.randomSplit([0.8, 0.2], seed=42)

candidate = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42).fit(train_df)
preds = candidate.transform(val_df)
auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(preds)
print("Candidate ROC-AUC:", auc)

# IMPORTANT production TODOs:
# 1. Compare candidate against current champion on the SAME controlled holdout dataset.
# 2. Add recall/precision/FPR and P1/P2 capture gates, not just ROC-AUC.
# 3. Register/promote only when gates pass.
# 4. Update serving endpoint only after approval/promotion.
# 5. Mark feedback rows consumed only after successful promotion.
