from __future__ import annotations

from io import BytesIO
import ast
import uuid
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from riskcheck.features import REQUIRED_COLUMNS
from riskcheck.scoring import (
    score_via_databricks,
    validate_invoice_frame,
)
from riskcheck.emailer import send_priority_alerts
from riskcheck.databricks_io import (
    ensure_tables,
    persist_scored_results,
    load_scored_batch,
    save_feedback,
    count_unconsumed_labels,
    trigger_retraining_job,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RiskCheck AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ RiskCheck AI")
st.caption(
    "Invoice fraud-risk detection and investigation prioritisation"
)


# =========================================================
# SECRETS CHECK
# =========================================================

REQUIRED_SECRET_SECTIONS = ["databricks", "databricks_sql", "retraining", "email"]


def get_missing_secrets() -> list[str]:
    return [s for s in REQUIRED_SECRET_SECTIONS if s not in st.secrets]


missing_secret_sections = get_missing_secrets()
if missing_secret_sections:
    with st.expander("⚠️ Configuration Required: Secrets Not Configured", expanded=True):
        st.warning(
            f"Missing secret section(s): `[{', '.join(missing_secret_sections)}]`.\n\n"
            "If running on **Streamlit Community Cloud**, go to your app dashboard: "
            "**App Settings → Secrets** and paste your configuration.\n\n"
            "If running **locally**, ensure `.streamlit/secrets.toml` exists."
        )
# =========================================================

def read_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded)

    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(
            BytesIO(uploaded.getvalue())
        )

    raise ValueError(
        "Upload a CSV or Parquet file."
    )




def cfg(section):
    if section not in st.secrets:
        raise RuntimeError(
            f"Missing [{section}] in .streamlit/secrets.toml (or Streamlit Cloud Secrets). "
            f"Please configure [{section}] in your secrets."
        )

    return st.secrets[section]


def show_email_status():
    """
    Display email results saved before st.rerun().
    """

    statuses = st.session_state.pop(
        "email_statuses",
        [],
    )

    for status in statuses:

        if status["type"] == "success":
            st.success(status["message"])

        elif status["type"] == "warning":
            st.warning(status["message"])

        elif status["type"] == "info":
            st.info(status["message"])


# =========================================================
# SESSION STATE
# =========================================================

scored = st.session_state.get("scored")

scored_batches = st.session_state.get(
    "scored_batches",
    [],
)

review_batch = st.query_params.get(
    "review_batch"
)


# =========================================================
# EMAIL REVIEW MODE
# =========================================================

if review_batch:

    try:

        loaded_batch = load_scored_batch(
            str(review_batch),
            cfg("databricks_sql"),
        )

        if not loaded_batch.empty:

            scored = loaded_batch

            st.session_state[
                "scored"
            ] = loaded_batch

        else:

            st.warning(
                "No scored invoices were found "
                "for this review batch."
            )

    except Exception as e:

        st.warning(
            f"Could not load review batch: {e}"
        )


# =========================================================
# LANDING PAGE
# =========================================================

if (
    not review_batch
    and (scored is None or scored.empty)
    and not scored_batches
):

    st.html("""
<style>
.hero-box {
    padding: 30px 32px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #111827 0%,
        #172033 100%
    );
    border: 1px solid #283548;
    margin-top: 18px;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero-text {
    font-size: 15px;
    color: #aeb8c7;
    max-width: 750px;
    line-height: 1.6;
}

.feature-card {
    background: #161b22;
    border: 1px solid #2a313b;
    border-radius: 14px;
    padding: 20px;
    min-height: 145px;
    box-sizing: border-box;
}

.feature-icon {
    font-size: 24px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 7px;
}

.feature-text {
    font-size: 13px;
    color: #9aa4b2;
    line-height: 1.5;
}
</style>

<div class="hero-box">
    <div class="hero-title">
        Smarter Invoice Fraud Risk Detection
    </div>

    <div class="hero-text">
        RiskCheck AI analyses invoice and supplier patterns
        to identify potentially suspicious activity,
        prioritise investigations and help investigators
        focus attention where it matters most.
    </div>
</div>
""")

    f1, f2, f3 = st.columns(3)

    with f1:

        st.html("""
<div class="feature-card">
    <div class="feature-icon">🔍</div>

    <div class="feature-title">
        Detect Risk
    </div>

    <div class="feature-text">
        Analyse invoice, supplier and behavioural indicators
        to identify unusual patterns that may require
        investigation.
    </div>
</div>
""")

    with f2:

        st.html("""
<div class="feature-card">
    <div class="feature-icon">⚡</div>

    <div class="feature-title">
        Prioritise Investigations
    </div>

    <div class="feature-text">
        RiskCheck Scores and P1-P5 priorities help
        investigators focus first on invoices with the
        strongest risk indicators.
    </div>
</div>
""")

    with f3:

        st.html("""
<div class="feature-card">
    <div class="feature-icon">✓</div>

    <div class="feature-title">
        Human-Verified Learning
    </div>

    <div class="feature-text">
        Investigator-confirmed outcomes provide verified
        feedback for controlled future model improvement.
    </div>
</div>
""")

    st.write("")

    st.markdown(
        "### Analyse Invoice Batches"
    )

    st.caption(
        "Upload one or more CSV or Parquet files "
        "to begin fraud-risk analysis."
    )


# =========================================================
# EMAIL STATUS FROM PREVIOUS SCORING RUN
# =========================================================

if not review_batch:
    show_email_status()


# =========================================================
# MULTI-FILE UPLOAD
# =========================================================

if (
    not review_batch
    and not scored_batches
):

    uploads = st.file_uploader(
        "Upload invoice batches",
        type=[
            "csv",
            "parquet",
            "pq",
        ],
        accept_multiple_files=True,
        help=(
            "Each file must contain invoice_id plus "
            "the 32 RiskCheck AI model features."
        ),
    )

else:

    uploads = []


# =========================================================
# PREVIEW AND SCORE
# =========================================================

if uploads:

    prepared_batches = []

    for uploaded in uploads:

        try:

            raw = read_upload(
                uploaded
            )

            with st.expander(
                f"Preview: {uploaded.name}",
                expanded=len(uploads) == 1,
            ):

                st.dataframe(
                    raw.head(20),
                    width="stretch",
                )

                st.caption(
                    f"{len(raw):,} invoice(s) detected."
                )

            missing = validate_invoice_frame(
                raw
            )

            if missing:

                st.error(
                    f"{uploaded.name} is missing "
                    "required fields: "
                    + ", ".join(missing)
                )

            else:

                prepared_batches.append(
                    {
                        "file": uploaded,
                        "raw": raw,
                    }
                )

        except Exception as e:

            st.error(
                f"Could not read "
                f"{uploaded.name}: {e}"
            )

    if prepared_batches:

        st.caption(
            f"{len(prepared_batches)} "
            "file(s) ready for scoring."
        )

        if missing_secret_sections:
            st.error(
                f"❌ Cannot score batches: Required secret section(s) `[{', '.join(missing_secret_sections)}]` "
                "are missing. Please configure them in **Streamlit Community Cloud → App Settings → Secrets**."
            )
        elif st.button(
            "Score invoice batches",
            type="primary",
            width="stretch",
        ):

            batch_results = []
            email_statuses = []

            total_batches = len(
                prepared_batches
            )

            progress = st.progress(0)

            progress_text = st.empty()

            for batch_number, batch in enumerate(
                prepared_batches,
                start=1,
            ):

                uploaded = batch["file"]
                raw = batch["raw"]

                source_file = uploaded.name

                batch_id = str(
                    uuid.uuid4()
                )

                progress_text.info(
                    f"Processing batch "
                    f"{batch_number} of "
                    f"{total_batches}: "
                    f"{source_file}"
                )

                try:

                    # -------------------------------------
                    # SCORE
                    # -------------------------------------

                    score_start = time.perf_counter()

                    with st.spinner(
                        f"Scoring {source_file}..."
                    ):
                        scored_batch = (
                            score_via_databricks(
                                raw,
                                cfg("databricks")[
                                    "host"
                                ],
                                cfg("databricks")[
                                    "token"
                                ],
                                cfg("databricks")[
                                    "serving_endpoint"
                                ],
                            )
                        )

                    score_seconds = (
                        time.perf_counter()
                        - score_start
                    )

                    st.write(
                        f"⏱ Model scoring: "
                        f"{score_seconds:.2f} seconds"
                    )


                    # -------------------------------------
                    # PERSIST
                    # -------------------------------------

                    persist_start = time.perf_counter()

                    persist_scored_results(
                        scored_batch,
                        cfg("databricks_sql"),
                        batch_id=batch_id,
                        source_file=source_file,
                    )

                    persist_seconds = (
                        time.perf_counter()
                        - persist_start
                    )

                    st.write(
                        f"⏱ Databricks persistence: "
                        f"{persist_seconds:.2f} seconds"
                    )

                    # -------------------------------------
                    # EMAIL
                    # One consolidated P1/P2 email
                    # for this file
                    # -------------------------------------

                    try:

                        email_start = time.perf_counter()

                        with st.spinner(
                            f"Sending priority "
                            f"alerts for "
                            f"{source_file}..."
                        ):

                            sent = (
                                send_priority_alerts(
                                    scored_batch,
                                    cfg("email"),
                                    batch_name=source_file,
                                    batch_id=batch_id,
                                )
                            )

                        email_seconds = (
                            time.perf_counter()
                            - email_start
                        )

                        st.write(
                            f"⏱ Email: "
                            f"{email_seconds:.2f} seconds"
                        )

                        if sent:

                            email_statuses.append(
                                {
                                    "type": "success",
                                    "message": (
                                        f"{source_file}: "
                                        "priority review "
                                        "email sent — "
                                        + "; ".join(sent)
                                    ),
                                }
                            )

                        else:

                            email_statuses.append(
                                {
                                    "type": "info",
                                    "message": (
                                        f"{source_file}: "
                                        "no P1/P2 invoices "
                                        "required an email "
                                        "alert."
                                    ),
                                }
                            )

                    except Exception as email_error:

                        email_statuses.append(
                            {
                                "type": "warning",
                                "message": (
                                    f"{source_file}: "
                                    "scoring succeeded, "
                                    "but email notification "
                                    "failed: "
                                    f"{email_error}"
                                ),
                            }
                        )

                    # -------------------------------------
                    # KEEP BATCH SEPARATE
                    # -------------------------------------

                    batch_results.append(
                        {
                            "batch_id": batch_id,
                            "source_file": (
                                source_file
                            ),
                            "scored": scored_batch,
                        }
                    )

                except Exception as batch_error:

                    st.error(
                        f"{source_file}: scoring or "
                        "persistence failed: "
                        f"{batch_error}"
                    )

                progress.progress(
                    batch_number
                    / total_batches
                )

            progress_text.empty()
            progress.empty()

            if batch_results:

                st.session_state[
                    "scored_batches"
                ] = batch_results

                st.session_state[
                    "email_statuses"
                ] = email_statuses

                st.session_state.pop(
                    "scored",
                    None,
                )

                st.rerun()


# =========================================================
# KPI STYLE
# =========================================================

st.html("""
<style>
.kpi-card {
    background: #161b22;
    border: 1px solid #2a313b;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 105px;
    box-sizing: border-box;
}

.kpi-label {
    font-size: 13px;
    color: #9aa4b2;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 12px;
    color: #7f8a99;
    margin-top: 7px;
}
</style>
""")


# =========================================================
# DASHBOARD
# =========================================================

def render_batch_dashboard(
    batch_scored: pd.DataFrame,
    batch_name: str,
    batch_key: str,
    review_mode: bool = False,
):

    if (
        batch_scored is None
        or batch_scored.empty
    ):
        return

    scored_view = batch_scored.copy()

    # -----------------------------------------------------
    # REVIEW MODE
    # -----------------------------------------------------

    if review_mode:

        scored_view = scored_view[
            scored_view[
                "priority_rank"
            ].isin([1, 2])
        ].copy()

        st.info(
            "📩 Review Mode — showing only "
            "Priority 1 and Priority 2 invoices "
            "requiring review from this email batch."
        )

    if scored_view.empty:

        st.success(
            "There are no invoices requiring "
            "priority review in this batch."
        )

        return

    # -----------------------------------------------------
    # HEADING
    # -----------------------------------------------------

    if review_mode:

        st.markdown(
            f"## Priority Review — {batch_name}"
        )

    else:

        st.markdown(
            f"## {batch_name}"
        )

        st.caption(
            "Scoring results for this uploaded file."
        )

    # -----------------------------------------------------
    # KPI VALUES
    # -----------------------------------------------------

    total_invoices = len(
        scored_view
    )

    ranks = pd.to_numeric(
        scored_view["priority_rank"],
        errors="coerce",
    )

    critical_count = int(
        (ranks == 1).sum()
    )

    high_count = int(
        (ranks == 2).sum()
    )

    average_score = float(
        pd.to_numeric(
            scored_view["riskcheck_score"],
            errors="coerce",
        ).mean()
    )

    flagged_count = int(
        pd.to_numeric(
            scored_view["fraud_prediction"],
            errors="coerce",
        ).fillna(0).sum()
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    with c1:

        st.html(
            f"""
<div class="kpi-card">
    <div class="kpi-label">
        Invoices
    </div>
    <div class="kpi-value">
        {total_invoices:,}
    </div>
    <div class="kpi-sub">
        Invoices shown
    </div>
</div>
"""
        )

    with c2:

        st.html(
            f"""
<div class="kpi-card">
    <div class="kpi-label">
        Critical
    </div>
    <div class="kpi-value">
        {critical_count}
    </div>
    <div class="kpi-sub">
        Priority 1 review
    </div>
</div>
"""
        )

    with c3:

        st.html(
            f"""
<div class="kpi-card">
    <div class="kpi-label">
        High
    </div>
    <div class="kpi-value">
        {high_count}
    </div>
    <div class="kpi-sub">
        Priority 2 review
    </div>
</div>
"""
        )

    with c4:

        st.html(
            f"""
<div class="kpi-card">
    <div class="kpi-label">
        Average Risk Score
    </div>
    <div class="kpi-value">
        {average_score:.1f}
    </div>
    <div class="kpi-sub">
        Across this batch
    </div>
</div>
"""
        )

    with c5:

        st.html(
            f"""
<div class="kpi-card">
    <div class="kpi-label">
        Flagged for Review
    </div>
    <div class="kpi-value">
        {flagged_count}
    </div>
    <div class="kpi-sub">
        Invoices requiring attention
    </div>
</div>
"""
        )

    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        dist = (
            scored_view[
                "risk_category"
            ]
            .value_counts()
            .rename_axis(
                "risk_category"
            )
            .reset_index(
                name="count"
            )
        )

        fig = px.bar(
            dist,
            x="risk_category",
            y="count",
            color="risk_category",
            title="Risk Distribution",
            labels={
                "risk_category":
                    "Risk Category",
                "count":
                    "Invoices",
            },
            color_discrete_map={
                "Low Risk":
                    "#22c55e",
                "Moderate Risk":
                    "#38bdf8",
                "Elevated Risk":
                    "#eab308",
                "High Risk":
                    "#f97316",
                "Critical Risk":
                    "#ef4444",
            },
        )

        fig.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key=(
                f"risk_distribution_"
                f"{batch_key}"
            ),
        )

    with right:

        # For large files, showing thousands of bars
        # makes the chart slow and unreadable.
        # Show the 50 highest-risk invoices instead.

        ranked_scores = (
            scored_view
            .sort_values(
                "riskcheck_score",
                ascending=False,
            )
            .head(50)
            .sort_values(
                "riskcheck_score",
                ascending=True,
            )
        )

        fig2 = px.bar(
            ranked_scores,
            x="riskcheck_score",
            y="invoice_id",
            orientation="h",
            color="risk_category",
            title=(
                "Highest Invoice Risk Scores"
                if len(scored_view) > 50
                else "Invoice Risk Scores"
            ),
            labels={
                "riskcheck_score":
                    "RiskCheck Score",
                "invoice_id":
                    "Invoice",
                "risk_category":
                    "Risk Category",
            },
            color_discrete_map={
                "Low Risk":
                    "#22c55e",
                "Moderate Risk":
                    "#38bdf8",
                "Elevated Risk":
                    "#eab308",
                "High Risk":
                    "#f97316",
                "Critical Risk":
                    "#ef4444",
            },
        )

        fig2.update_layout(
            yaxis_title=None,
            xaxis_range=[
                0,
                100,
            ],
        )

        st.plotly_chart(
            fig2,
            width="stretch",
            key=(
                f"invoice_scores_"
                f"{batch_key}"
            ),
        )

    # -----------------------------------------------------
    # PRIORITY QUEUE
    # -----------------------------------------------------

    st.subheader(
        "Priority Queue"
    )

    filter1, filter2, filter3 = (
        st.columns(
            [2, 1, 1]
        )
    )

    with filter1:

        search_text = st.text_input(
            "Search invoice",
            placeholder=(
                "Enter invoice ID..."
            ),
            key=f"search_{batch_key}",
        )

    with filter2:

        risk_options = [
            "All"
        ] + sorted(
            scored_view[
                "risk_category"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_risk = st.selectbox(
            "Risk category",
            risk_options,
            key=(
                f"risk_filter_"
                f"{batch_key}"
            ),
        )

    with filter3:

        priority_options = [
            "All"
        ] + sorted(
            scored_view[
                "priority_level"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_priority = st.selectbox(
            "Priority",
            priority_options,
            key=(
                f"priority_filter_"
                f"{batch_key}"
            ),
        )

    filtered_scored = (
        scored_view.copy()
    )

    if search_text:

        filtered_scored = (
            filtered_scored[
                filtered_scored[
                    "invoice_id"
                ]
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    na=False,
                )
            ]
        )

    if selected_risk != "All":

        filtered_scored = (
            filtered_scored[
                filtered_scored[
                    "risk_category"
                ] == selected_risk
            ]
        )

    if selected_priority != "All":

        filtered_scored = (
            filtered_scored[
                filtered_scored[
                    "priority_level"
                ] == selected_priority
            ]
        )

    filtered_scored = (
        filtered_scored
        .sort_values(
            "riskcheck_score",
            ascending=False,
        )
    )

    display_cols = [
        "invoice_id",
        "riskcheck_score",
        "risk_category",
        "priority_level",
        "recommended_action",
        "investigation_reasons",
    ]

    st.dataframe(
        filtered_scored[
            display_cols
        ],
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"Showing {len(filtered_scored):,} "
        f"of {len(scored_view):,} "
        "invoices in this batch."
    )

    # -----------------------------------------------------
    # INVESTIGATION
    # -----------------------------------------------------

    st.markdown(
        "### Invoice Investigation"
    )

    if filtered_scored.empty:

        st.info(
            "No invoices match the "
            "current filters."
        )

    else:

        selected_invoice = (
            st.selectbox(
                "Select invoice to investigate",
                filtered_scored[
                    "invoice_id"
                ]
                .astype(str)
                .tolist(),
                key=(
                    f"investigation_invoice_"
                    f"{batch_key}"
                ),
            )
        )

        invoice_row = (
            filtered_scored[
                filtered_scored[
                    "invoice_id"
                ]
                .astype(str)
                == selected_invoice
            ]
            .iloc[0]
        )

        detail1, detail2, detail3 = (
            st.columns(3)
        )

        with detail1:

            st.metric(
                "RiskCheck Score",
                f"{invoice_row['riskcheck_score']:.1f}",
            )

        with detail2:

            st.metric(
                "Risk Category",
                str(
                    invoice_row[
                        "risk_category"
                    ]
                ),
            )

        with detail3:

            st.metric(
                "Priority",
                str(
                    invoice_row[
                        "priority_level"
                    ]
                ),
            )

        st.markdown(
            "**Why was this invoice flagged?**"
        )

        reasons = invoice_row[
            "investigation_reasons"
        ]

        if isinstance(reasons, str):
            try:
                import json
                parsed = json.loads(reasons)
                if isinstance(parsed, (list, tuple)):
                    reasons = parsed
                else:
                    reasons = [reasons]
            except Exception:
                try:
                    parsed = ast.literal_eval(reasons)
                    if isinstance(parsed, (list, tuple)):
                        reasons = parsed
                    else:
                        reasons = [reasons]
                except (ValueError, SyntaxError):
                    reasons = [reasons]

        if not isinstance(
            reasons,
            (list, tuple),
        ):
            reasons = [
                reasons
            ]

        for reason in reasons:

            st.markdown(
                f"- ⚠️ {reason}"
            )

        st.markdown(
            "**Recommended action**"
        )

        st.write(
            str(
                invoice_row[
                    "recommended_action"
                ]
            )
        )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    safe_download_name = (
        batch_name
        .replace(
            ".csv",
            "",
        )
        .replace(
            ".parquet",
            "",
        )
        .replace(
            ".pq",
            "",
        )
    )

    st.download_button(
        "Download this batch",
        data=(
            scored_view
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        ),
        file_name=(
            f"riskcheck_scored_"
            f"{safe_download_name}.csv"
        ),
        mime="text/csv",
        key=(
            f"download_"
            f"{batch_key}"
        ),
    )

    # -----------------------------------------------------
    # INVESTIGATOR FEEDBACK
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "Investigator Feedback"
    )

    st.caption(
        "RiskCheck AI identifies potential risk and "
        "prioritises invoices for investigation. "
        "The investigator makes the final determination."
    )

    feedback_invoice = st.selectbox(
        "Invoice",
        scored_view[
            "invoice_id"
        ]
        .astype(str)
        .tolist(),
        key=(
            f"feedback_invoice_"
            f"{batch_key}"
        ),
    )

    outcome = st.radio(
        "Verified outcome",
        [
            "Legitimate",
            "Fraud",
        ],
        horizontal=True,
        key=(
            f"feedback_outcome_"
            f"{batch_key}"
        ),
    )

    note = st.text_area(
        "Investigator note",
        placeholder=(
            "Why was this invoice "
            "confirmed or cleared?"
        ),
        key=(
            f"feedback_note_"
            f"{batch_key}"
        ),
    )

    if st.button(
        "Save verified outcome",
        key=(
            f"save_feedback_"
            f"{batch_key}"
        ),
    ):

        try:

            ensure_tables(
                cfg(
                    "databricks_sql"
                )
            )

            label = (
                1
                if outcome == "Fraud"
                else 0
            )

            save_feedback(
                feedback_invoice,
                label,
                note,
                cfg(
                    "databricks_sql"
                ),
            )

            new_labels = (
                count_unconsumed_labels(
                    cfg(
                        "databricks_sql"
                    )
                )
            )

            threshold = int(
                cfg("retraining").get(
                    "min_new_verified_labels",
                    cfg("retraining").get(
                        "minimum_new_verified_labels",
                        200,
                    ),
                )
            )

            st.success(
                f"Feedback saved. "
                f"{new_labels}/{threshold} "
                "new verified labels available "
                "for retraining."
            )

            if new_labels >= threshold:

                run_id = (
                    trigger_retraining_job(
                        cfg("databricks"),
                        cfg("retraining"),
                    )
                )

                st.info(
                    "Retraining job triggered "
                    "in Databricks. "
                    f"Run ID: {run_id}"
                )

        except Exception as e:

            st.error(
                "Could not save feedback / "
                f"trigger retraining: {e}"
            )


# =========================================================
# EMAIL REVIEW PAGE
# =========================================================

if (
    review_batch
    and scored is not None
    and not scored.empty
):

    source_name = (
        "Priority Review Batch"
    )

    if "source_file" in scored.columns:

        source_files = (
            scored[
                "source_file"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if source_files:

            source_name = (
                source_files[0]
            )

    render_batch_dashboard(
        scored,
        source_name,
        str(review_batch),
        review_mode=True,
    )


# =========================================================
# NORMAL MULTI-FILE RESULTS
# =========================================================

elif scored_batches:

    top_left, top_right = (
        st.columns(
            [5, 1]
        )
    )

    with top_left:

        st.subheader(
            "Scored Batch Results"
        )

        st.caption(
            f"{len(scored_batches)} uploaded "
            "file(s) were scored separately."
        )

    with top_right:

        if st.button(
            "New upload",
            width="stretch",
        ):

            st.session_state.pop(
                "scored_batches",
                None,
            )

            st.session_state.pop(
                "scored",
                None,
            )

            st.session_state.pop(
                "email_statuses",
                None,
            )

            st.query_params.clear()

            st.rerun()

    if len(scored_batches) == 1:

        batch = scored_batches[0]

        render_batch_dashboard(
            batch["scored"],
            batch["source_file"],
            batch["batch_id"],
            review_mode=False,
        )

    else:

        tab_names = [
            f"📄 {batch['source_file']}"
            for batch in scored_batches
        ]

        tabs = st.tabs(
            tab_names
        )

        for tab, batch in zip(
            tabs,
            scored_batches,
        ):

            with tab:

                render_batch_dashboard(
                    batch["scored"],
                    batch["source_file"],
                    batch["batch_id"],
                    review_mode=False,
                )


# =========================================================
# LEGACY SINGLE-BATCH SUPPORT
# =========================================================

elif (
    scored is not None
    and not scored.empty
    and not review_batch
):

    render_batch_dashboard(
        scored,
        "Scored Invoice Batch",
        "single_batch",
        review_mode=False,
    )


# =========================================================
# REVIEW MODE EXIT
# =========================================================

if review_batch:

    st.divider()

    if st.button(
        "← Return to upload page"
    ):

        st.session_state.pop(
            "scored",
            None,
        )

        st.session_state.pop(
            "scored_batches",
            None,
        )

        st.query_params.clear()

        st.rerun()


# =========================================================
# EXPECTED INPUT SCHEMA
# =========================================================

with st.expander(
    "Expected upload schema (33 columns)"
):

    st.code(
        "\n".join(
            REQUIRED_COLUMNS
        )
    )