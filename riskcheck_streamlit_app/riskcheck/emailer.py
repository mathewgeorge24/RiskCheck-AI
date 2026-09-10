from __future__ import annotations

import html
import smtplib
from email.message import EmailMessage

import pandas as pd


def _send(
    cfg,
    recipients: list[str],
    subject: str,
    text_body: str,
    html_body: str,
) -> None:
    if not recipients:
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_address"]
    msg["To"] = ", ".join(recipients)

    # Plain-text fallback
    msg.set_content(text_body)

    # Styled HTML version
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(
        cfg["smtp_host"],
        int(cfg.get("smtp_port", 587))
    ) as server:
        server.starttls()
        server.login(
            cfg["smtp_username"],
            cfg["smtp_password"]
        )
        server.send_message(msg)


def _invoice_card(row) -> str:
    invoice_id = html.escape(str(row["invoice_id"]))
    score = float(row["riskcheck_score"])
    risk_category = html.escape(str(row["risk_category"]))
    action = html.escape(str(row["recommended_action"]))

    reasons = row.get("investigation_reasons", "")
    if isinstance(reasons, str) and reasons.strip().startswith(("[", "{")):
        try:
            import json
            parsed = json.loads(reasons)
            if isinstance(parsed, (list, tuple)):
                reasons = parsed
        except Exception:
            pass

    if isinstance(reasons, (list, tuple)):
        reasons_text = ", ".join(str(reason) for reason in reasons)
    else:
        reasons_text = str(reasons)

    reasons_text = html.escape(reasons_text)

    return f"""
    <div style="
        background:#f8fafc;
        border:1px solid #e2e8f0;
        border-radius:10px;
        padding:16px;
        margin-bottom:12px;
    ">
        <div style="
            font-size:16px;
            font-weight:700;
            color:#0f172a;
            margin-bottom:10px;
        ">
            Invoice {invoice_id}
        </div>

        <table style="
            width:100%;
            border-collapse:collapse;
            font-size:14px;
        ">
            <tr>
                <td style="padding:4px 0;color:#64748b;">
                    RiskCheck Score
                </td>
                <td style="
                    padding:4px 0;
                    font-weight:700;
                    text-align:right;
                    color:#0f172a;
                ">
                    {score:.2f}
                </td>
            </tr>

            <tr>
                <td style="padding:4px 0;color:#64748b;">
                    Risk Category
                </td>
                <td style="
                    padding:4px 0;
                    font-weight:600;
                    text-align:right;
                    color:#0f172a;
                ">
                    {risk_category}
                </td>
            </tr>
        </table>

        <div style="margin-top:12px;">
            <strong style="color:#334155;">
                Why flagged
            </strong>
            <div style="
                margin-top:4px;
                color:#475569;
                line-height:1.5;
            ">
                {reasons_text}
            </div>
        </div>

        <div style="margin-top:10px;">
            <strong style="color:#334155;">
                Recommended action
            </strong>
            <div style="
                margin-top:4px;
                color:#475569;
                line-height:1.5;
            ">
                {action}
            </div>
        </div>
    </div>
    """


def send_priority_alerts(
    scored: pd.DataFrame,
    cfg,
    batch_name: str = "Invoice batch",
    batch_id: str | None = None,
) -> list[str]:

    if not bool(cfg.get("enabled", False)):
        return []

    # Only P1 and P2 require email notification
    p1 = scored[pd.to_numeric(scored["priority_rank"], errors="coerce") == 1]
    p2 = scored[pd.to_numeric(scored["priority_rank"], errors="coerce") == 2]

    if p1.empty and p2.empty:
        return []

    def _to_list(val):
        if not val:
            return []
        if isinstance(val, str):
            return [val]
        return list(val)

    # Combine recipients so only ONE email is sent for this batch
    recipients = list(
        dict.fromkeys(
            _to_list(cfg.get("priority_1_recipients", []))
            + _to_list(cfg.get("priority_2_recipients", []))
        )
    )

    if not recipients:
        return []

    batch_safe = html.escape(str(batch_name))

    sections = []

    if not p1.empty:
        cards = "".join(
            _invoice_card(row)
            for _, row in p1.head(50).iterrows()
        )

        sections.append(f"""
        <div style="margin-top:24px;">
            <h2 style="
                color:#dc2626;
                font-size:18px;
                margin-bottom:12px;
            ">
                🚨 Immediate Review Required — P1 ({len(p1)})
            </h2>
            {cards}
        </div>
        """)

    if not p2.empty:
        cards = "".join(
            _invoice_card(row)
            for _, row in p2.head(50).iterrows()
        )

        sections.append(f"""
        <div style="margin-top:24px;">
            <h2 style="
                color:#ea580c;
                font-size:18px;
                margin-bottom:12px;
            ">
                ⚠️ High Priority Review — P2 ({len(p2)})
            </h2>
            {cards}
        </div>
        """)

    app_url = str(cfg.get("app_url", "")).strip()

    if app_url and batch_id:
        separator = "&" if "?" in app_url else "?"
        app_url = f"{app_url}{separator}review_batch={batch_id}"

    button_html = ""

    if app_url:
        safe_url = html.escape(app_url, quote=True)

        button_html = f"""
        <div style="
            text-align:center;
            margin:30px 0 20px 0;
        ">
            <a href="{safe_url}"
               style="
                   background:#2563eb;
                   color:#ffffff;
                   text-decoration:none;
                   padding:12px 22px;
                   border-radius:8px;
                   font-weight:700;
                   display:inline-block;
               ">
                Open RiskCheck AI
            </a>
        </div>
        """

    total_alerts = len(p1) + len(p2)

    html_body = f"""
    <html>
    <body style="
        margin:0;
        padding:0;
        background:#f1f5f9;
        font-family:Arial,Helvetica,sans-serif;
    ">

        <div style="
            max-width:720px;
            margin:0 auto;
            padding:24px;
        ">

            <div style="
                background:#0f172a;
                color:#ffffff;
                padding:24px;
                border-radius:12px 12px 0 0;
            ">
                <div style="
                    font-size:24px;
                    font-weight:700;
                ">
                    🛡️ RiskCheck AI
                </div>

                <div style="
                    margin-top:6px;
                    color:#cbd5e1;
                ">
                    Invoice Risk Investigation Alert
                </div>
            </div>

            <div style="
                background:#ffffff;
                padding:24px;
                border-radius:0 0 12px 12px;
            ">

                <h1 style="
                    margin-top:0;
                    color:#0f172a;
                    font-size:21px;
                ">
                    Investigator attention required
                </h1>

                <p style="
                    color:#475569;
                    line-height:1.6;
                ">
                    RiskCheck AI analysed
                    <strong>{batch_safe}</strong>
                    and identified
                    <strong>{total_alerts}</strong>
                    invoice(s) requiring priority review.
                </p>

                <div style="
                    background:#f8fafc;
                    border-radius:8px;
                    padding:14px;
                    margin:18px 0;
                ">
                    <strong>Batch:</strong> {batch_safe}<br>
                    <strong>Total invoices scored:</strong> {len(scored)}<br>
                    <strong>P1 Immediate Review:</strong> {len(p1)}<br>
                    <strong>P2 High Priority:</strong> {len(p2)}
                </div>

                {''.join(sections)}

                {button_html}

                <div style="
                    border-top:1px solid #e2e8f0;
                    margin-top:24px;
                    padding-top:16px;
                    color:#64748b;
                    font-size:12px;
                    line-height:1.5;
                ">
                    RiskCheck AI provides decision-support risk
                    indicators. These alerts are not final fraud
                    determinations. Investigator review is required
                    before any final decision or action.
                </div>

            </div>
        </div>

    </body>
    </html>
    """

    text_body = (
        f"RiskCheck AI - Investigator attention required\n\n"
        f"Batch: {batch_name}\n"
        f"Total invoices scored: {len(scored)}\n"
        f"P1 Immediate Review: {len(p1)}\n"
        f"P2 High Priority: {len(p2)}\n\n"
        f"Please open RiskCheck AI to review the flagged invoices.\n\n"
        f"These are decision-support risk indicators and require "
        f"human review."
    )

    subject = (
        f"RiskCheck AI - {total_alerts} priority invoice(s) "
        f"- {batch_name}"
    )

    _send(
        cfg,
        recipients,
        subject,
        text_body,
        html_body,
    )

    return [
        f"P1: {len(p1)} invoice(s)",
        f"P2: {len(p2)} invoice(s)",
    ]