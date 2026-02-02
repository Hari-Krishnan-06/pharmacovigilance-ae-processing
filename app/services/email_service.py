import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SAFETY_OFFICER_EMAIL = os.getenv("SAFETY_OFFICER_EMAIL")


def send_escalation_email(
    drug: str,
    adverse_event: str,
    risk_level: str,
    report_id: str,
    explanation: str
):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SAFETY_OFFICER_EMAIL]):
        print("⚠️ Email not configured. Skipping email.")
        return

    subject = f"[{risk_level}] Pharmacovigilance Alert - {drug}"

    body = f"""
PHARMACOVIGILANCE ALERT

Risk Level: {risk_level}
Report ID: {report_id}

Drug:
{drug}

Adverse Event:
{adverse_event}

Explanation:
{explanation}

Action Required:
Immediate safety review.
"""

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = SAFETY_OFFICER_EMAIL
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"📧 Escalation email sent for report {report_id}")

    except Exception as e:
        print(f" Failed to send escalation email: {e}")
