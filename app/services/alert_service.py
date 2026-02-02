"""
Alert Service - None-Safe Implementation
Defensive coding to prevent crashes from None values
"""

from typing import Optional


def trigger_alert(
    report_id: str,
    drugname: Optional[str],
    adverse_event: str,
    risk_level: str,
    explanation: str
):
    """
    Trigger alert for escalated cases.
    
    DEFENSIVE: Handles None drugname gracefully.
    """
    # CRITICAL: Defensive None-safety
    safe_drug = str(drugname) if drugname else "UNKNOWN"
    safe_event = str(adverse_event) if adverse_event else "No description"
    safe_explanation = str(explanation) if explanation else "No explanation"
    
    # Example: Truncate with None-safety
    drug_short = safe_drug[:60] if safe_drug else "UNKNOWN"
    event_short = safe_event[:200] if safe_event else "No description"
    
    # Build alert message
    alert_message = f"""
    🚨 PHARMACOVIGILANCE ALERT 🚨
    
    Report ID: {report_id}
    Drug: {drug_short}
    Risk Level: {risk_level}
    
    Adverse Event:
    {event_short}
    
    Explanation:
    {safe_explanation}
    
    ACTION REQUIRED: Review this case immediately.
    """
    
    # Log alert (replace with your actual alerting mechanism)
    print(f"[ALERT] {risk_level} - {drug_short} - {report_id}")
    print(alert_message)
    
    # Send to actual alerting system (Slack, email, PagerDuty, etc.)
    # Example:
    # send_to_slack(alert_message)
    # send_to_pagerduty(report_id, risk_level)
    
    return {
        "alert_sent": True,
        "report_id": report_id,
        "drug": safe_drug,
        "risk_level": risk_level
    }


def send_to_slack(message: str):
    """
    Send alert to Slack (placeholder)
    """
    # Implement Slack webhook
    pass


def send_to_pagerduty(report_id: str, severity: str):
    """
    Send alert to PagerDuty (placeholder)
    """
    # Implement PagerDuty API
    pass