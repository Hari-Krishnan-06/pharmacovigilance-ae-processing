import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"



def generate_api_explanation(
    drug: str,
    adverse_event: str,
    classification: str,
    triggered_keywords: list,
    serious_prob: float,
    risk_level: str
) -> str:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set")
        return "LLM explanation not available (API key not set)."

    prompt = f"""
You are a pharmacovigilance safety assistant.

Explain why this adverse event was classified as {classification}.
Do NOT change the classification. Only explain.

Drug: {drug}
Adverse Event: {adverse_event}
Triggered Keywords: {', '.join(triggered_keywords) if triggered_keywords else 'None'}
Risk Level: {risk_level}
ML Serious Probability: {serious_prob:.2f}

Provide a short, clinical explanation (2-3 sentences).
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a pharmacovigilance safety assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 120,
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("===== GROQ BAD REQUEST DEBUG =====")
        print("Exception:", repr(e))
        try:
            print("Status:", resp.status_code)
            print("Response body:", resp.text)
        except:
            print("No response body available")
        print("=================================")
        return "LLM explanation not available due to an internal error."


# ============================================================
# Wrapper to match main.py expected signature (ADAPTER)
# ============================================================

def generate_deterministic_explanation(
    drug: str,
    adverse_event: str,
    ml_result: dict,
    escalation
) -> str:
    """
    Adapter to convert internal ML + escalation objects
    to API explanation format expected by Groq.
    Keeps main.py unchanged.
    """

    classification = ml_result.get("prediction")
    serious_prob = ml_result.get("serious_probability")

    triggered_keywords = getattr(escalation, "triggered_keywords", [])
    risk_level = getattr(escalation, "risk_level", None)

    return generate_api_explanation(
        drug=drug,
        adverse_event=adverse_event,
        classification=classification,
        triggered_keywords=triggered_keywords,
        serious_prob=serious_prob,
        risk_level=risk_level
    )
