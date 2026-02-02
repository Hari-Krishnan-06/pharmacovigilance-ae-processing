"""
Rule-Based Escalation Engine
Combines ML predictions with regulatory rules for escalation decisions.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


# Regulatory seriousness keywords (ICH E2A guidelines)
SERIOUS_KEYWORDS = {
    "death": 10,
    "died": 10,
    "fatal": 10,
    "life-threatening": 9,
    "life threatening": 9,
    "hospitalization": 8,
    "hospitalisation": 8,
    "hospitalized": 8,
    "hospitalised": 8,
    "inpatient": 7,
    "disability": 8,
    "incapacity": 8,
    "congenital anomaly": 8,
    "birth defect": 8,
    "cancer": 7,
    "overdose": 7,
    "suicide": 9,
    "suicidal": 8,
    "cardiac arrest": 9,
    "respiratory failure": 9,
    "coma": 9,
    "seizure": 7,
    "anaphylaxis": 8,
    "anaphylactic": 8,
    "stroke": 8,
    "myocardial infarction": 8,
    "heart attack": 8,
    "renal failure": 8,
    "liver failure": 8,
    "hepatic failure": 8,
    "sepsis": 8,
    "shock": 7,
    "transplant": 7,
    "surgery required": 7,
    "surgical intervention": 7,
    "icu": 8,
    "intensive care": 8,
    "ventilator": 8,
    "intubation": 8,
    "resuscitation": 9,
}

# Threshold for escalation
ESCALATION_THRESHOLD = 0.6  # ML probability threshold
KEYWORD_WEIGHT = 0.3  # Weight for keyword-based scoring


@dataclass
class EscalationResult:
    """Result of escalation decision."""
    should_escalate: bool
    final_score: float
    ml_probability: float
    keyword_score: float
    triggered_keywords: List[str]
    explanation: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL


def calculate_keyword_score(text: str) -> tuple:
    """
    Calculate seriousness score based on regulatory keywords.
    
    Returns (score, triggered_keywords)
    """
    text_lower = text.lower()
    triggered = []
    max_score = 0
    
    for keyword, score in SERIOUS_KEYWORDS.items():
        if keyword in text_lower:
            triggered.append(keyword)
            max_score = max(max_score, score)
    
    # Normalize to 0-1 range
    normalized_score = max_score / 10.0 if max_score > 0 else 0
    
    return normalized_score, triggered


def determine_risk_level(score: float) -> str:
    """Determine risk level from combined score."""
    if score >= 0.85:
        return "CRITICAL"
    elif score >= 0.7:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def generate_explanation(result: EscalationResult) -> str:
    """Generate human-readable explanation for the escalation decision."""
    parts = []
    
    if result.should_escalate:
        parts.append(f"⚠️ ESCALATION REQUIRED - Risk Level: {result.risk_level}")
    else:
        parts.append(f"✓ No escalation needed - Risk Level: {result.risk_level}")
    
    parts.append(f"\nReasoning:")
    parts.append(f"• ML Classification Probability (Serious): {result.ml_probability:.1%}")
    
    if result.triggered_keywords:
        parts.append(f"• Regulatory Keywords Detected: {', '.join(result.triggered_keywords)}")
        parts.append(f"• Keyword Severity Score: {result.keyword_score:.1%}")
    else:
        parts.append("• No regulatory keywords detected")
    
    parts.append(f"\nFinal Combined Score: {result.final_score:.1%}")
    
    if result.should_escalate:
        if result.triggered_keywords:
            parts.append(f"\n📋 Action: Report contains serious safety signal(s). Immediate review required.")
        else:
            parts.append(f"\n📋 Action: ML model indicates elevated seriousness probability. Manual review recommended.")
    
    return "\n".join(parts)


def evaluate_escalation(
    ml_prediction: Dict,
    drugname: str,
    adverse_event: str,
    entities: Optional[Dict] = None
) -> EscalationResult:
    """
    Evaluate whether a report should be escalated.
    
    Combines:
    1. ML model probability
    2. Regulatory keyword detection
    3. Entity severity (if available)
    
    Args:
        ml_prediction: Dict from classifier with 'serious_probability'
        drugname: Drug name
        adverse_event: Adverse event description
        entities: Optional extracted entities
        
    Returns:
        EscalationResult with decision and explanation
    """
    # Get ML probability
    ml_prob = ml_prediction.get("serious_probability", 0.5)
    
    # Calculate keyword score
    combined_text = f"{drugname} {adverse_event}"
    if entities:
        symptoms = entities.get("symptoms", [])
        if isinstance(symptoms, list):
            combined_text += " " + " ".join(symptoms)
    
    keyword_score, triggered_keywords = calculate_keyword_score(combined_text)
    
    # Calculate final score (weighted combination)
    # Keywords get override power if highly serious terms are found
    if keyword_score >= 0.8:
        # Critical keywords override ML
        final_score = max(ml_prob, keyword_score)
    else:
        # Weighted combination
        final_score = (1 - KEYWORD_WEIGHT) * ml_prob + KEYWORD_WEIGHT * keyword_score
    
    # Make escalation decision
    should_escalate = final_score >= ESCALATION_THRESHOLD or keyword_score >= 0.8
    
    # Determine risk level
    risk_level = determine_risk_level(final_score)
    
    # Build result
    result = EscalationResult(
        should_escalate=should_escalate,
        final_score=final_score,
        ml_probability=ml_prob,
        keyword_score=keyword_score,
        triggered_keywords=triggered_keywords,
        explanation="",  # Will be filled below
        risk_level=risk_level
    )
    
    # Generate explanation
    result.explanation = generate_explanation(result)
    
    return result


def get_serious_keywords() -> Dict[str, int]:
    """Return the list of serious keywords and their scores."""
    return SERIOUS_KEYWORDS.copy()
