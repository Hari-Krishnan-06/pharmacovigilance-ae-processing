"""
ML Classifier Service
Loads trained model and provides prediction interface.
"""
import os
import joblib
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from app.services.preprocessor import combine_features, preprocess_text


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "ml" / "vectorizer.pkl"

# Global model and vectorizer (loaded once)
_model = None
_vectorizer = None


# =========================
# RULE-BASED SAFETY LAYER
# =========================

SERIOUS_KEYWORDS = [
    # Regulatory criteria
    "hospital", "hospitalized", "hospitalised", "admitted", "admission",
    "icu", "intensive care", "critical care", "ccu",
    "death", "died", "fatal", "expired", "mortality",

    # Life-threatening / emergency
    "life threatening", "life-threatening",
    "cardiac arrest", "respiratory arrest",
    "shock", "septic shock", "anaphylaxis",

    # Major bleeding
    "gastrointestinal bleeding", "gi bleeding", "gi bleed",
    "melena", "black tarry stools",
    "hematemesis", "vomiting blood",
    "hemorrhage", "haemorrhage", "bleeding requiring transfusion",
    "blood transfusion", "transfusion",
    "hematuria", "blood in urine",
    "hemoptysis", "coughing up blood",
    "rectal bleeding",

    # Neurologic / brain
    "intracranial hemorrhage", "intracranial haemorrhage",
    "brain bleed", "cerebral hemorrhage",
    "stroke", "cva", "subarachnoid hemorrhage",

    # Cardiac
    "myocardial infarction", "heart attack",
    "ventricular fibrillation", "ventricular tachycardia",
    "cardiac arrhythmia", "asystole",

    # Organ failure
    "acute renal failure", "kidney failure",
    "acute liver failure", "hepatic failure",
    "respiratory failure", "ventilator", "intubated",

    # Severe allergic / dermatologic
    "anaphylactic shock", "stevens-johnson syndrome", "sjs",
    "toxic epidermal necrolysis", "ten",

    # Pregnancy / congenital
    "congenital anomaly", "birth defect", "teratogenic",

    # Disability / permanent damage
    "permanent disability", "paralysis", "blindness", "coma",

    # Surgical / emergency intervention
    "emergency surgery", "urgent surgery",
    "intubation", "mechanical ventilation",

    # High-risk labs / conditions
    "severe anemia", "hemoglobin drop",
    "coagulopathy", "overdose", "toxicity"
]

SERIOUS_THRESHOLD = 0.30  # Lower threshold to favor recall


def rule_based_serious_override(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in SERIOUS_KEYWORDS)


class ClassifierNotLoadedError(Exception):
    """Raised when classifier is used before loading."""
    pass


def load_model() -> bool:
    """
    Load the trained model and vectorizer from disk.
    
    Returns:
        True if loaded successfully, False otherwise
    """
    global _model, _vectorizer
    
    try:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            print(f"Model files not found at {MODEL_PATH}")
            return False
        
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and vectorizer loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return _model is not None and _vectorizer is not None


def predict_single(text: str) -> Dict:
    """
    Predict seriousness for a single text input.
    
    Args:
        text: Preprocessed text (drugname + adverse_event combined)
        
    Returns:
        Dict with prediction, probability, and confidence
    """
    if not is_model_loaded():
        if not load_model():
            raise ClassifierNotLoadedError("Classifier model not loaded. Run train_classifier.py first.")
    
    # -------------------------
    # RULE-BASED OVERRIDE FIRST
    # -------------------------
    if rule_based_serious_override(text):
        return {
            "prediction": "Serious",
            "serious_probability": 1.0,
            "non_serious_probability": 0.0,
            "confidence": 1.0,
            "reason": "Rule-based override: high-risk keyword detected"
        }
    
    # Vectorize
    X = _vectorizer.transform([text])
    
    # Predict probabilities
    probabilities = _model.predict_proba(X)[0]
    classes = _model.classes_
    serious_idx = list(classes).index('Serious') if 'Serious' in classes else 1
    serious_prob = probabilities[serious_idx]
    
    # -------------------------
    # THRESHOLD-BASED DECISION
    # -------------------------
    if serious_prob >= SERIOUS_THRESHOLD:
        prediction = "Serious"
    else:
        prediction = "Non-Serious"
    
    return {
        "prediction": prediction,
        "serious_probability": float(serious_prob),
        "non_serious_probability": float(1 - serious_prob),
        "confidence": float(max(probabilities)),
        "reason": "ML classifier"
    }


def predict_from_features(drugname: str, adverse_event: str) -> Dict:
    """
    Predict seriousness from raw drug name and adverse event.
    
    Args:
        drugname: Name of the drug
        adverse_event: Description of the adverse event
        
    Returns:
        Dict with prediction details
    """
    combined_text = combine_features(drugname, adverse_event)
    return predict_single(combined_text)


def predict_batch(texts: List[str]) -> List[Dict]:
    """
    Predict seriousness for multiple texts.
    
    Args:
        texts: List of preprocessed texts
        
    Returns:
        List of prediction dicts
    """
    if not is_model_loaded():
        if not load_model():
            raise ClassifierNotLoadedError("Classifier model not loaded.")
    
    results = []
    
    # Handle rule-based overrides individually
    for text in texts:
        if rule_based_serious_override(text):
            results.append({
                "prediction": "Serious",
                "serious_probability": 1.0,
                "non_serious_probability": 0.0,
                "confidence": 1.0,
                "reason": "Rule-based override: high-risk keyword detected"
            })
        else:
            results.append(None)  # placeholder
    
    # Vectorize only non-overridden
    idx_map = [i for i, r in enumerate(results) if r is None]
    if idx_map:
        filtered_texts = [texts[i] for i in idx_map]
        X = _vectorizer.transform(filtered_texts)
        probabilities = _model.predict_proba(X)
        
        classes = _model.classes_
        serious_idx = list(classes).index('Serious') if 'Serious' in classes else 1
        
        for j, probs in enumerate(probabilities):
            serious_prob = probs[serious_idx]
            if serious_prob >= SERIOUS_THRESHOLD:
                prediction = "Serious"
            else:
                prediction = "Non-Serious"
            
            results[idx_map[j]] = {
                "prediction": prediction,
                "serious_probability": float(serious_prob),
                "non_serious_probability": float(1 - serious_prob),
                "confidence": float(max(probs)),
                "reason": "ML classifier"
            }
    
    return results


def get_model_info() -> Dict:
    """Get information about the loaded model."""
    if not is_model_loaded():
        return {"loaded": False}
    
    return {
        "loaded": True,
        "model_type": type(_model).__name__,
        "classes": list(_model.classes_),
        "n_features": _vectorizer.max_features if hasattr(_vectorizer, 'max_features') else "unknown",
        "serious_threshold": SERIOUS_THRESHOLD,
        "rule_based_keywords": SERIOUS_KEYWORDS
    }
