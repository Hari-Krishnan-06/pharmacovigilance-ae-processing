"""
LLM-based Entity Extractor
Uses Ollama with Qwen 2.5 to extract drug names and symptoms from text.
Falls back to regex extraction if LLM is unavailable.
"""
import re
import json
from typing import Dict, List, Optional
import httpx


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:latest"
TIMEOUT = 30  # seconds

# Common drug name patterns for fallback
DRUG_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:mg|ml|mcg|tablet|capsule|injection)',
    r'\b([A-Z]{2,}(?:\s+[A-Z]+)*)\b',  # All caps words
]

# Common adverse event terms for fallback
SYMPTOM_KEYWORDS = [
    'pain', 'headache', 'nausea', 'vomiting', 'dizziness', 'fatigue', 'rash',
    'fever', 'seizure', 'bleeding', 'swelling', 'death', 'hospitalization',
    'disability', 'coma', 'cardiac arrest', 'respiratory failure', 'anaphylaxis',
    'hypotension', 'hypertension', 'tachycardia', 'bradycardia', 'dyspnea',
    'edema', 'infection', 'sepsis', 'stroke', 'infarction', 'thrombosis'
]


def extract_with_llm(text: str) -> Optional[Dict]:
    """
    Extract entities using Ollama Qwen 2.5.
    
    Returns dict with 'drug' and 'symptoms' keys, or None if failed.
    """
    prompt = f"""Extract the drug name and adverse event symptoms from this medical report.
Return ONLY a JSON object in this exact format, with no other text:
{{"drug": "drug name here", "symptoms": ["symptom1", "symptom2"]}}

Medical Report:
{text}

JSON:"""

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200
                }
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            
            # Parse JSON from response
            try:
                # Find JSON in response
                json_match = re.search(r'\{[^}]+\}', generated_text)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
                
    except (httpx.RequestError, httpx.TimeoutException) as e:
        print(f"LLM request failed: {e}")
    
    return None


def extract_with_regex(drugname: str, adverse_event: str) -> Dict:
    """
    Fallback entity extraction using regex patterns.
    """
    # For drug, use the provided drugname
    drug = drugname.strip() if drugname else "Unknown"
    
    # For symptoms, match known keywords
    symptoms = []
    if adverse_event:
        text_lower = adverse_event.lower()
        for keyword in SYMPTOM_KEYWORDS:
            if keyword in text_lower:
                symptoms.append(keyword.title())
    
    # If no matches, use the adverse event as is
    if not symptoms and adverse_event:
        symptoms = [adverse_event.strip()]
    
    return {
        "drug": drug,
        "symptoms": symptoms,
        "extraction_method": "regex"
    }


def extract_entities(drugname: str, adverse_event: str) -> Dict:
    """
    Extract entities from drug name and adverse event text.
    Tries LLM first, falls back to regex.
    
    Args:
        drugname: Name of the drug
        adverse_event: Adverse event description
        
    Returns:
        Dict with 'drug', 'symptoms', and 'extraction_method'
    """
    combined_text = f"Drug: {drugname}. Adverse Event: {adverse_event}"
    
    # Try LLM first
    llm_result = extract_with_llm(combined_text)
    
    if llm_result and 'drug' in llm_result and 'symptoms' in llm_result:
        return {
            "drug": llm_result['drug'],
            "symptoms": llm_result['symptoms'] if isinstance(llm_result['symptoms'], list) else [llm_result['symptoms']],
            "extraction_method": "llm"
        }
    
    # Fallback to regex
    return extract_with_regex(drugname, adverse_event)


def check_llm_availability() -> bool:
    """Check if Ollama is running and model is available."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any("qwen" in name.lower() for name in model_names)
    except Exception:
        pass
    return False
