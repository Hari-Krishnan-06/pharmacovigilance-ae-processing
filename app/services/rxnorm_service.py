import requests
from functools import lru_cache
from typing import Optional
import re

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

TIMEOUT = 5


@lru_cache(maxsize=1000)
def lookup_drug_rxnorm(drugname: str):
    """
    Validate + normalize drug using RxNorm.
    Resilient design:
      1. Try RxNorm (primary)
      2. If blocked/unavailable → try openFDA (secondary)
      3. If still fails → fallback to input drug (DO NOT BLOCK PIPELINE)
    """
    drugname = drugname.strip()

    # ============================
    # 1️⃣ PRIMARY — RxNorm
    # ============================
    try:
        url = f"{RXNORM_BASE}/rxcui.json"
        resp = requests.get(url, params={"name": drugname}, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        rx_ids = data.get("idGroup", {}).get("rxnormId", [])

        # Approximate lookup
        if not rx_ids:
            approx_url = f"{RXNORM_BASE}/approximateTerm.json"
            approx_resp = requests.get(
                approx_url,
                params={"term": drugname, "maxEntries": 1},
                timeout=TIMEOUT
            )
            approx_resp.raise_for_status()
            approx_data = approx_resp.json()
            candidates = approx_data.get("approximateGroup", {}).get("candidate", [])

            if not candidates:
                raise ValueError("RxNorm: No candidates found")

            rxcui = candidates[0].get("rxcui")
        else:
            rxcui = rx_ids[0]

        # Canonical name
        name_url = f"{RXNORM_BASE}/rxcui/{rxcui}.json"
        name_resp = requests.get(name_url, timeout=TIMEOUT)
        name_resp.raise_for_status()
        name_data = name_resp.json()

        canonical_name = name_data.get("idGroup", {}).get("name")
        
        # CRITICAL: Ensure RxNorm returned a valid name
        if not canonical_name:
            raise ValueError("RxNorm returned no canonical name")

        print(f"[RxNorm] Normalized '{drugname}' → '{canonical_name}' ({rxcui})")

        return {
            "input": drugname,
            "rxcui": rxcui,
            "canonical_name": canonical_name
        }

    except Exception as e:
        print(f"[RxNorm] FAILED for '{drugname}': {e}")

    # ============================
    # 2️⃣ SECONDARY — openFDA (Validation Only)
    # ============================
    try:
        params = {
            "search": f'openfda.generic_name:"{drugname.lower()}"',
            "limit": 1
        }
        resp = requests.get(OPENFDA_BASE, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("results"):
            print(f"[openFDA] Validated drug '{drugname}' via FDA label")

            return {
                "input": drugname,
                "rxcui": None,
                "canonical_name": drugname  # accept input as canonical
            }

    except Exception as e:
        print(f"[openFDA] FAILED for '{drugname}': {e}")

    # ============================
    # 3️⃣ FINAL FALLBACK — REJECT UNVALIDATED (FAIL HARD)
    # ============================
    print(f"[Fallback] Rejecting unvalidated drug name: '{drugname}'")
    return None


# =========================
# Drug Detection from PDF Text (NEW)
# =========================

# Common drug name patterns
COMMON_DRUGS = [
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol",
    "metformin", "atorvastatin", "lisinopril", "amlodipine",
    "omeprazole", "levothyroxine", "simvastatin", "losartan",
    "gabapentin", "hydrochlorothiazide", "metoprolol", "prednisone",
    "warfarin", "amoxicillin", "azithromycin", "cephalexin",
    "ciprofloxacin", "doxycycline", "penicillin", "insulin",
    "albuterol", "fluticasone", "montelukast", "sertraline",
    "escitalopram", "duloxetine", "venlafaxine", "alprazolam",
    "clonazepam", "lorazepam", "zolpidem", "trazodone",
    "furosemide", "spironolactone", "carvedilol", "digoxin",
    "clopidogrel", "rivaroxaban", "apixaban", "dabigatran"
]

# Medical stopwords that should NEVER be detected as drugs (CRITICAL)
INVALID_DRUG_WORDS = {
    "reaction", "event", "patient", "report", "serious",
    "adverse", "case", "hospital", "death", "unknown",
    "outcome", "hospitalization", "disability", "congenital",
    "anomaly", "intervention", "required", "recover", "recovered",
    "fatal", "nonfatal", "causality", "assessment", "narrative",
    "description", "summary", "background", "medical", "history"
}


def normalize_for_drug_detection(text: str) -> str:
    """
    Normalize text specifically for drug name detection.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase for matching
    return text.lower().strip()


def detect_drug_from_text(text: str) -> Optional[str]:
    """
    Auto-detect drug name from PDF text.
    
    CRITICAL OPTIMIZATION: Only search first 3000 characters.
    Drug names almost always appear early in safety reports.
    
    Strategy:
    1. Search limited window (first 3000 chars)
    2. Look for STRICT labeled patterns only
    3. Look for common drug names
    4. Validate with RxNorm
    5. Return first valid match
    """
    # Normalize text
    normalized = normalize_for_drug_detection(text)
    
    # 🔒 LIMIT SEARCH WINDOW (prevents noise from full document)
    search_text = normalized[:3000]
    
    # ============================
    # 1️⃣ Look for STRICT explicit labels (FIXED)
    # ============================
    patterns = [
        r'suspected\s+drug[\s:]+([a-z][a-z0-9\-]+)',
        r'drug\s+name[\s:]+([a-z][a-z0-9\-]+)',
        r'medication[\s:]+([a-z][a-z0-9\-]+)',
        r'product[\s:]+([a-z][a-z0-9\-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, search_text)
        if match:
            drug_candidate = match.group(1).strip()
            
            # CRITICAL: Check against stopwords
            if drug_candidate in INVALID_DRUG_WORDS:
                print(f"[Drug Detection] Rejected stopword: '{drug_candidate}'")
                continue
            
            # Validate it's a real drug with minimum length
            if len(drug_candidate) > 3:
                # 🔥 CRITICAL: Validate with RxNorm BEFORE returning
                rx_check = lookup_drug_rxnorm(drug_candidate)
                
                if rx_check and rx_check.get("canonical_name"):
                    print(f"[Drug Detection] Found VALID drug via pattern: '{drug_candidate}'")
                    return drug_candidate
                else:
                    print(f"[Drug Detection] Rejected invalid drug via pattern: '{drug_candidate}'")
                    continue
    
    # ============================
    # 2️⃣ Look for common drug names (WITH VALIDATION)
    # ============================
    for drug in COMMON_DRUGS:
        # Word boundary search to avoid partial matches
        pattern = r'\b' + re.escape(drug) + r'\b'
        if re.search(pattern, search_text):
            # 🔥 CRITICAL: Validate with RxNorm even for common drugs
            rx_check = lookup_drug_rxnorm(drug)
            
            if rx_check and rx_check.get("canonical_name"):
                print(f"[Drug Detection] Found VALID common drug: '{drug}'")
                return drug
            else:
                print(f"[Drug Detection] Rejected invalid common drug: '{drug}'")
                continue
    
    # ============================
    # 3️⃣ Look for capitalized words (potential brand names) WITH VALIDATION
    # ============================
    # This is more aggressive - use with caution
    # Looks for capitalized words that might be brand names
    brand_pattern = r'\b([A-Z][a-z]{3,})\b'
    matches = re.findall(brand_pattern, text[:3000])  # Use original case
    
    if matches:
        # Filter out common words
        common_words = {'Patient', 'Report', 'Date', 'Event', 'Adverse', 
                       'Serious', 'Death', 'Hospital', 'Doctor', 'Physician',
                       'Medical', 'History', 'Outcome', 'Narrative'}
        
        for match in matches:
            if match not in common_words and len(match) > 4:
                # 🔥 CRITICAL: Validate with RxNorm
                rx_check = lookup_drug_rxnorm(match)
                
                if rx_check and rx_check.get("canonical_name"):
                    print(f"[Drug Detection] Found VALID brand name: '{match}'")
                    return match
                else:
                    print(f"[Drug Detection] Rejected invalid brand name: '{match}'")
                    continue
    
    print("[Drug Detection] No drug name detected in PDF")
    return None


def extract_drug_from_filename(filename: str) -> Optional[str]:
    """
    Try to extract drug name from PDF filename.
    Example: "aspirin_safety_report.pdf" → "aspirin"
    """
    # Remove extension
    name = filename.lower().replace('.pdf', '')
    
    # Split by common separators
    parts = re.split(r'[_\-\s]+', name)
    
    # Check each part against common drugs
    for part in parts:
        if part in COMMON_DRUGS:
            print(f"[Drug Detection] Found in filename: '{part}'")
            return part
    
    return None