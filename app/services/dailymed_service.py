"""
DailyMed FDA Drug Information Service
Fetches and caches FDA drug label information from DailyMed API
"""

import requests
import sqlite3
from datetime import datetime
from typing import Optional, Dict
import re
from bs4 import BeautifulSoup


# Database setup
DB_PATH = "app/database/drug_info_cache.db"


def init_drug_info_db():
    """Initialize the drug info cache database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drug_info_cache (
            drug_name TEXT PRIMARY KEY,
            indications TEXT,
            warnings TEXT,
            adverse_reactions TEXT,
            source TEXT,
            fetched_at DATETIME
        )
    """)
    
    conn.commit()
    conn.close()


def fetch_from_sqlite(drug_name: str) -> Optional[Dict]:
    """Check if drug info is cached in SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT indications, warnings, adverse_reactions, source, fetched_at
            FROM drug_info_cache
            WHERE drug_name = ?
        """, (drug_name.upper(),))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "source": row[3],
                "indications": row[0],
                "warnings": row[1],
                "adverse_reactions": row[2],
                "fetched_at": row[4]
            }
        
        return None
    
    except Exception as e:
        print(f"SQLite fetch error: {e}")
        return None


def save_to_sqlite(drug_name: str, parsed_data: Dict):
    """Save parsed drug info to SQLite cache."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO drug_info_cache 
            (drug_name, indications, warnings, adverse_reactions, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            drug_name.upper(),
            parsed_data["indications"],
            parsed_data["warnings"],
            parsed_data["adverse_reactions"],
            parsed_data["source"],
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Cached drug info for: {drug_name}")
    
    except Exception as e:
        print(f"SQLite save error: {e}")


def clean_text(text: str) -> str:
    """Clean and normalize text from FDA labels."""
    if not text:
        return "Information not available"
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common XML/HTML artifacts
    text = re.sub(r'<[^>]+>', '', text)
    
    # Limit length for UI display
    if len(text) > 1500:
        text = text[:1500] + "..."
    
    return text.strip()


def parse_spl_sections(spl_xml: str) -> Dict:
    """Parse SPL XML to extract key sections."""
    try:
        soup = BeautifulSoup(spl_xml, 'xml')
        
        # Default values
        indications = "Information not available"
        warnings = "Information not available"
        adverse_reactions = "Information not available"
        
        # Find sections by code
        sections = soup.find_all('section')
        
        for section in sections:
            code = section.find('code')
            if not code:
                continue
            
            code_value = code.get('code', '')
            text_elem = section.find('text')
            
            if not text_elem:
                continue
            
            section_text = text_elem.get_text(separator=' ', strip=True)
            
            # Map section codes to our fields
            if code_value in ['34067-9', '43678-2']:  # Indications
                indications = clean_text(section_text)
            
            elif code_value in ['34071-1', '43685-7', '34076-0']:  # Warnings
                warnings = clean_text(section_text)
            
            elif code_value in ['34084-4']:  # Adverse Reactions
                adverse_reactions = clean_text(section_text)
        
        return {
            "source": "FDA DailyMed",
            "indications": indications,
            "warnings": warnings,
            "adverse_reactions": adverse_reactions
        }
    
    except Exception as e:
        print(f"SPL parsing error: {e}")
        return {
            "source": "FDA DailyMed",
            "indications": "Information not available",
            "warnings": "Information not available",
            "adverse_reactions": "Information not available"
        }


def fetch_from_dailymed(drug_name: str) -> Optional[Dict]:
    """Fetch drug information from DailyMed API."""
    try:
        # Step 1: Search for drug to get setid
        search_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
        search_params = {
            "drug_name": drug_name
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=10)
        
        if search_response.status_code != 200:
            print(f"DailyMed search failed for {drug_name}")
            return None
        
        search_data = search_response.json()
        
        if not search_data.get('data') or len(search_data['data']) == 0:
            print(f"No DailyMed results for {drug_name}")
            return None
        
        # Get the first result's setid
        setid = search_data['data'][0].get('setid')
        
        if not setid:
            return None
        
        # Step 2: Fetch SPL XML using setid
        spl_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
        spl_response = requests.get(spl_url, timeout=10)
        
        if spl_response.status_code != 200:
            return None
        
        # Step 3: Parse the SPL XML
        parsed = parse_spl_sections(spl_response.text)
        return parsed
    
    except Exception as e:
        print(f"DailyMed API error for {drug_name}: {e}")
        return None


def get_drug_info(drug_name: str) -> Dict:
    """
    Main function: Get FDA drug information (cached or fresh).
    
    Returns dictionary with:
    - source
    - indications
    - warnings
    - adverse_reactions
    """
    
    # Normalize drug name
    normalized_name = drug_name.upper().strip()
    
    # Check cache first
    cached = fetch_from_sqlite(normalized_name)
    if cached:
        print(f"✅ Using cached drug info for: {drug_name}")
        return cached
    
    # Fetch from DailyMed
    print(f"🔍 Fetching drug info from DailyMed for: {drug_name}")
    fresh_data = fetch_from_dailymed(drug_name)
    
    if fresh_data:
        # Save to cache
        save_to_sqlite(normalized_name, fresh_data)
        return fresh_data
    
    # Fallback if API fails
    print(f"⚠️ Could not fetch drug info for: {drug_name}")
    return {
        "source": "FDA DailyMed",
        "indications": "Information not available",
        "warnings": "Information not available",
        "adverse_reactions": "Information not available"
    }


# Initialize database on import
init_drug_info_db()