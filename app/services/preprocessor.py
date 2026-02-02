"""
Text Preprocessor Module
Handles text normalization for adverse event narratives.
"""
import re
import string
from typing import Optional

# Lazy NLTK imports to avoid startup overhead
_stopwords = None
_lemmatizer = None


def _get_stopwords():
    """Lazy load NLTK stopwords."""
    global _stopwords
    if _stopwords is None:
        try:
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words('english'))
        except Exception:
            # Fallback to basic stopwords if NLTK not available
            _stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                         'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                         'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below',
                         'between', 'under', 'again', 'further', 'then', 'once',
                         'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                         'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                         'very', 'just', 'also', 'now', 'here', 'there', 'when',
                         'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
                         'other', 'some', 'such', 'no', 'any', 'i', 'me', 'my',
                         'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                         'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                         'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                         'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                         'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                         'those', 'am'}
    return _stopwords


def _get_lemmatizer():
    """Lazy load NLTK lemmatizer."""
    global _lemmatizer
    if _lemmatizer is None:
        try:
            from nltk.stem import WordNetLemmatizer
            _lemmatizer = WordNetLemmatizer()
        except Exception:
            _lemmatizer = None
    return _lemmatizer


def clean_text(text: str) -> str:
    """
    Clean and normalize text for ML processing.
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters and digits
    3. Remove extra whitespace
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation except hyphens in medical terms
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Replace multiple hyphens with single
    text = re.sub(r'-+', '-', text)
    
    # Remove standalone hyphens
    text = re.sub(r'\s-\s', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords while preserving medical terms.
    
    Args:
        text: Cleaned text string
        
    Returns:
        Text with stopwords removed
    """
    if not text:
        return ""
    
    stopwords = _get_stopwords()
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
    
    return ' '.join(filtered_words)


def lemmatize_text(text: str) -> str:
    """
    Apply lemmatization to reduce words to base forms.
    
    Args:
        text: Text string
        
    Returns:
        Lemmatized text
    """
    if not text:
        return ""
    
    lemmatizer = _get_lemmatizer()
    if lemmatizer is None:
        return text  # Return unchanged if lemmatizer not available
    
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    
    return ' '.join(lemmatized_words)


def preprocess_text(text: str, 
                    remove_stops: bool = True,
                    apply_lemmatization: bool = False) -> str:
    """
    Full preprocessing pipeline for adverse event text.
    
    Args:
        text: Raw adverse event narrative
        remove_stops: Whether to remove stopwords
        apply_lemmatization: Whether to apply lemmatization
        
    Returns:
        Preprocessed text ready for vectorization
    """
    # Clean the text
    text = clean_text(text)
    
    # Remove stopwords if requested
    if remove_stops:
        text = remove_stopwords(text)
    
    # Apply lemmatization if requested
    if apply_lemmatization:
        text = lemmatize_text(text)
    
    return text


def combine_features(drugname: str, adverse_event: str) -> str:
    """
    Combine drug name and adverse event into a single text feature.
    
    Args:
        drugname: Name of the drug
        adverse_event: Description of the adverse event
        
    Returns:
        Combined preprocessed text
    """
    drug_clean = clean_text(drugname) if drugname else ""
    event_clean = preprocess_text(adverse_event) if adverse_event else ""
    
    # Combine with drug name given slight emphasis
    combined = f"{drug_clean} {drug_clean} {event_clean}"
    
    return combined.strip()
