#================================================================================================
# DESCRIPTION : defines Pydantic request and response schemas for the pharmacovigilance API, 
# ensuring strict validation,consistent data exchange, explainable ML outputs, auditability.
#================================================================================================

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# ============ Request Models ============

#=================================================================
# Validates the minimum input (drug name + adverse event text) 
#=================================================================

class ProcessReportRequest(BaseModel):
    """Request to process a single adverse event report."""
    drugname: str = Field(..., description="Name of the drug")
    adverse_event: str = Field(..., description="Description of the adverse event")
    
    class Config:
        json_schema_extra = {
            "example": {
                "drugname": "Aspirin",
                "adverse_event": "Patient experienced severe bleeding after taking medication"
            }
        }

#========================================================
# Validates query parameters for retrieving audit logs
#========================================================

class AuditQueryRequest(BaseModel):
    """Request for querying audit logs."""
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)
    risk_level: Optional[str] = Field(default=None, description="Filter by risk level")
    escalated_only: bool = Field(default=False)
    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD format")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD format")


# ============ Response Models ============

#============================================================================
# Holds structured drug and symptom entities extracted from raw clinical text
#============================================================================

class EntityExtractionResult(BaseModel):
    """Extracted entities from the report."""
    drug: str
    symptoms: List[str]
    extraction_method: str = Field(description="regex or rule-based")


#=========================================================================
# Contains the ML model's classification output, including probabilities 
#=========================================================================
class MLClassificationResult(BaseModel):
    """ML classification result."""
    prediction: str = Field(description="Serious or Non-Serious")
    serious_probability: float
    non_serious_probability: float
    confidence: float

#============================================================================
# Contains the escalation decision details, including risk level, final score
#============================================================================

class EscalationDecision(BaseModel):
   
    should_escalate: bool
    risk_level: str = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    final_score: float
    ml_probability: float
    keyword_score: float
    triggered_keywords: List[str]
    explanation: str

#==================================================================================
# Represents similar historical serious events retrieved based on vector similarity
#==================================================================================

class SimilarEventResult(BaseModel):
   
    report_id: str
    drugname: str
    adverse_event: str
    timestamp: Optional[str] = None
    risk_level: Optional[str] = None
    ml_probability: Optional[float] = None
    final_score: Optional[float] = None
    matched_symptoms: Optional[List[str]] = None

#==================================================================================
# Contains FDA drug information retrieved from DailyMed for contextual enrichment
#==================================================================================

class DrugInfo(BaseModel):
    """FDA Drug Information from DailyMed"""
    source: str
    indications: str
    warnings: str
    adverse_reactions: str

#===========================================================================================
# Represents the medical reasoning analysis from Phi-3, including alignment with ML decision
#===========================================================================================

class Phi3ReasoningResult(BaseModel):
    """Phi-3 medical reasoning analysis result."""
    reasoning_alignment: str = Field(
        ..., 
        description="How Phi-3's reasoning relates to ML decision: SUPPORTS, CHALLENGES, or UNAVAILABLE"
    )
    reasoning: str = Field(
        ..., 
        description="Medical reasoning explanation from Phi-3"
    )
    key_factors: List[str] = Field(
        default_factory=list, 
        description="Medical factors considered by Phi-3"
    )
    reasoning_certainty: str = Field(
        ..., 
        description="Self-reported certainty level: HIGH, MEDIUM, LOW, or UNKNOWN"
    )

#===============================================================================================
# Response model for the main report processing endpoint, combining all results and explanations
#===============================================================================================

class ProcessReportResponse(BaseModel):
    """Response for a processed adverse event report."""
    report_id: str
    drugname: str
    adverse_event: str
    entities: EntityExtractionResult
    classification: MLClassificationResult
    escalation: EscalationDecision
    processing_time_ms: float
    explanation: str = Field(description="AI-generated clinical explanation")
    email_notification: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Email notification status for high-risk cases"
    )
    similar_events: List[SimilarEventResult] = Field(
        default_factory=list,
        description="Similar historical serious events (vector-based retrieval)"
    )
    drug_info: Optional[DrugInfo] = Field(
        None,
        description="FDA drug information from DailyMed"
    )
    
    phi3_reasoning: Optional[Phi3ReasoningResult] = Field(
        None,
        description="Medical reasoning analysis from Phi-3 (if available)"
    )
    needs_human_review: Optional[bool] = Field(
        None,
        description="Flag indicating if human review is recommended"
    )
    review_reason: Optional[str] = Field(
        None,
        description="Reason why human review is recommended"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "RPT-20240121120000-ABC12345",
                "drugname": "Aspirin",
                "adverse_event": "Patient experienced severe bleeding",
                "entities": {
                    "drug": "Aspirin",
                    "symptoms": ["bleeding"],
                    "extraction_method": "regex"
                },
                "classification": {
                    "prediction": "Serious",
                    "serious_probability": 0.85,
                    "non_serious_probability": 0.15,
                    "confidence": 0.85
                },
                "escalation": {
                    "should_escalate": True,
                    "risk_level": "HIGH",
                    "final_score": 0.82,
                    "ml_probability": 0.85,
                    "keyword_score": 0.7,
                    "triggered_keywords": ["bleeding"],
                    "explanation": "High-risk clinical indicators and hospitalization detected."
                },
                "processing_time_ms": 125.5,
                "explanation": "Based on the ML analysis (Serious probability: 85.2%), combined with triggered regulatory keywords: bleeding, this case is classified as HIGH risk.",
                "email_notification": {
                    "attempted": True,
                    "sent": True,
                    "recipient": "safety@company.com",
                    "timestamp": "2024-01-21T12:00:00"
                },
                "similar_events": [
                    {
                        "report_id": "AE-20240115-XYZ",
                        "drugname": "Aspirin",
                        "adverse_event": "Patient developed gastrointestinal bleeding",
                        "timestamp": "2024-01-15T10:30:00",
                        "risk_level": "HIGH",
                        "ml_probability": 0.82,
                        "final_score": 0.78,
                        "matched_symptoms": ["bleeding"]
                    }
                ],
                "drug_info": {
                    "source": "DailyMed",
                    "indications": "Pain relief, fever reduction",
                    "warnings": "Risk of bleeding, gastrointestinal issues",
                    "adverse_reactions": "Bleeding, stomach upset, allergic reactions"
                },
                "phi3_reasoning": {
                    "reasoning_alignment": "SUPPORTS",
                    "reasoning": "The event meets FDA seriousness criteria including severe bleeding requiring medical intervention.",
                    "key_factors": [
                        "Severe bleeding - life-threatening",
                        "Medical intervention required",
                        "Meets FDA criteria for serious adverse event"
                    ],
                    "reasoning_certainty": "HIGH"
                },
                "needs_human_review": False,
                "review_reason": None
            }
        }

#============================================================================
# Represents a single audit log record for traceability and analysis
#============================================================================

class AuditLogRecord(BaseModel):
    
    id: int
    report_id: str
    timestamp: str
    drugname: str
    adverse_event: str
    ml_prediction: str
    ml_probability: float
    extracted_drug: str
    extracted_symptoms: str
    escalation_decision: str
    risk_level: str
    final_score: float
    triggered_keywords: str
    explanation: str
    processing_time_ms: float


#=====================================================================================
# Response model for querying audit logs, including pagination and filtering metadata
#=====================================================================================
class AuditLogResponse(BaseModel):
    total: int
    records: List[Dict]
