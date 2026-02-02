"""
FastAPI Main Application
Pharmacovigilance Adverse Event Processing API
ENHANCED: With similar serious event retrieval + FDA Drug Information + Phi-3 Medical Reasoning
"""

import os
from dotenv import load_dotenv
from app.services.rag_service import (
    extract_text_from_pdf,
    embed_and_store_pdf,
    retrieve_context,
    normalize_text
)


load_dotenv()

print("GROQ_API_KEY loaded in main:", bool(os.getenv("GROQ_API_KEY")))

import time
from typing import Optional, List
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Auth imports
from app.models.user import UserCreate, UserLogin, AuthResponse
from app.services.auth_service import create_user, authenticate_user, get_user_by_id, verify_token


from app.models.schemas import (
    ProcessReportRequest, ProcessReportResponse,
    BatchProcessRequest, BatchProcessResponse,
    AuditLogResponse, AuditSummaryResponse,
    HealthCheckResponse, ModelInfoResponse,
    EntityExtractionResult, MLClassificationResult, EscalationDecision
)
from app.services.classifier import predict_from_features, load_model, is_model_loaded, get_model_info
from app.services.rxnorm_service import lookup_drug_rxnorm, detect_drug_from_text, extract_drug_from_filename
from app.services.entity_extractor import extract_entities
from app.services.escalation_engine import evaluate_escalation
from app.services.audit_logger import (
    log_processing, generate_report_id,
    get_audit_logs, get_audit_by_report_id, get_audit_summary,
    get_similar_serious_events  
)
from app.services.alert_service import trigger_alert
from app.database.db import get_db_stats
from app.services.explanation_api_service import generate_deterministic_explanation
from app.services.email_service import send_escalation_email
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from app.services.audit_logger import get_similar_serious_events_vector
from fastapi.concurrency import run_in_threadpool
from app.services.drug_suggestion import get_drug_suggestions
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer


# NEW: Import FDA drug information service
from app.services.dailymed_service import get_drug_info

# NEW: Import Phi-3 Reasoning Analyzer
from app.services.llm_reasoning_analyzer import llmReasoningAnalyzer


# Initialize FastAPI
app = FastAPI(
    title="Pharmacovigilance AE Processing System",
    description="End-to-end adverse event processing with ML classification, entity extraction, and rule-based escalation",
    version="1.0.0"
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Initialize Phi-3 Reasoning Analyzer (singleton)
phi3_analyzer = None


# Startup event - load model
@app.on_event("startup")
async def startup_event():
    """Load ML model and initialize Phi-3 analyzer on startup."""
    global phi3_analyzer
    
    print("Loading ML model...")
    if load_model():
        print("Model loaded successfully!")
    else:
        print("WARNING: Model not loaded. Train the model first with: python ml/train_classifier.py")
    


# ============ CORE PROCESSING FUNCTION ============

def process_adverse_event_core(drugname: str, adverse_event: str):
    """
    Core processing pipeline used by both manual text and PDF uploads.
    Returns standardized dictionary with all processing results.
    ENHANCED: Includes similar serious event retrieval for escalated cases + FDA drug info + Phi-3 medical reasoning.
    """
    start_time = time.time()

    # 1. RxNorm Drug Validation
    rxnorm_result = lookup_drug_rxnorm(drugname)
    if not rxnorm_result:
        raise HTTPException(status_code=400, detail=f"Invalid drug name '{drugname}'")

    # CRITICAL: Ensure canonical_drug is never None
    canonical_drug = rxnorm_result.get("canonical_name") or drugname
    
    if not canonical_drug:
        raise HTTPException(status_code=400, detail=f"Could not determine canonical drug name for '{drugname}'")

    # NEW: Fetch FDA Drug Information (cached)
    drug_info = get_drug_info(canonical_drug)

    # 2. ML Classification (AUTHORITATIVE DECISION)
    ml_result = predict_from_features(canonical_drug, adverse_event)

    # NEW: 2b. Phi-3 Medical Reasoning Analysis
    phi3_result = None
    reasoning_alignment = "UNAVAILABLE"
    
    if phi3_analyzer:
        try:
            phi3_result = phi3_analyzer.analyze_prediction(
                drugname=canonical_drug,
                adverse_event=adverse_event,
                ml_prediction=ml_result.get("prediction"),
                ml_confidence=ml_result.get("confidence"),
                ml_reason=ml_result.get("reason"),
                serious_probability=ml_result.get("serious_probability")
            )
            reasoning_alignment = phi3_result.get("reasoning_alignment", "UNAVAILABLE")
            
            print(f"🧠 Phi-3 Reasoning: {reasoning_alignment}")
            print(f"   Medical factors: {phi3_result.get('key_factors', [])}")
            
        except Exception as e:
            print(f"⚠️ Phi-3 analysis failed: {str(e)}")
            phi3_result = {
                "reasoning_alignment": "UNAVAILABLE",
                "reasoning": f"Phi-3 reasoning unavailable: {str(e)}",
                "key_factors": [],
                "reasoning_certainty": "UNKNOWN",
                "error": str(e)
            }
            reasoning_alignment = "UNAVAILABLE"
    else:
        # Phi-3 not available
        phi3_result = {
            "reasoning_alignment": "UNAVAILABLE",
            "reasoning": "Phi-3 reasoning analyzer not initialized",
            "key_factors": [],
            "reasoning_certainty": "UNKNOWN"
        }

    # 3. Entity Extraction
    entities = extract_entities(canonical_drug, adverse_event)

    # 4. Escalation Evaluation
    escalation = evaluate_escalation(
        ml_prediction=ml_result,
        drugname=canonical_drug,
        adverse_event=adverse_event,
        entities=entities
    )

    # 5. Deterministic Explanation
    explanation_text = generate_deterministic_explanation(
        drug=canonical_drug,
        adverse_event=adverse_event,
        ml_result=ml_result,
        escalation=escalation
    )

    processing_time_ms = (time.time() - start_time) * 1000
    report_id = generate_report_id()

    # 6. Audit Logging (with None-safety)
    log_processing(
        report_id=report_id,
        drugname=canonical_drug or "UNKNOWN",  # DEFENSIVE: Never log None
        adverse_event=adverse_event,
        ml_prediction=ml_result,
        entities=entities,
        escalation_result=escalation,
        processing_time_ms=processing_time_ms
    )

    # 7. Alert Triggers
    if escalation.should_escalate:
        trigger_alert(
            report_id=report_id,
            drugname=canonical_drug,
            adverse_event=adverse_event,
            risk_level=escalation.risk_level,
            explanation=escalation.explanation
        )

    # 8. Email Escalation
    email_notification = {
        "attempted": False,
        "sent": False
    }

    if escalation.risk_level in ["CRITICAL", "HIGH"]:
        try:
            send_escalation_email(
                drug=canonical_drug,
                adverse_event=adverse_event,
                risk_level=escalation.risk_level,
                report_id=report_id,
                explanation=escalation.explanation
            )

            email_notification = {
                "attempted": True,
                "sent": True,
                "recipient": "safety@company.com",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            email_notification = {
                "attempted": True,
                "sent": False,
                "error": str(e)
            }
    else:
        email_notification = {
            "attempted": False,
            "sent": False,
            "reason": "Severity below escalation threshold"
        }

    # 9. Retrieve Similar Serious Events (VECTOR-BASED)
    similar_events = []
    if (ml_result.get("prediction") == "Serious" and escalation.should_escalate):
        similar_events = get_similar_serious_events_vector(
            drugname=canonical_drug,
            adverse_event=adverse_event,
            current_symptoms=entities.get("symptoms", []),
            risk_level=escalation.risk_level,
            current_report_id=report_id,
            limit=5
        )

    # NEW: 10. Determine if human review is needed based on Phi-3 reasoning
    needs_human_review = False
    review_reason = None
    
    if phi3_result:
        # Flag for review if:
        # - Phi-3 reasoning challenges the ML decision
        # - Reasoning alignment is unknown/unavailable
        # - ML confidence is low
        # - Phi-3 reasoning certainty is low
        needs_human_review = (
            reasoning_alignment == "CHALLENGES" or
            reasoning_alignment == "UNKNOWN" or
            reasoning_alignment == "UNAVAILABLE" or
            ml_result.get("confidence", 0) < 0.7 or
            phi3_result.get("reasoning_certainty") == "LOW"
        )
        
        if needs_human_review:
            if reasoning_alignment == "CHALLENGES":
                review_reason = "Phi-3 medical reasoning challenges the ML prediction"
            elif reasoning_alignment in ["UNKNOWN", "UNAVAILABLE"]:
                review_reason = "Unable to determine Phi-3 reasoning alignment"
            elif ml_result.get("confidence", 0) < 0.7:
                review_reason = "Low ML confidence"
            elif phi3_result.get("reasoning_certainty") == "LOW":
                review_reason = "Low certainty in Phi-3 medical reasoning"

    return {
        "report_id": report_id,
        "canonical_drug": canonical_drug,
        "ml_result": ml_result,
        "phi3_result": phi3_result,  # NEW: Include Phi-3 reasoning
        "needs_human_review": needs_human_review,  # NEW: Review flag
        "review_reason": review_reason,  # NEW: Why it needs review
        "entities": entities,
        "escalation": escalation,
        "explanation_text": explanation_text,
        "processing_time_ms": processing_time_ms,
        "email_notification": email_notification,
        "similar_events": similar_events,
        "drug_info": drug_info  # Include FDA drug information
    }


# ============ Processing Endpoints ============

@app.post("/api/process", response_model=ProcessReportResponse, tags=["Processing"])
async def process_report(request: ProcessReportRequest):
    """
    Process adverse event report from manual text input.
    Uses core processing pipeline.
    ENHANCED: Returns similar serious events if escalated + FDA drug info + Phi-3 medical reasoning.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    core = process_adverse_event_core(
        drugname=request.drugname,
        adverse_event=request.adverse_event
    )

    # Build response with Phi-3 reasoning
    response_data = {
        "report_id": core["report_id"],
        "drugname": core["canonical_drug"],
        "adverse_event": request.adverse_event,
        "entities": EntityExtractionResult(
            drug=core["entities"]["drug"],
            symptoms=core["entities"]["symptoms"],
            extraction_method=core["entities"]["extraction_method"]
        ),
        "classification": MLClassificationResult(
            prediction=core["ml_result"]["prediction"],
            serious_probability=core["ml_result"]["serious_probability"],
            non_serious_probability=core["ml_result"]["non_serious_probability"],
            confidence=core["ml_result"]["confidence"]
        ),
        "escalation": EscalationDecision(
            should_escalate=core["escalation"].should_escalate,
            risk_level=core["escalation"].risk_level,
            final_score=core["escalation"].final_score,
            ml_probability=core["escalation"].ml_probability,
            keyword_score=core["escalation"].keyword_score,
            triggered_keywords=core["escalation"].triggered_keywords,
            explanation=core["escalation"].explanation
        ),
        "processing_time_ms": core["processing_time_ms"],
        "explanation": core["explanation_text"],
        "email_notification": core["email_notification"],
        "similar_events": core["similar_events"],
        "drug_info": core["drug_info"]
    }
    
    # NEW: Add Phi-3 reasoning fields
    if core.get("phi3_result"):
        response_data["phi3_reasoning"] = {
            "reasoning_alignment": core["phi3_result"].get("reasoning_alignment"),
            "reasoning": core["phi3_result"].get("reasoning"),
            "key_factors": core["phi3_result"].get("key_factors", []),
            "reasoning_certainty": core["phi3_result"].get("reasoning_certainty")
        }
        response_data["needs_human_review"] = core.get("needs_human_review", False)
        response_data["review_reason"] = core.get("review_reason")

    return ProcessReportResponse(**response_data)


@app.post("/api/process-pdf", response_model=ProcessReportResponse, tags=["Processing"])
async def process_pdf_report(
    drugname: str = Query(None, description="Drug name (optional if auto-detection succeeds)"),
    file: UploadFile = File(...)
):
    """
    Process adverse event report from PDF upload.
    
    IMPROVEMENTS:
    - Auto-detects drug name from PDF text
    - OCR fallback for scanned PDFs
    - Text normalization
    - Manual override if auto-detection fails
    ENHANCED: Returns similar serious events if escalated + FDA drug info + Phi-3 medical reasoning.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    # Extract text from PDF (with OCR fallback)
    contents = await file.read()
    extracted_text = extract_text_from_pdf(contents)

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    # Normalize text
    normalized_text = normalize_text(extracted_text)

    # ============================
    # DRUG AUTO-DETECTION WITH VALIDATION
    # ============================
    
    # Try 1: Detect from PDF content
    detected_drug = detect_drug_from_text(normalized_text)
    
    # CRITICAL: Validate auto-detected drug with RxNorm (HARD GATE)
    if detected_drug:
        rx_check = lookup_drug_rxnorm(detected_drug)
        
        # HARD GATE: must be real drug
        if rx_check and rx_check.get("canonical_name"):
            print(f"✅ Auto-detected VALID drug: '{detected_drug}'")
            # Keep detected_drug
        else:
            print(f"❌ Auto-detected INVALID drug word: '{detected_drug}' - rejecting")
            detected_drug = None  # Reset to trigger fallback
    
    # Try 2: Extract from filename (if auto-detection failed)
    if not detected_drug and file.filename:
        filename_drug = extract_drug_from_filename(file.filename)
        if filename_drug:
            # Also validate filename extraction
            rx_check = lookup_drug_rxnorm(filename_drug)
            if rx_check and rx_check.get("canonical_name"):
                print(f"✅ Filename-detected VALID drug: '{filename_drug}'")
                detected_drug = filename_drug
            else:
                print(f"❌ Filename-detected INVALID drug word: '{filename_drug}' - rejecting")
    
    # Determine final drug name with priority order and DEFENSIVE VALIDATION
    final_drug = None
    
    if detected_drug:
        print(f"✅ Using auto-detected drug: '{detected_drug}'")
        final_drug = detected_drug
    elif drugname:
        # DEFENSIVE: Validate manual drugname too
        rx_check = lookup_drug_rxnorm(drugname)
        if rx_check and rx_check.get("canonical_name"):
            print(f"📝 Using manually provided + validated drug: '{drugname}'")
            final_drug = drugname
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Provided drug '{drugname}' is not a valid drug name."
            )
    
    if not final_drug:
        raise HTTPException(
            status_code=400,
            detail="Drug name could not be detected or validated. Please enter a valid drug name manually."
        )

    # Prepare adverse event (full PDF text)
    adverse_event = extracted_text

    # Run through SAME CORE PIPELINE
    core = process_adverse_event_core(
        drugname=final_drug,
        adverse_event=adverse_event
    )

    # ALSO store PDF for RAG linked to SAME case
    embed_and_store_pdf(extracted_text, core["report_id"])

    # Build response with Phi-3 reasoning
    response_data = {
        "report_id": core["report_id"],
        "drugname": core["canonical_drug"],
        "adverse_event": adverse_event[:1000],  # Truncate for response
        "entities": EntityExtractionResult(
            drug=core["entities"]["drug"],
            symptoms=core["entities"]["symptoms"],
            extraction_method=core["entities"]["extraction_method"]
        ),
        "classification": MLClassificationResult(
            prediction=core["ml_result"]["prediction"],
            serious_probability=core["ml_result"]["serious_probability"],
            non_serious_probability=core["ml_result"]["non_serious_probability"],
            confidence=core["ml_result"]["confidence"]
        ),
        "escalation": EscalationDecision(
            should_escalate=core["escalation"].should_escalate,
            risk_level=core["escalation"].risk_level,
            final_score=core["escalation"].final_score,
            ml_probability=core["escalation"].ml_probability,
            keyword_score=core["escalation"].keyword_score,
            triggered_keywords=core["escalation"].triggered_keywords,
            explanation=core["escalation"].explanation
        ),
        "processing_time_ms": core["processing_time_ms"],
        "explanation": core["explanation_text"],
        "email_notification": core["email_notification"],
        "similar_events": core["similar_events"],
        "drug_info": core["drug_info"]
    }
    
    # NEW: Add Phi-3 reasoning fields
    if core.get("phi3_result"):
        response_data["phi3_reasoning"] = {
            "reasoning_alignment": core["phi3_result"].get("reasoning_alignment"),
            "reasoning": core["phi3_result"].get("reasoning"),
            "key_factors": core["phi3_result"].get("key_factors", []),
            "reasoning_certainty": core["phi3_result"].get("reasoning_certainty")
        }
        response_data["needs_human_review"] = core.get("needs_human_review", False)
        response_data["review_reason"] = core.get("review_reason")

    return ProcessReportResponse(**response_data)


# ============ Audit Endpoints ============

@app.get("/api/audit", response_model=AuditLogResponse, tags=["Audit"])
async def get_audit(
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    risk_level: Optional[str] = Query(default=None),
    escalated_only: bool = Query(default=False),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None)
):
    logs = await run_in_threadpool(
        get_audit_logs,
        limit,
        offset,
        risk_level,
        escalated_only,
        start_date,
        end_date
    )

    return AuditLogResponse(
        total=len(logs),
        records=logs
    )


@app.get("/api/audit/{report_id}", tags=["Audit"])
async def get_audit_record(report_id: str):
    record = get_audit_by_report_id(report_id)
    if not record:
        raise HTTPException(status_code=404, detail="Report not found")
    return record


@app.get("/api/audit/summary", response_model=AuditSummaryResponse, tags=["Audit"])
async def get_summary():
    return get_audit_summary()


# ============ Drug Autocomplete Endpoint ============

@app.get("/api/drugs/suggest", tags=["Drugs"])
async def drug_suggest(q: str, limit: int = 10):
    return {
        "query": q,
        "suggestions": get_drug_suggestions(q, limit)
    }


# ============ System Endpoints ============

@app.get("/api/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    db_stats = get_db_stats()
    
    # Check if Phi-3 is available
    phi3_available = phi3_analyzer is not None
    
    return HealthCheckResponse(
        status="healthy" if is_model_loaded() else "degraded",
        model_loaded=is_model_loaded(),
        llm_available=True,   
        database_connected=db_stats["total_records"] >= 0,
        timestamp=datetime.now().isoformat(),
        phi3_available=phi3_available  # NEW: Add Phi-3 status
    )


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    info = get_model_info()
    
    # Add Phi-3 information
    info["phi3_reasoning"] = {
        "available": phi3_analyzer is not None,
        "model_name": "phi3" if phi3_analyzer else None,
        "description": "Medical reasoning analyzer for ML decision interpretation"
    }
    
    return info


# ============ Root ============

@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Pharmacovigilance AE Processing System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "features": {
            "ml_classification": True,
            "phi3_reasoning": phi3_analyzer is not None,
            "entity_extraction": True,
            "escalation_engine": True,
            "rag_retrieval": True,
            "fda_drug_info": True
        }
    }


# ============ Authentication Endpoints ============

@app.post("/api/auth/signup", response_model=AuthResponse, tags=["Authentication"])
async def signup(request: UserCreate):
    """
    Create a new user account.
    Password is hashed using Argon2.
    """
    result = create_user(
        username=request.username,
        email=request.email,
        password=request.password
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return AuthResponse(
        success=True,
        message="Account created successfully",
        user=result["user"]
    )


@app.post("/api/auth/login", tags=["Authentication"])
async def login(request: UserLogin):
    result = authenticate_user(
        username=request.username,
        password=request.password
    )

    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])

    return {
        "success": True,
        "message": "Login successful",
        "user": result["user"],
        "access_token": result["token"],   
        "token_type": "bearer"
    }



@app.get("/api/auth/me", tags=["Authentication"])
async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(payload["user_id"])

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "success": True,
        "user": user
    }


# ============ RAG Endpoints ============

@app.post("/api/upload-pdf", tags=["RAG"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    RAG-only PDF indexing endpoint.
    Stores PDF text for retrieval without processing through pipeline.
    
    NOTE: This endpoint does NOT run ML classification or escalation.
    Use /api/process-pdf if you need full processing with ML + audit + RAG.
    This endpoint is for cases where you only want to index documents for later retrieval.
    """
    contents = await file.read()

    text = extract_text_from_pdf(contents)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    report_id = generate_report_id()

    chunks_indexed = embed_and_store_pdf(text, report_id)

    return {
        "report_id": report_id,
        "chunks_indexed": chunks_indexed
    }


@app.post("/api/rag-summary", tags=["RAG"])
async def rag_summary(query: str, report_id: str):
    """
    Generate RAG-based narrative summary for a specific report.
    Case-isolated retrieval.
    """
    # Case-isolated retrieval
    chunks, metadatas = retrieve_context(
        query=query,
        report_id=report_id,
        top_k=5
    )

    if not chunks:
        return {"summary": "No relevant PDF context found for this report."}

    context = "\n\n".join(chunks)

    prompt = f"""
You are a pharmacovigilance safety assistant.

Use the following PDF report context to generate a clinical summary
and assess seriousness.

CONTEXT:
{context}

Provide a concise pharmacovigilance-style summary and risk assessment.
"""

    # Reuse Groq LLaMA
    from app.services.explanation_api_service import generate_api_explanation

    summary = generate_api_explanation(
        drug="Unknown",
        adverse_event=prompt,
        classification="Unknown",
        triggered_keywords=[],
        serious_prob=0.0,
        risk_level="Unknown"
    )

    return {
        "summary": summary,
        "chunks_used": len(chunks),
        "report_id": report_id
    }