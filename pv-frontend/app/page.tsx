"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export default function Home() {
  const router = useRouter();

  // ===== AUTH STATE ===== (SECURITY FIXED)
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [authChecking, setAuthChecking] = useState(true);

  // Check authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("access_token"); // ✅ FIXED: consistent naming

      if (!token) {
        router.replace("/login"); // ✅ FIXED: use replace, not push
        return;
      }

      try {
        // ✅ FIXED: Proper Bearer token authentication (no token in URL!)
        const res = await fetch(`${API_BASE}/api/auth/me`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!res.ok) throw new Error("Invalid token");

        const data = await res.json();
        setCurrentUser(data.user);
      } catch {
        // Token invalid, clear and redirect
        localStorage.removeItem("access_token");
        router.replace("/login"); // ✅ FIXED: use replace
      } finally {
        setAuthChecking(false);
      }
    };

    checkAuth();
  }, [router]);

  // Logout handler
  const handleLogout = () => {
    localStorage.removeItem("access_token"); // ✅ FIXED: consistent naming
    router.replace("/login"); // ✅ FIXED: use replace
  };

  // ===== TAB STATE =====
  const [activeTab, setActiveTab] = useState<"manual" | "pdf">("manual");

  // ===== MANUAL TEXT INPUT STATES =====
  const [drug, setDrug] = useState("");
  const [event, setEvent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ===== PDF UPLOAD STATES =====
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfDrugName, setPdfDrugName] = useState("");
  const [pdfUploading, setPdfUploading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  // ===== SHARED RESULTS STATES =====
  const [result, setResult] = useState<any>(null);
  const [similar, setSimilar] = useState<any[]>([]);

  // ===== SELECTED EVENT STATE =====
  const [selectedEvent, setSelectedEvent] = useState<any | null>(null);

  // ===== DRUG AUTOCOMPLETE STATES =====
  const [drugSuggestions, setDrugSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Download Handler
  const handleDownload = () => {
    if (!result) return;

    const content = `
=================================================
PHARMACOVIGILANCE CASE REPORT
=================================================

Case Report ID: ${result.report_id}
Drug: ${result.drugname}

Adverse Event:
${result.adverse_event}

=================================================
ANALYSIS RESULTS
=================================================

Seriousness: ${result.classification.prediction}
ML Probability: ${(result.classification.serious_probability * 100).toFixed(1)}%
Risk Level: ${result.escalation.risk_level}
Escalation Required: ${result.escalation.should_escalate ? "YES" : "NO"}

Triggered Keywords: ${result.escalation.triggered_keywords.length > 0
        ? result.escalation.triggered_keywords.join(", ")
        : "None"}

=================================================
PHI-3 MEDICAL REASONING ANALYSIS
=================================================

${result.phi3_reasoning ? `
Reasoning Alignment: ${result.phi3_reasoning.reasoning_alignment}
Certainty Level: ${result.phi3_reasoning.reasoning_certainty}

Medical Reasoning:
${result.phi3_reasoning.reasoning}

Key Medical Factors:
${result.phi3_reasoning.key_factors.map((f: string) => `- ${f}`).join('\n')}

Human Review Needed: ${result.needs_human_review ? "YES" : "NO"}
${result.review_reason ? `Review Reason: ${result.review_reason}` : ''}
` : "Phi-3 reasoning not available"}

=================================================
AI CLINICAL SUMMARY
=================================================

${result.explanation || "N/A"}

=================================================
FDA DRUG INFORMATION
=================================================

${result.drug_info ? `
Source: ${result.drug_info.source}

INDICATIONS:
${result.drug_info.indications}

WARNINGS:
${result.drug_info.warnings}

ADVERSE REACTIONS:
${result.drug_info.adverse_reactions}
` : "Not available"}

=================================================
SIMILAR HISTORICAL CASES
=================================================

${similar.length > 0
        ? similar.map((s, i) => `
Case ${i + 1}: ${s.report_id}
Drug: ${s.drugname || result.drugname}
Adverse Event: ${s.adverse_event}
---
`).join('\n')
        : "No similar cases found."}

Generated: ${new Date().toLocaleString()}
`;

    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `${result.report_id}.txt`;
    a.click();

    URL.revokeObjectURL(url);
  };

  // FLOW A: MANUAL TEXT ANALYSIS
  const handleAnalyze = async () => {
    if (!drug || !event) {
      setError("Please enter both drug name and adverse event.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setSimilar([]);

    try {
      const res = await fetch(`${API_BASE}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          drugname: drug,
          adverse_event: event,
        }),
      });

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();

      console.log("API Response:", data);
      console.log("Similar Events:", data.similar_events);
      console.log("Drug Info:", data.drug_info);
      console.log("Phi-3 Reasoning:", data.phi3_reasoning);

      setResult(data);

      if (data.similar_events && data.similar_events.length > 0) {
        setSimilar(data.similar_events);
      } else {
        setSimilar([]);
      }
    } catch (err) {
      console.error(err);
      setError("Failed to call backend API.");
    } finally {
      setLoading(false);
    }
  };

  // FLOW B: PDF UPLOAD + ML ANALYSIS
  const handlePdfAnalyze = async () => {
    if (!pdfFile) {
      setPdfError("Please select a PDF file.");
      return;
    }

    setPdfUploading(true);
    setPdfError(null);
    setResult(null);
    setSimilar([]);

    try {
      const formData = new FormData();
      formData.append("file", pdfFile);

      if (pdfDrugName.trim()) {
        formData.append("drugname", pdfDrugName.trim());
      }

      const res = await fetch(`${API_BASE}/api/process-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }

      const data = await res.json();

      console.log("PDF API Response:", data);
      console.log("Similar Events:", data.similar_events);
      console.log("Drug Info:", data.drug_info);
      console.log("Phi-3 Reasoning:", data.phi3_reasoning);

      setResult(data);

      if (data.similar_events && data.similar_events.length > 0) {
        setSimilar(data.similar_events);
      } else {
        setSimilar([]);
      }

      setPdfError(null);
    } catch (err: any) {
      console.error(err);

      if (err.message.includes("could not be detected")) {
        setPdfError(
          "Drug name could not be auto-detected. Please enter it manually above and try again."
        );
      } else {
        setPdfError("Failed to process PDF: " + err.message);
      }
    } finally {
      setPdfUploading(false);
    }
  };

  // Get risk level color
  const getRiskColor = (level: string) => {
    switch (level) {
      case "CRITICAL":
        return "bg-red-600";
      case "HIGH":
        return "bg-orange-500";
      case "MEDIUM":
        return "bg-yellow-500";
      default:
        return "bg-green-500";
    }
  };

  // Get reasoning alignment color
  const getReasoningColor = (alignment: string) => {
    switch (alignment) {
      case "SUPPORTS":
        return "bg-green-100 text-green-800 border-green-300";
      case "CHALLENGES":
        return "bg-orange-100 text-orange-800 border-orange-300";
      case "UNAVAILABLE":
      case "UNKNOWN":
        return "bg-gray-100 text-gray-800 border-gray-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  // Show loading while checking auth
  if (authChecking) {
    return (
      <main className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </main>
    );
  }

  // Don't render if not authenticated
  if (!currentUser) {
    return null;
  }

  return (
    <main className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Pharmacovigilance Analysis
              </h1>
              <p className="text-gray-600 mt-1">
                Analyze adverse event reports for safety assessment.
              </p>
            </div>
            <div className="flex items-center gap-4">
              {currentUser && (
                <span className="text-sm text-gray-600">
                  Welcome, <span className="font-semibold">{currentUser.username}</span>
                </span>
              )}
              <button
                onClick={() => router.push("/audit")}
                className="bg-gray-800 text-white px-4 py-2 rounded-md hover:bg-gray-900 transition"
              >
                View Audit Logs
              </button>
              <button
                onClick={handleLogout}
                className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600 transition"
              >
                Logout
              </button>
            </div>
          </div>
        </div>

        {/* Input Section with Tabs */}
        <div className="bg-white rounded-lg shadow-sm">
          {/* Tabs */}
          <div className="border-b border-gray-200">
            <div className="flex space-x-1 p-1">
              <button
                onClick={() => setActiveTab("manual")}
                className={`px-6 py-3 text-sm font-medium rounded-t-lg transition ${activeTab === "manual"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
              >
                Enter Information
              </button>
              <button
                onClick={() => setActiveTab("pdf")}
                className={`px-6 py-3 text-sm font-medium rounded-t-lg transition ${activeTab === "pdf"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
              >
                Upload PDF
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === "manual" ? (
              <div className="space-y-6">
                <div className="relative">
                  <label className="block text-sm font-semibold text-gray-900 mb-2">
                    Drug Name
                  </label>
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-md p-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Amoxicillin"
                    value={drug}
                    onChange={async (e) => {
                      const value = e.target.value;
                      setDrug(value);

                      if (value.length < 2) {
                        setDrugSuggestions([]);
                        setShowSuggestions(false);
                        return;
                      }

                      try {
                        const res = await fetch(
                          `${API_BASE}/api/drugs/suggest?q=${value}`
                        );
                        const data = await res.json();
                        setDrugSuggestions(data.suggestions || []);
                        setShowSuggestions(true);
                      } catch {
                        setDrugSuggestions([]);
                        setShowSuggestions(false);
                      }
                    }}
                  />
                  {showSuggestions && drugSuggestions.length > 0 && (
                    <ul className="border border-gray-300 rounded-md mt-1 bg-white max-h-48 overflow-y-auto shadow-md absolute w-full z-10">
                      {drugSuggestions.map((name, idx) => (
                        <li
                          key={idx}
                          className="px-4 py-2 hover:bg-blue-100 cursor-pointer text-sm"
                          onClick={() => {
                            setDrug(name);
                            setShowSuggestions(false);
                          }}
                        >
                          {name}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-2">
                    Adverse Event Description
                  </label>
                  <textarea
                    className="w-full border border-gray-300 rounded-md p-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={4}
                    placeholder="Patient developed severe fatigue, nausea, dark urine, and jaundice after 10 days of Amoxicillin therapy."
                    value={event}
                    onChange={(e) => setEvent(e.target.value)}
                  />
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700 disabled:bg-gray-400 transition"
                >
                  {loading ? "Analyzing..." : "Analyze Report"}
                </button>

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-md p-3">
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-2">
                    Drug Name (Optional - Auto-detected)
                  </label>
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-md p-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Leave blank for auto-detection"
                    value={pdfDrugName}
                    onChange={(e) => setPdfDrugName(e.target.value)}
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-2">
                    Select PDF File
                  </label>
                  <input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
                    className="w-full border border-gray-300 rounded-md p-3 text-gray-900"
                  />
                </div>

                <button
                  onClick={handlePdfAnalyze}
                  disabled={pdfUploading || !pdfFile}
                  className="bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700 disabled:bg-gray-400 transition"
                >
                  {pdfUploading ? "Processing PDF..." : "Analyze Report"}
                </button>

                {pdfError && (
                  <div className="bg-red-50 border border-red-200 rounded-md p-3">
                    <p className="text-red-700 text-sm">{pdfError}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        {result && (
          <>
            <h2 className="text-2xl font-bold text-gray-900">
              ANALYSIS RESULTS
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Column - Summary & Analysis */}
              <div className="lg:col-span-2 space-y-6">
                {/* Human Review Alert (if needed) */}
                {result.needs_human_review && (
                  <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-r-lg">
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <span className="text-yellow-400 text-2xl">⚠️</span>
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-bold text-yellow-800">
                          HUMAN REVIEW RECOMMEND
                        </h3>
                        <p className="mt-1 text-sm text-yellow-700">
                          {result.review_reason || "This case requires human review"}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Pharmacovigilance Summary Card */}
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <h3 className="text-lg font-bold text-gray-900 mb-4">
                    PHARMACOVIGILANCE SUMMARY
                  </h3>

                  <div className="space-y-2 text-sm">
                    <div className="flex">
                      <span className="font-semibold text-gray-700 w-48">
                        CASE REPORT ID:
                      </span>
                      <span className="text-gray-900">
                        {result.report_id}
                      </span>
                    </div>
                    <div className="flex">
                      <span className="font-semibold text-gray-700 w-48">
                        SUSPECTED DRUG:
                      </span>
                      <span className="text-gray-900">{result.drugname}</span>
                    </div>
                    <div className="flex">
                      <span className="font-semibold text-gray-700 w-48">
                        ADVERSE EVENT:
                      </span>
                      <span className="text-gray-900">
                        {result.adverse_event.substring(0, 100)}...
                      </span>
                    </div>
                    <div className="flex">
                      <span className="font-semibold text-gray-700 w-48">
                        SERIOUS ASSESSMENT :
                      </span>
                      <span className="text-gray-900 font-semibold">
                        {result.classification.prediction}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Three Column Cards */}
                <div className="grid grid-cols-3 gap-4">
                  {/* Risk Level */}
                  <div className="bg-white rounded-lg shadow-sm p-6 text-center">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">
                      RISK LEVEL
                    </h4>
                    <div
                      className={`${getRiskColor(
                        result.escalation.risk_level
                      )} text-white px-4 py-2 rounded-full font-bold text-lg inline-block mb-3`}
                    >
                      {result.escalation.risk_level}
                    </div>
                    <div className="text-sm text-gray-600 mb-4">
                      {result.escalation.should_escalate
                        ? "Escalation Required"
                        : "No Escalation"}
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                      <div
                        className="bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 h-3 rounded-full"
                        style={{
                          width: `${result.escalation.final_score * 100}%`,
                        }}
                      />
                    </div>
                    <div className="text-xs text-gray-600 flex justify-between">
                      <span>
                        Confidence:{" "}
                        {(result.classification.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  {/* ML Prediction */}
                  <div className="bg-white rounded-lg shadow-sm p-6 text-center">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">
                      ML PREDICTION
                    </h4>
                    <div className="text-3xl font-bold text-gray-900">
                      {result.classification.prediction}
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div className="bg-white rounded-lg shadow-sm p-6">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">
                      QUICK STATS
                    </h4>
                    <div className="space-y-3">
                      {similar.length > 0 ? (
                        <>
                          <div className="flex items-center gap-2 text-sm">
                            <span className="text-gray-400">📄</span>
                            <span className="text-gray-700">
                              <span className="font-bold text-gray-900">{similar.length}</span> similar case(s)
                            </span>
                          </div>

                          {similar.some((s) => s.risk_level) && (
                            <div className="flex items-center gap-2 text-sm">
                              <span className="text-gray-400">⚠️</span>
                              <span className="text-gray-700">
                                <span className="font-bold text-gray-900">
                                  {
                                    similar.filter(
                                      (s) => s.risk_level === "CRITICAL" || s.risk_level === "HIGH"
                                    ).length
                                  }
                                </span>{" "}
                                high-risk
                              </span>
                            </div>
                          )}

                          {similar.some((s) => s.ml_probability) && (
                            <div className="flex items-center gap-2 text-sm">
                              <span className="text-gray-400">📊</span>
                              <span className="text-gray-700">
                                Avg:{" "}
                                <span className="font-bold text-gray-900">
                                  {(
                                    (similar.reduce(
                                      (sum, s) => sum + (s.ml_probability || 0),
                                      0
                                    ) /
                                      similar.length) *
                                    100
                                  ).toFixed(1)}
                                  %
                                </span>
                              </span>
                            </div>
                          )}
                        </>
                      ) : result.escalation.should_escalate ? (
                        <div className="text-sm text-gray-500 italic">
                          NO SIMILAR CASES FOUND
                        </div>
                      ) : (
                        <div className="text-sm text-gray-500 italic">
                          ONLY FOR ESCALATED ISSUES
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Phi-3 Medical Reasoning Card */}
                {result.phi3_reasoning && (
                  <div className="bg-gradient-to-br from-purple-50 to-indigo-50 rounded-lg shadow-sm p-6 border border-purple-100">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <span className="text-purple-600"></span>
                      MEDICAL REASONING ANALYSIS
                    </h3>

                    {/* Reasoning Alignment Badge */}
                    <div className="mb-4">
                      <div className="text-sm font-semibold text-gray-700 mb-2">
                        REASONING ALLIGNMENT
                      </div>
                      <div
                        className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border font-bold text-sm ${getReasoningColor(
                          result.phi3_reasoning.reasoning_alignment
                        )}`}
                      >
                        {result.phi3_reasoning.reasoning_alignment === "SUPPORTS" && "✅"}
                        {result.phi3_reasoning.reasoning_alignment === "CHALLENGES" && "⚠️"}
                        {(result.phi3_reasoning.reasoning_alignment === "UNAVAILABLE" ||
                          result.phi3_reasoning.reasoning_alignment === "UNKNOWN") && "❓"}
                        <span>{result.phi3_reasoning.reasoning_alignment}</span>
                      </div>
                      <div className="text-xs text-gray-600 mt-2">
                        Certainty: {result.phi3_reasoning.reasoning_certainty}
                      </div>
                    </div>

                    {/* Medical Reasoning */}
                    <div className="mb-4">
                      <div className="text-sm font-semibold text-gray-700 mb-2">
                        MEDICAL REASONING
                      </div>
                      <p className="text-sm text-gray-800 leading-relaxed bg-white/50 p-3 rounded-lg">
                        {result.phi3_reasoning.reasoning}
                      </p>
                    </div>

                    {/* Key Medical Factors */}
                    {result.phi3_reasoning.key_factors &&
                      result.phi3_reasoning.key_factors.length > 0 && (
                        <div>
                          <div className="text-sm font-semibold text-gray-700 mb-2">
                            KEY MEDICAL FACTORS
                          </div>
                          <ul className="space-y-2">
                            {result.phi3_reasoning.key_factors.map((factor: string, idx: number) => (
                              <li key={idx} className="flex items-start gap-2 text-sm">
                                <span className="text-purple-600 mt-1">•</span>
                                <span className="text-gray-800">{factor}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                    {/* Interpretation Note */}
                    <div className="mt-4 pt-4 border-t border-purple-200">
                      <p className="text-xs text-gray-600 italic">
                        <strong>Note:</strong> Phi-3 provides medical reasoning to help interpret the ML decision.
                        It does not validate or override the ML classifier's prediction.
                      </p>
                    </div>
                  </div>
                )}

                {/* FDA Drug Information Card */}
                {result.drug_info && (
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg shadow-sm p-6 border border-blue-100">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <span className="text-blue-600"></span>
                      DRUG INFORMATION — {result.drugname}
                    </h3>

                    <div className="text-xs text-gray-500 mb-4">
                      SOURCE: {result.drug_info.source}
                    </div>

                    <div className="space-y-4">
                      {/* Indications */}
                      <div>
                        <h4 className="text-sm font-bold text-gray-800 mb-2">
                          INDICATIONS
                        </h4>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {result.drug_info.indications}
                        </p>
                      </div>

                      {/* Warnings */}
                      <div>
                        <h4 className="text-sm font-bold text-gray-800 mb-2">
                          KEY WARNINGS
                        </h4>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {result.drug_info.warnings}
                        </p>
                      </div>

                      {/* Adverse Reactions */}
                      <div>
                        <h4 className="text-sm font-bold text-gray-800 mb-2">
                          KNOWN ADVERSE REACTIONS
                        </h4>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {result.drug_info.adverse_reactions}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* AI Clinical Explanation */}
                {result.explanation && (
                  <div className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-lg shadow-sm p-6 border border-emerald-100">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <span className="text-emerald-600"></span>
                      AI CLINICAL SUMMARY
                    </h3>
                    <div className="prose prose-sm max-w-none">
                      <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-line">
                        {result.explanation}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column - Similar Events */}
              <div className="space-y-6">
                {similar.length > 0 && (
                  <div className="bg-white rounded-lg shadow-sm p-6">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <span className="text-orange-500"></span>
                      SIMILAR HISTORICAL EVENTS
                    </h3>
                    <p className="text-sm text-gray-600 mb-4">
                      Found {similar.length} similar escalated case(s) with the same drug. Click any case to view full details.
                    </p>

                    <div className="space-y-4">
                      {similar.map((event: any, idx: number) => (
                        <div
                          key={idx}
                          onClick={() => setSelectedEvent(event)}
                          className="border border-gray-200 rounded-lg p-4 hover:border-blue-500 hover:shadow-md cursor-pointer transition"
                        >
                          <div className="text-xs font-mono text-gray-500 mb-3 flex items-center justify-between">
                            <span>Case ID: {event.report_id}</span>
                            {event.timestamp && (
                              <span className="text-gray-400">
                                {new Date(event.timestamp).toLocaleDateString()}
                              </span>
                            )}
                          </div>

                          <div className="space-y-2 text-sm">
                            <div>
                              <span className="font-semibold text-gray-700">Drug:</span>{" "}
                              <span className="text-gray-900">
                                {event.drugname || result.drugname}
                              </span>
                            </div>

                            <div>
                              <span className="font-semibold text-gray-700">Adverse Event:</span>
                              <p className="text-gray-700 mt-1 line-clamp-3 leading-relaxed">
                                {event.adverse_event}
                              </p>
                            </div>
                          </div>

                          <div className="mt-3 text-xs text-blue-600 font-medium">
                            Click to view full details →
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Download Button */}
                    <button
                      onClick={handleDownload}
                      className="mt-6 w-full bg-blue-600 text-white px-4 py-3 rounded-md font-medium hover:bg-blue-700 transition flex items-center justify-center gap-2"
                    >
                      <span>⬇</span>
                      Download Case Report
                    </button>
                  </div>
                )}

                {/* Placeholder if no similar events */}
                {similar.length === 0 && (
                  <div className="bg-white rounded-lg shadow-sm p-6">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                      <span className="text-gray-400">📋</span>
                      Similar Historical Events
                    </h3>
                    <p className="text-sm text-gray-500 italic">
                      {result.escalation.should_escalate
                        ? "No similar historical cases found for this drug."
                        : "Similar event retrieval only active for escalated serious cases."}
                    </p>

                    {/* Download Button */}
                    <button
                      onClick={handleDownload}
                      className="mt-6 w-full bg-blue-600 text-white px-4 py-3 rounded-md font-medium hover:bg-blue-700 transition flex items-center justify-center gap-2"
                    >
                      <span>⬇</span>
                      Download Case Report
                    </button>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Modal: Similar Event Details */}
      {selectedEvent && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedEvent(null)}
        >
          <div
            className="bg-white max-w-3xl w-full rounded-lg shadow-2xl max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
              <h3 className="text-xl font-bold text-gray-900">
                Similar Event Details
              </h3>
              <button
                onClick={() => setSelectedEvent(null)}
                className="text-gray-500 hover:text-gray-900 text-2xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-gray-100 transition"
              >
                ×
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-4">
              {/* Case ID */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-xs font-semibold text-gray-500 mb-1">CASE REPORT ID</div>
                <div className="text-lg font-mono font-bold text-gray-900">
                  {selectedEvent.report_id}
                </div>
              </div>

              {/* Drug Name */}
              <div>
                <div className="text-sm font-semibold text-gray-700 mb-2">Drug</div>
                <div className="text-base text-gray-900">
                  {selectedEvent.drugname || result.drugname}
                </div>
              </div>

              {/* Adverse Event */}
              <div>
                <div className="text-sm font-semibold text-gray-700 mb-2">Adverse Event Description</div>
                <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-800 leading-relaxed whitespace-pre-line">
                  {selectedEvent.adverse_event}
                </div>
              </div>

              {/* Additional Details Grid */}
              <div className="grid grid-cols-2 gap-4">
                {selectedEvent.risk_level && (
                  <div>
                    <div className="text-sm font-semibold text-gray-700 mb-2">Risk Level</div>
                    <div className={`${getRiskColor(selectedEvent.risk_level)} text-white px-3 py-1 rounded-full text-sm font-bold inline-block`}>
                      {selectedEvent.risk_level}
                    </div>
                  </div>
                )}

                {selectedEvent.timestamp && (
                  <div>
                    <div className="text-sm font-semibold text-gray-700 mb-2">Date Reported</div>
                    <div className="text-base text-gray-900">
                      {new Date(selectedEvent.timestamp).toLocaleDateString()}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="sticky bottom-0 bg-gray-50 border-t border-gray-200 px-6 py-4">
              <button
                onClick={() => setSelectedEvent(null)}
                className="w-full bg-gray-800 text-white px-4 py-3 rounded-md font-medium hover:bg-gray-900 transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}