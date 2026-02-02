"use client";

import { useEffect, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export default function AuditPage() {
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/audit`);
        if (!res.ok) throw new Error("Failed to fetch audit logs");
        const data = await res.json();
        setLogs(data.records || []);
      } catch (err) {
        console.error(err);
        setError("Failed to load audit logs.");
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, []);

  return (
    <main className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold">Audit Logs</h1>

        {loading && <p>Loading audit logs...</p>}
        {error && <p className="text-red-600">{error}</p>}

        {!loading && !error && (
          <div className="bg-white rounded-xl shadow overflow-x-auto">
            <table className="min-w-full border-collapse">
              <thead className="bg-slate-100">
                <tr>
                  <th className="p-3 text-left">Timestamp</th>
                  <th className="p-3 text-left">Report ID</th>
                  <th className="p-3 text-left">Drug</th>
                  <th className="p-3 text-left">Event</th>
                  <th className="p-3 text-left">Prediction</th>
                  <th className="p-3 text-left">Risk</th>
                  <th className="p-3 text-left">Decision</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, idx) => (
                  <tr key={idx} className="border-t">
                    <td className="p-3 text-sm">
                      {log.timestamp}
                    </td>
                    <td className="p-3 text-sm">
                      {log.report_id}
                    </td>
                    <td className="p-3 text-sm">
                      {log.drugname}
                    </td>
                    <td className="p-3 text-sm max-w-xs truncate">
                      {log.adverse_event}
                    </td>
                    <td className="p-3 text-sm">
                      {log.ml_prediction}
                    </td>
                    <td className="p-3 text-sm font-semibold">
                      {log.risk_level}
                    </td>
                    <td className="p-3 text-sm">
                      {log.escalation_decision}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
