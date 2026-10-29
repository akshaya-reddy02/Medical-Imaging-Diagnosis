
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function formatStudyId(id = "") {
  return id.length > 22 ? `${id.slice(0, 22)}…` : id;
}

export default function App() {
  const [studies, setStudies] = useState([]);
  const [selectedStudy, setSelectedStudy] = useState(null);
  const [studyDetails, setStudyDetails] = useState(null);
  const [previewInstances, setPreviewInstances] = useState([]);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [previewError, setPreviewError] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [error, setError] = useState("");
  const [detailsError, setDetailsError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [classificationFile, setClassificationFile] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [classificationError, setClassificationError] = useState("");
  const [reviewComment, setReviewComment] = useState("");
  const [reviewSaving, setReviewSaving] = useState(false);
  const [reviewMessage, setReviewMessage] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [auditLogs, setAuditLogs] = useState([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState("");
  const loadAuditLogs = useCallback(async () => {
    setAuditLoading(true);
    setAuditError("");

    try {
      const response = await fetch(`${API_BASE}/audit-logs?limit=50`);

      if (!response.ok) {
        throw new Error(`Could not load audit history (HTTP ${response.status}).`);
      }

      const data = await response.json();
      setAuditLogs(Array.isArray(data.items) ? data.items : []);
    } catch (err) {
      setAuditError(err.message || "Could not load audit history.");
    } finally {
      setAuditLoading(false);
    }
  }, []);

  const loadStudies = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/orthanc/studies`);

      if (!response.ok) {
        throw new Error(`API returned HTTP ${response.status}`);
      }

      const data = await response.json();
      const ids = Array.isArray(data.studies) ? data.studies : [];

      setStudies(ids);
      setLastUpdated(new Date());

      if (ids.length === 0) {
        setSelectedStudy(null);
        setStudyDetails(null);
      } else {
        setSelectedStudy((current) =>
          current && ids.includes(current) ? current : ids[0]
        );
      }
    } catch {
      setError(
        "Unable to load studies. Check that FastAPI and Orthanc are running."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStudies();
  }, [loadStudies]);

  useEffect(() => {
    if (!selectedStudy) {
      setStudyDetails(null);
      return;
    }

    const controller = new AbortController();

    async function loadStudyDetails() {
      setDetailsLoading(true);
      setDetailsError("");
      setStudyDetails(null);

      try {
        const response = await fetch(
          `${API_BASE}/orthanc/studies/${encodeURIComponent(selectedStudy)}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(`API returned HTTP ${response.status}`);
        }

        const data = await response.json();
        setStudyDetails(data.study ?? null);
      } catch (err) {
        if (err.name !== "AbortError") {
          setDetailsError("Could not retrieve this study's details.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setDetailsLoading(false);
        }
      }
    }

    loadStudyDetails();

    return () => controller.abort();
  }, [selectedStudy]);

  useEffect(() => {
    if (!selectedStudy) {
      setPreviewInstances([]);
      setPreviewIndex(0);
      setPreviewError(false);
      return;
    }

    const controller = new AbortController();

    async function loadPreviewInstance() {
      setPreviewInstances([]);
      setPreviewIndex(0);
      setPreviewError(false);

      try {
        const response = await fetch(
          `${API_BASE}/orthanc/studies/${encodeURIComponent(selectedStudy)}/series`,
          { signal: controller.signal }
        );

        if (!response.ok) throw new Error("Series request failed");

        const data = await response.json();
        const instanceIds = [...new Set((data.series ?? [])
          .flatMap((series) => series.Instances ?? [])
          .filter((id) => typeof id === "string"))];

        if (!controller.signal.aborted) {
          if (instanceIds.length > 0) {
            setPreviewInstances(instanceIds);
            setPreviewIndex(0);
          } else {
            setPreviewError(true);
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") setPreviewError(true);
      }
    }

    loadPreviewInstance();
    return () => controller.abort();
  }, [selectedStudy]);
  async function runClassification(event) {
    event.preventDefault();
    setClassificationError("");
    setClassificationResult(null);

    if (!classificationFile) {
      setClassificationError("Choose a chest X-ray image first.");
      return;
    }

    setClassificationLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", classificationFile);

      const response = await fetch(`${API_BASE}/predict/classification`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          data.detail || `Classification request failed (HTTP ${response.status}).`
        );
      }

      setClassificationResult(data);
    } catch (err) {
      setClassificationError(
        err.message || "Could not run classification. Check that the API is running."
      );
    } finally {
      setClassificationLoading(false);
    }
  }

  async function submitReview(decision) {
    if (!classificationResult) {
      setReviewError("Run a classification before submitting a review.");
      return;
    }

    setReviewSaving(true);
    setReviewMessage("");
    setReviewError("");

    try {
      const response = await fetch(`${API_BASE}/review-decisions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          decision,
          study_id: selectedStudy || null,
          predicted_label:
            classificationResult.predicted_label === 1
              ? "Positive (label 1)"
              : classificationResult.predicted_label === 0
                ? "Negative (label 0)"
                : "Unavailable",
          probability: Number(classificationResult.probability),
          comment: reviewComment.trim() || null,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.detail || `Could not save review (HTTP ${response.status}).`
        );
      }

      setReviewMessage(
        `Review saved successfully: ${data.decision} (record #${data.id}).`
      );
      setReviewComment("");
      await loadAuditLogs();
    } catch (err) {
      setReviewError(
        err.message || "Could not save the review. Check that the API is running."
      );
    } finally {
      setReviewSaving(false);
    }
  }

  useEffect(() => {
  loadAuditLogs();
  }, [loadAuditLogs]);

  const details = studyDetails?.MainDicomTags ?? {};
  const patient = studyDetails?.PatientMainDicomTags ?? {};

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#dashboard" aria-label="MedVision home">
          <span className="brand-icon">✳</span>
          <span>
            <strong>MedVision</strong>
            <small>IMAGING INTELLIGENCE</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="navigation">
          <a className="nav-item active" href="#dashboard">
            <span>▦</span> Dashboard
          </a>
          <a className="nav-item" href="#studies">
            <span>▤</span> Imaging studies
          </a>
          <a className="nav-item" href="#ai-analysis">
            <span>✧</span> AI analysis
          </a>
          <a className="nav-item" href="#review">
            <span>☷</span> Clinical review
          </a>
          <a className="nav-item" href="#audit">
            <span>◷</span> Audit history
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="environment-card">
            <span className="environment-dot" />
            <div>
              <strong>Development mode</strong>
              <small>Local environment</small>
            </div>
          </div>
          <p>EDUCATIONAL PROTOTYPE</p>
          <p className="sidebar-version">MedVision · v1.0</p>
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <strong>Dashboard</strong>
          </div>
          <div className="topbar-right">
            <span className="prototype-badge">DEMO ENVIRONMENT</span>
            <div className="avatar">MD</div>
          </div>
        </header>

        <section className="page-content">
          <div className="welcome-row">
            <div>
              <div className="eyebrow">MEDICAL IMAGING WORKSPACE</div>
              <h1>Imaging dashboard</h1>
              <p className="subtitle">
                Explore imaging studies and review experimental AI outputs.
              </p>
            </div>
            <button
              className="primary-button"
              onClick={loadStudies}
              disabled={loading}
            >
              <span className={loading ? "spin" : ""}>↻</span>
              {loading ? "Refreshing…" : "Refresh studies"}
            </button>
          </div>

          <div className="notice-banner">
            <span className="notice-icon">i</span>
            <p>
              <strong>Research and education only.</strong> AI outputs are
              experimental and must not be used for diagnosis or treatment.
            </p>
          </div>

          <section className="stats-grid" aria-label="Study statistics">
            <article className="stat-card">
              <div className="stat-top">
                <span>Total studies</span>
                <span className="stat-icon purple">▤</span>
              </div>
              <div className="stat-value">
                {loading ? "—" : studies.length}
              </div>
              <div className="stat-caption">Studies indexed in Orthanc</div>
            </article>

            <article className="stat-card">
              <div className="stat-top">
                <span>Selected study</span>
                <span className="stat-icon blue">⌕</span>
              </div>
              <div className="stat-value small-value">
                {selectedStudy ? "Available" : "None"}
              </div>
              <div className="stat-caption">
                {selectedStudy ? "Ready to inspect" : "Select a study below"}
              </div>
            </article>

            <article className="stat-card">
              <div className="stat-top">
                <span>AI classification</span>
                <span className="stat-icon green">✧</span>
              </div>
              <div className="stat-value small-value">Experimental</div>
              <div className="stat-caption">
                Not a clinically validated system
              </div>
            </article>

            <article className="stat-card">
              <div className="stat-top">
                <span>Service status</span>
                <span className="stat-icon green">⌁</span>
              </div>
              <div className="stat-value small-value">
                {error ? "Check service" : loading ? "Checking…" : "Connected"}
              </div>
              <div className="stat-caption">
                {error ? "API request failed" : "Study service response received"}
              </div>
            </article>
          </section>

          <section className="workspace-grid" id="studies">
            <article className="panel studies-panel">
              <div className="panel-header">
                <div>
                  <h2>Imaging studies</h2>
                  <p>Studies retrieved from your local Orthanc server</p>
                </div>
                <span className="count-badge">{studies.length}</span>
              </div>

              {loading && (
                <div className="empty-state">
                  <div className="loading-spinner" />
                  <strong>Loading imaging studies…</strong>
                </div>
              )}

              {!loading && error && (
                <div className="error-state">
                  <strong>Connection unsuccessful</strong>
                  <p>{error}</p>
                  <button className="secondary-button" onClick={loadStudies}>
                    Try again
                  </button>
                </div>
              )}

              {!loading && !error && studies.length === 0 && (
                <div className="empty-state">
                  <span className="empty-icon">▤</span>
                  <strong>No studies found</strong>
                  <p>Upload a sample study to your local Orthanc server.</p>
                </div>
              )}

              {!loading && !error && studies.length > 0 && (
                <div className="study-list">
                  {studies.map((studyId, index) => (
                    <button
                      className={`study-row ${
                        selectedStudy === studyId ? "selected" : ""
                      }`}
                      key={studyId}
                      onClick={() => setSelectedStudy(studyId)}
                    >
                      <span className="study-thumbnail">▧</span>
                      <span className="study-row-info">
                        <strong>Imaging study {index + 1}</strong>
                        <small title={studyId}>{formatStudyId(studyId)}</small>
                      </span>
                      <span className="study-row-status">
                        {selectedStudy === studyId ? "Selected" : "View"}
                      </span>
                    </button>
                  ))}
                </div>
              )}

              <div className="panel-footer">
                <span>
                  {lastUpdated
                    ? `Updated ${lastUpdated.toLocaleTimeString()}`
                    : "Waiting for data"}
                </span>
                <span>Local PACS connection</span>
              </div>
            </article>

            <article className="panel details-panel">
              <div className="panel-header">
                <div>
                  <h2>Study details</h2>
                  <p>Metadata supplied by Orthanc</p>
                </div>
                <span className="metadata-tag">DICOM</span>
              </div>

              {!selectedStudy && !loading && (
                <div className="empty-state">
                  <span className="empty-icon">⌕</span>
                  <strong>Select an imaging study</strong>
                  <p>Choose a study from the list to inspect its metadata.</p>
                </div>
              )}

              {detailsLoading && (
                <div className="empty-state">
                  <div className="loading-spinner" />
                  <strong>Loading study details…</strong>
                </div>
              )}

              {detailsError && (
                <div className="error-state">
                  <p>{detailsError}</p>
                </div>
              )}

              {studyDetails && !detailsLoading && (
                <>
                  <div className="detail-heading">
                    <span className="detail-modality">
                      {details.Modality || "IMAGING"}
                    </span>
                    <div>
                      <strong>
                        {details.StudyDescription || "Imaging examination"}
                      </strong>
                      <small>{details.StudyDate || "Date unavailable"}</small>
                    </div>
                  </div>

                  <section className="dicom-preview" aria-label="DICOM image preview">
                    <div className="preview-heading">
                      <strong>Image preview</strong>
                      <span className="metadata-tag">DICOM</span>
                    </div>

                    {previewInstances.length > 0 ? (
                      <>
                        <img
                          className="dicom-preview-image"
                          src={`${API_BASE}/orthanc/instances/${encodeURIComponent(previewInstances[previewIndex])}/preview`}
                          alt={`DICOM image ${previewIndex + 1} of ${previewInstances.length}`}
                          onError={() => setPreviewError(true)}
                        />
                        <div className="slice-controls">
                          <button
                            className="secondary-button"
                            disabled={previewIndex === 0}
                            onClick={() => {
                              setPreviewError(false);
                              setPreviewIndex((index) => Math.max(0, index - 1));
                            }}
                          >
                            Previous
                          </button>
                          <span>Image {previewIndex + 1} of {previewInstances.length}</span>
                          <button
                            className="secondary-button"
                            disabled={previewIndex >= previewInstances.length - 1}
                            onClick={() => {
                              setPreviewError(false);
                              setPreviewIndex((index) => Math.min(previewInstances.length - 1, index + 1));
                            }}
                          >
                            Next
                          </button>
                        </div>
                        {previewError && (
                          <p className="preview-disclaimer">
                            This image could not be loaded. Try another image.
                          </p>
                        )}
                      </>
                    ) : (
                      <div className="preview-placeholder">
                        {previewError
                          ? "No image preview is available for this study."
                          : "Finding images in the selected study..."}
                      </div>
                    )}

                    <p className="preview-disclaimer">
                      Research preview only. Not for clinical diagnosis.
                    </p>
                  </section>
                  <div className="metadata-list">
                    <div>
                      <span>Patient ID</span>
                      <strong>{patient.PatientID || "Not provided"}</strong>
                    </div>
                    <div>
                      <span>Patient name</span>
                      <strong>{patient.PatientName || "Not provided"}</strong>
                    </div>
                    <div>
                      <span>Study ID</span>
                      <strong>{details.StudyID || "Not provided"}</strong>
                    </div>
                    <div>
                      <span>Institution</span>
                      <strong>
                        {details.InstitutionName || "Not provided"}
                      </strong>
                    </div>
                    <div>
                      <span>Series count</span>
                      <strong>
                        {Array.isArray(studyDetails.Series)
                          ? studyDetails.Series.length
                          : 0}
                      </strong>
                    </div>
                    <div>
                      <span>Study status</span>
                      <strong>
                        {studyDetails.IsStable ? "Stable" : "Processing"}
                      </strong>
                    </div>
                  </div>

                  <div className="details-actions">
                    <button
                      className="secondary-button"
                      onClick={() =>
                        navigator.clipboard?.writeText(selectedStudy)
                      }
                    >
                      Copy study ID
                    </button>
                    <span className="metadata-tag">Metadata only</span>
                  </div>
                </>
              )}

              <div className="panel-footer">
                <span>Source: Orthanc DICOM server</span>
                <span>Not a diagnosis</span>
              </div>
            </article>
          </section>

          <section className="panel" id="ai-analysis" style={{ marginTop: 24 }}>
            <div className="panel-header">
              <div>
                <h2>Experimental chest X-ray classification</h2>
                <p>Upload an image to request a prediction from the local ResNet18 model.</p>
              </div>
              <span className="metadata-tag">AI DEMO</span>
            </div>

            <form onSubmit={runClassification} style={{ padding: 20 }}>
              <label htmlFor="classification-image" style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
                Chest X-ray image
              </label>
              <input
                id="classification-image"
                type="file"
                accept="image/*"
                onChange={(event) => {
                  setClassificationFile(event.target.files?.[0] ?? null);
                  setClassificationResult(null);
                  setClassificationError("");
                }}
                style={{ display: "block", width: "100%", marginBottom: 16 }}
              />
              <button
                className="primary-button"
                type="submit"
                disabled={classificationLoading || !classificationFile}
              >
                {classificationLoading ? "Analyzing…" : "Run experimental classification"}
              </button>
            </form>

            {classificationError && (
              <div className="error-state" role="alert" style={{ margin: "0 20px 20px" }}>
                <strong>Classification could not be completed</strong>
                <p>{classificationError}</p>
              </div>
            )}

            {classificationResult && (
              <div style={{ margin: "0 20px 20px", padding: 20, border: "1px solid #d9deea", borderRadius: 12 }}>
                <h3 style={{ marginTop: 0 }}>Model response</h3>
                <div className="metadata-list">
                  <div>
                    <span>Target</span>
                    <strong>{classificationResult.target || "Not provided"}</strong>
                  </div>
                  <div>
                    <span>Predicted label</span>
                    <strong>
                      {classificationResult.predicted_label === 1
                        ? "Positive (label 1)"
                        : classificationResult.predicted_label === 0
                          ? "Negative (label 0)"
                          : "Unavailable"}
                    </strong>
                  </div>
                  <div>
                    <span>Model probability</span>
                    <strong>
                      {Number.isFinite(Number(classificationResult.probability))
                        ? `${(Number(classificationResult.probability) * 100).toFixed(2)}%`
                        : "Unavailable"}
                    </strong>
                  </div>
                  <div>
                    <span>Decision threshold</span>
                    <strong>
                      {Number.isFinite(Number(classificationResult.threshold))
                        ? `${(Number(classificationResult.threshold) * 100).toFixed(0)}%`
                        : "Unavailable"}
                    </strong>
                  </div>
                </div>
                <p className="preview-disclaimer" style={{ marginTop: 16 }}>
                  {classificationResult.notice ||
                    "Experimental output only. Not validated for clinical diagnosis or treatment."}
                </p>
                                <div style={{ marginTop: 20, borderTop: "1px solid #d9deea", paddingTop: 16 }}>
                  <h3>Reviewer decision</h3>
                  <label htmlFor="review-comment">Review comment (optional)</label>
                  <textarea
                    id="review-comment"
                    value={reviewComment}
                    onChange={(event) => setReviewComment(event.target.value)}
                    placeholder="Add a brief explanation for your decision"
                    rows={3}
                    style={{
                      display: "block",
                      width: "100%",
                      boxSizing: "border-box",
                      marginTop: 8,
                      marginBottom: 12,
                      padding: 10,
                      border: "1px solid #d9deea",
                      borderRadius: 8,
                    }}
                  />
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    <button
                      type="button"
                      disabled={reviewSaving}
                      onClick={() => submitReview("approved")}
                    >
                      {reviewSaving ? "Saving..." : "Approve result"}
                    </button>
                    <button
                      type="button"
                      disabled={reviewSaving}
                      onClick={() => submitReview("rejected")}
                    >
                      {reviewSaving ? "Saving..." : "Reject result"}
                    </button>
                  </div>
                  {reviewMessage && (
                    <p role="status" style={{ color: "green" }}>{reviewMessage}</p>
                  )}
                  {reviewError && (
                    <p role="alert" style={{ color: "crimson" }}>{reviewError}</p>
                  )}
                </div>
              </div>
            )}

            <div className="panel-footer">
              <span>Local FastAPI inference service</span>
              <span>Research and education only</span>
            </div>
          </section>

          <section id="audit" className="panel" style={{ marginTop: 24 }}>
            <div style={{ padding: 20 }}>
              <h2 style={{ color: "#173b70" }}>Audit History</h2>
              <p>Recent reviewer decisions saved by the local audit service.</p>

              <button
                type="button"
                onClick={loadAuditLogs}
                disabled={auditLoading}
              >
                {auditLoading ? "Loading..." : "Refresh history"}
              </button>

              {auditError && (
                <p role="alert" style={{ color: "crimson" }}>{auditError}</p>
              )}

              {auditLoading && auditLogs.length === 0 && (
                <p>Loading audit records...</p>
              )}

              {!auditLoading && !auditError && auditLogs.length === 0 && (
                <p>No review decisions have been recorded yet.</p>
              )}

              {auditLogs.length > 0 && (
                <div style={{ overflowX: "auto", marginTop: 16 }}>
                  <table style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    textAlign: "left",
                  }}>
                    <thead>
                      <tr>
                        {["ID", "Timestamp (UTC)", "Decision", "Study ID",
                          "Prediction", "Probability", "Comment"].map((heading) => (
                          <th key={heading} style={{
                            padding: 10,
                            borderBottom: "2px solid #d9deea",
                          }}>
                            {heading}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {auditLogs.map((log) => (
                        <tr key={log.id}>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                            {log.id}
                          </td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                            {log.timestamp_utc
                              ? new Date(log.timestamp_utc).toLocaleString()
                              : "Unavailable"}
                          </td>
                          <td style={{
                            padding: 10,
                            borderBottom: "1px solid #eee",
                            color: log.decision === "approved" ? "green" : "crimson",
                            fontWeight: 600,
                          }}>
                            {log.decision}
                          </td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                            {log.study_id || "—"}
                          </td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                            {log.predicted_label || "—"}
                          </td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                            {log.probability == null
                              ? "—"
                              : `${(Number(log.probability) * 100).toFixed(2)}%`}
                          </td>
                          <td style={{
                            padding: 10,
                            borderBottom: "1px solid #eee",
                            whiteSpace: "pre-wrap",
                          }}>
                            {log.comment || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>

          <footer className="page-footer">
            <span>MedVision · Medical Imaging Diagnosis</span>
            <span>Local demonstration · No clinical use</span>
          </footer>
        </section>
      </main>
    </div>
  );
}
