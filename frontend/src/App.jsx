import { useRef, useState, useEffect, useMemo } from "react";
import Webcam from "react-webcam";
import "./App.css";

const API_BASE = "http://localhost:5000";

const resolveThumb = (u) => {
  if (!u) return "";
  return /^https?:\/\//i.test(u)
    ? u
    : `${API_BASE}${u.startsWith("/") ? u : `/${u}`}`;
};

// key MUST match backend: artwork_id | id | "title|artist"
const keyOf = (a) =>
  (a?.artwork_id && String(a.artwork_id)) ||
  (a?.id && String(a.id)) ||
  `${(a?.title || "").trim()}|${(a?.artist || "").trim()}`;

export default function App() {
  const webcamRef = useRef(null);
  const boxRef = useRef(null);
  const fileInputRef = useRef(null);

  const [candidates, setCandidates] = useState([]);
  const [bbox, setBbox] = useState(null);
  const [loading, setLoading] = useState(false);

  // summary-related state
  const [summary, setSummary] = useState("");   // streamed text shown in the panel
  const [selected, setSelected] = useState(null);
  const [sumMap, setSumMap] = useState({});     // {key: fullSummaryText}
  const [sumLoading, setSumLoading] = useState({}); // {key: boolean}

  const [error, setError] = useState("");

  const [facingMode, setFacingMode] = useState("user");        // "user" | "environment"
  const [torchOn, setTorchOn] = useState(false);                // best-effort torch toggle
  const webcamKey = useMemo(() => `cam-${facingMode}`, [facingMode]); // remount on flip

  // --- Overlay box in px from normalized bbox ---
  const overlayStyle = (() => {
    if (!bbox) return { display: "none" };
    const container = boxRef.current;
    if (!container) return { display: "none" };
    const rect = container.getBoundingClientRect();
    const left = bbox.x * rect.width;
    const top = bbox.y * rect.height;
    const width = bbox.w * rect.width;
    const height = bbox.h * rect.height;
    return { left, top, width, height };
  })();

  // --- Streaming summarizer (SSE over POST) ---
  const streamSummaryFor = async (art) => {
    const k = keyOf(art);
    if (!k) return;

    // if already loaded, just show cached
    if (sumMap[k]) {
      setSummary(sumMap[k]);
      return;
    }

    setSumLoading((s) => ({ ...s, [k]: true }));
    setSummary(""); // clear panel to start streaming

    try {
      const res = await fetch(`${API_BASE}/api/summarize/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream", // important for SSE
        },
        body: JSON.stringify({
          artwork_id: art.artwork_id || art.id || k,
          title: art.title,
          artist: art.artist,
          year: art.year,
          source: art.source,
          nocache: true, // force fresh once; remove later to reuse cache
        }),
      });

      if (!res.ok || !res.body) throw new Error("stream failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let full = "";

      const handleEvent = (evt, dataStr) => {
        if (evt === "token") {
          try {
            const obj = JSON.parse(dataStr);
            const piece = obj?.text || "";
            if (piece) {
              full += piece;
              setSummary((s) => s + piece);
            }
          } catch {
            // ignore
          }
        } else if (evt === "done") {
          setSumMap((m) => ({ ...m, [k]: full }));
        } else if (evt === "error") {
          try {
            const obj = JSON.parse(dataStr);
            setSummary(`⚠️ ${obj?.message || "Error"}`);
          } catch {
            setSummary("⚠️ Error");
          }
        }
      };

      // Parse SSE frames: "event: X\ndata: Y\n\n"
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          let evt = "message";
          let data = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) evt = line.slice(6).trim();
            else if (line.startsWith("data:")) {
              data += (data ? "\n" : "") + line.slice(5).trim();
            }
          }
          handleEvent(evt, data);
        }
      }
    } catch (e) {
      setSummary(`⚠️ ${e.message || "Stream error"}`);
    } finally {
      setSumLoading((s) => ({ ...s, [k]: false }));
    }
  };

  // --- Core: send a Blob to backend ---
  const analyzeBlob = async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "snapshot.jpg");

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || "Analyze failed");

    const cands = data.candidates || [];
    setCandidates(cands);
    setBbox(data.bbox || null);

    if (cands.length === 0) {
      setError("No close matches. Try getting closer, improve lighting, or center the artwork.");
    } else {
      // summarize on demand via streamSummaryFor (no prefetch to save tokens)
    }
  };

  // --- Capture from webcam and analyze ---
  const captureAndSend = async () => {
    if (!webcamRef.current) return;
    setLoading(true);
    setError("");
    setSelected(null);
    setSummary("");

    try {
      const imageSrc = webcamRef.current.getScreenshot({ width: 1280, height: 720 });
      if (!imageSrc) throw new Error("Could not capture webcam frame");
      const resp = await fetch(imageSrc);
      const blob = await resp.blob();
      await analyzeBlob(blob);
    } catch (e) {
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  // --- Upload image from device (mobile/desktop) ---
  const onPickFile = () => fileInputRef.current?.click();
  const onFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setSelected(null);
    setSummary("");
    setError("");
    try {
      await analyzeBlob(file);
    } catch (e2) {
      setError(e2.message || "Upload analyze failed.");
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  };

  // --- Flip camera (front/back) ---
  const flipCamera = () => setFacingMode((m) => (m === "user" ? "environment" : "user"));

  // --- Torch (best-effort) ---
  const toggleTorch = async () => {
    try {
      const stream = webcamRef.current?.stream;
      const track = stream?.getVideoTracks?.()[0];
      if (!track) throw new Error("No camera track");
      const capabilities = track.getCapabilities?.() || {};
      if (!("torch" in capabilities)) throw new Error("Torch not supported");
      const next = !torchOn;
      await track.applyConstraints({ advanced: [{ torch: next }] });
      setTorchOn(next);
    } catch {
      setTorchOn(false);
    }
  };

  // --- Show details for a chosen candidate (stream) ---
  const getSummary = (art) => {
    setSelected(art);
    setSummary(""); // show streaming immediately
    streamSummaryFor(art);
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div>
          <div className="title">ArtSpot 🎬</div>
          <div className="subtitle">Cinema-dark, responsive, and robust.</div>
        </div>
        <div className="row">
          <button className="btn btn-ghost" onClick={flipCamera} title="Flip camera">
            🔄 Flip
          </button>
          <button className="btn btn-ghost" onClick={() => window.location.reload()}>
            Reset
          </button>
        </div>
      </header>

      {/* Main grid */}
      <div className="grid">
        {/* Left: Camera */}
        <div className="card sticky-md">
          <div ref={boxRef} className="webcam-wrap">
            <Webcam
              key={webcamKey}
              ref={webcamRef}
              className="webcam-video"
              screenshotFormat="image/jpeg"
              videoConstraints={{ facingMode }}
            />
            {/* Detection overlay */}
            <div className="overlay-box" style={overlayStyle} />
          </div>

          <div className="row" style={{ marginTop: 12, flexWrap: "wrap" }}>
            <button className="btn btn-primary" onClick={captureAndSend} disabled={loading}>
              {loading ? "Analyzing…" : "Analyze"}
            </button>
            <button className="btn" onClick={onPickFile} disabled={loading} title="Upload from device">
              🖼️ Upload
            </button>
            <button className="btn" onClick={toggleTorch} disabled={loading} title="Toggle torch (if supported)">
              {torchOn ? "🔦 Torch On" : "🔦 Torch"}
            </button>
            <span className="muted">Tip: fill the frame, avoid glare.</span>
          </div>

          {error && (
            <div style={{ marginTop: 10 }} className="summary">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
            <div className="title" style={{ fontSize: 18 }}>Top Matches</div>
            <div className="muted">{candidates.length ? `${candidates.length} found` : "—"}</div>
          </div>

          {/* Loading skeleton */}
          {loading && (
            <div className="results">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="result-item">
                  <div className="skel" style={{ width: 72, height: 72 }} />
                  <div>
                    <div className="skel" style={{ height: 14, width: "60%" }} />
                    <div className="skel" style={{ height: 12, width: "40%", marginTop: 8 }} />
                  </div>
                  <div className="skel" style={{ width: 90, height: 32 }} />
                </div>
              ))}
            </div>
          )}

          {!loading && (
            <>
              {candidates.length === 0 ? (
                <div className="center" style={{ padding: 16 }}>
                  <div className="muted">No results yet. Click <b>Analyze</b> or <b>Upload</b> to begin.</div>
                </div>
              ) : (
                <div className="results">
                  {candidates.map((c, i) => {
                    const k = keyOf(c);
                    const isLoading = !!sumLoading[k] && !sumMap[k];
                    return (
                      <div key={i} className="result-item">
                        <img
                          className="thumb"
                          src={resolveThumb(c.thumbnail || c.image_url)}
                          alt={c.title || "artwork"}
                          loading="lazy"
                          referrerPolicy="no-referrer"
                        />
                        <div>
                          <div className="item-title">
                            {c.title || "Untitled"} {c.artist ? `— ${c.artist}` : ""}
                          </div>
                          <div className="item-sub">
                            {(c.year || "").toString()} · score {Number(c.score).toFixed(3)}
                          </div>
                        </div>
                        <button
                          className="btn btn-ghost"
                          onClick={() => getSummary(c)}
                          disabled={isLoading}
                          title="See details"
                        >
                          {isLoading ? "…" : "Details"}
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}

              {selected && !loading && (
                <div style={{ marginTop: 12 }}>
                  <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
                    <div className="item-title">About: {selected.title}</div>
                    <button className="btn btn-ghost" onClick={() => setSelected(null)}>Close</button>
                  </div>
                  <div className="summary">{summary || "..."}</div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={onFileChange}
      />

      {/* Mobile toolbar (hidden on desktop) */}
      <div className="mobile-toolbar">
        <button className="btn btn-primary" onClick={captureAndSend} disabled={loading} title="Analyze">
          <span className="icon">🎥</span> Analyze
        </button>
        <button className="btn" onClick={flipCamera} disabled={loading} title="Flip camera">
          <span className="icon">🔄</span> Flip
        </button>
        <button className="btn" onClick={toggleTorch} disabled={loading} title="Torch (if supported)">
          <span className="icon">🔦</span> {torchOn ? "On" : "Torch"}
        </button>
        <button className="btn" onClick={onPickFile} disabled={loading} title="Upload photo">
          <span className="icon">🖼️</span> Upload
        </button>
      </div>
    </div>
  );
}
