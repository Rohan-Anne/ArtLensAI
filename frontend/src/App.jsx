import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import Webcam from "react-webcam";
import "./App.css";

const API_BASE = "http://localhost:5000";

/* ---------------- Auth helper (JWT access + auto-refresh) ---------------- */
const AUTH_BASE = `${API_BASE}/api/auth`;

let accessToken = null;
const setAccessToken = (t) => { accessToken = t; };

async function apiFetch(url, opts = {}) {
  const headers = new Headers(opts.headers || {});
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
  const res = await fetch(url, { ...opts, headers, credentials: "include" });

  if (res.status !== 401) return res;

  // Try one silent refresh
  const rr = await fetch(`${AUTH_BASE}/refresh`, { method: "POST", credentials: "include" });
  if (rr.ok) {
    const data = await rr.json();
    if (data?.access_token) setAccessToken(data.access_token);
    const headers2 = new Headers(opts.headers || {});
    if (data?.access_token) headers2.set("Authorization", `Bearer ${data.access_token}`);
    return fetch(url, { ...opts, headers: headers2, credentials: "include" });
  }

  throw new Error("Unauthorized");
}

async function register(email, password) {
  const r = await fetch(`${AUTH_BASE}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ email, password }),
  });
  if (!r.ok) throw new Error((await r.json()).error || "Register failed");
  return true;
}

async function login(email, password) {
  const r = await fetch(`${AUTH_BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ email, password }),
  });
  if (!r.ok) throw new Error((await r.json()).error || "Login failed");
  const data = await r.json();
  setAccessToken(data.access_token);
  return data.user;
}

async function logout() {
  await fetch(`${AUTH_BASE}/logout`, { method: "POST", credentials: "include" });
  setAccessToken(null);
}

/* ---------------- Utilities ---------------- */
const resolveThumb = (u) => {
  if (!u) return "";
  return /^https?:\/\//i.test(u) ? u : `${API_BASE}${u.startsWith("/") ? u : `/${u}`}`;
};

// key MUST match backend: artwork_id | id | "title|artist"
const keyOf = (a) =>
  (a?.artwork_id && String(a.artwork_id)) ||
  (a?.id && String(a.id)) ||
  `${(a?.title || "").trim()}|${(a?.artist || "").trim()}`;

/* ---------- Auth UI ---------- */
function AuthBar({ user, setUser }) {
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [mode, setMode] = useState("login");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const doAuth = async (e) => {
    e.preventDefault();
    setBusy(true); setMsg("");
    try {
      if (mode === "register") {
        await register(email, pw);
        setMsg("Registered! Now login.");
        setMode("login");
      } else {
        const u = await login(email, pw);
        setUser(u);
        setMsg("");
      }
    } catch (err) {
      setMsg(`⚠️ ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  if (user) {
    return (
      <div className="row" style={{ gap: 8 }}>
        <div className="muted">Signed in as <b>{user.email}</b></div>
        <button className="btn btn-ghost" onClick={async () => { await logout(); setUser(null); }}>
          Logout
        </button>
      </div>
    );
  }

  return (
    <form className="row" onSubmit={doAuth} style={{ gap: 8 }}>
      <input className="input" placeholder="email" value={email} onChange={(e)=>setEmail(e.target.value)} />
      <input className="input" type="password" placeholder="password" value={pw} onChange={(e)=>setPw(e.target.value)} />
      <button className="btn btn-primary" disabled={busy}>
        {mode === "login" ? "Login" : "Register"}
      </button>
      <button type="button" className="btn btn-ghost" onClick={()=>setMode(mode==="login"?"register":"login")}>
        {mode==="login"?"Need an account?":"Have an account?"}
      </button>
      {msg && <span className="muted">{msg}</span>}
    </form>
  );
}

/* ---------- NEW: Chat panel component ---------- */
function ChatPanel({ artwork, thread, onSend, streaming }) {
  const [input, setInput] = useState("");
  const endRef = useRef(null);

  const submit = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || streaming) return;
    onSend(text);
    setInput("");
  };

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [thread, streaming]);

  return (
    <div className="chat-card">
      <div className="chat-header">
        <div className="item-title">Ask more about: {artwork?.title || "Artwork"}</div>
        <div className="muted">Follow-up Q&A (teacher mode)</div>
      </div>

      <div className="chat-log">
        {(thread?.messages || []).map((m, i) => (
          <div key={i} className={`chat-msg ${m.role === "user" ? "right" : "left"}`}>
            <div className="bubble">{m.content}</div>
          </div>
        ))}
        {streaming && (
          <div className="chat-msg left">
            <div className="bubble bubble-stream">...</div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="chat-input-row" onSubmit={submit}>
        <input
          className="chat-input"
          placeholder="Ask a question about this piece…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming}
        />
        <button className="btn btn-primary" disabled={streaming || !input.trim()}>
          {streaming ? "Thinking…" : "Send"}
        </button>
      </form>
    </div>
  );
}

export default function App() {
  const webcamRef = useRef(null);
  const boxRef = useRef(null);
  const fileInputRef = useRef(null);

  const [user, setUser] = useState(null);

  const [candidates, setCandidates] = useState([]);
  const [bbox, setBbox] = useState(null);
  const [loading, setLoading] = useState(false);

  // summary-related state
  const [summary, setSummary] = useState("");   // streamed text shown in the panel
  const [selected, setSelected] = useState(null);
  const [sumLoading, setSumLoading] = useState({}); // {key: boolean}

  // ---------- NEW: per-artwork chat state ----------
  // chatMap: { [key]: { messages: [{role, content}], streaming: boolean } }
  const [chatMap, setChatMap] = useState({});

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

    setSumLoading((s) => ({ ...s, [k]: true }));
    setSummary(""); // clear panel to start streaming

    try {
      const res = await apiFetch(`${API_BASE}/api/summarize/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({
          artwork_id: art.artwork_id || art.id || k,
          title: art.title,
          artist: art.artist,
          year: art.year,
          source: art.source,
          nocache: true, // explicitly bypass any caches
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
          } catch { /* ignore */ }
        } else if (evt === "done") {
          // nothing
        } else if (evt === "error") {
          try {
            const obj = JSON.parse(dataStr);
            setSummary(`⚠️ ${obj?.message || "Error"}`);
          } catch {
            setSummary("⚠️ Error");
          }
        }
      };

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
            else if (line.startsWith("data:")) data += (data ? "\n" : "") + line.slice(5).trim();
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

    const res = await apiFetch(`${API_BASE}/api/analyze`, {
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
      // Wait for explicit "Details" click -> streamSummaryFor
    }
  };

  // --- Capture from webcam and analyze ---
  const captureAndSend = async () => {
    if (!webcamRef.current) return;
    setLoading(true);
    setError("");
    setSelected(null);
    setSummary("");
    setChatMap({}); // reset any prior threads on new scan

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
    setChatMap({});
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
      const track = stream?.getVideoTracks?.[0];
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
    // initialize empty chat if not existing
    const k = keyOf(art);
    setChatMap((m) => (m[k] ? m : { ...m, [k]: { messages: [], streaming: false }}));
    streamSummaryFor(art);
  };

  // ---------- NEW: stream an assistant reply for the current artwork ----------
  const askStream = useCallback(async (questionText) => {
    if (!selected) return;
    const k = keyOf(selected);
    const thread = chatMap[k] || { messages: [], streaming: false };

    // optimistic update: add user msg + placeholder assistant msg
    const newUserMsg = { role: "user", content: questionText };
    setChatMap((m) => ({
      ...m,
      [k]: { messages: [...(thread.messages || []), newUserMsg], streaming: true }
    }));

    try {
      const res = await apiFetch(`${API_BASE}/api/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({
          artwork: {
            artwork_id: selected.artwork_id || selected.id || k,
            title: selected.title,
            artist: selected.artist,
            year: selected.year,
            source: selected.source,
          },
          messages: (thread.messages || []).concat([newUserMsg]),
          context: summary || "", // ground chat on the currently streamed FFCC
        }),
      });

      if (!res.ok || !res.body) throw new Error("chat stream failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const pushAssistant = (chunk) => {
        setChatMap((m) => {
          const cur = m[k] || { messages: [], streaming: true };
          const msgs = cur.messages.slice();
          if (msgs.length && msgs[msgs.length - 1].role === "assistant") {
            msgs[msgs.length - 1] = { role: "assistant", content: (msgs[msgs.length - 1].content || "") + chunk };
          } else {
            msgs.push({ role: "assistant", content: chunk });
          }
          return { ...m, [k]: { messages: msgs, streaming: true } };
        });
      };

      const handleEvent = (evt, dataStr) => {
        if (evt === "token") {
          try {
            const obj = JSON.parse(dataStr);
            const piece = obj?.text || "";
            if (piece) pushAssistant(piece);
          } catch { /* ignore */ }
        } else if (evt === "done") {
          setChatMap((m) => {
            const cur = m[k] || { messages: [], streaming: false };
            return { ...m, [k]: { ...cur, streaming: false } };
          });
        } else if (evt === "error") {
          setChatMap((m) => {
            const cur = m[k] || { messages: [], streaming: false };
            const msgs = cur.messages.concat([{ role: "assistant", content: "⚠️ Error answering your question." }]);
            return { ...m, [k]: { messages: msgs, streaming: false } };
          });
        }
      };

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
            else if (line.startsWith("data:")) data += (data ? "\n" : "") + line.slice(5).trim();
          }
          handleEvent(evt, data);
        }
      }
    } catch (e) {
      setChatMap((m) => {
        const cur = m[k] || { messages: [], streaming: false };
        const msgs = cur.messages.concat([{ role: "assistant", content: `⚠️ ${e.message || "Stream error"}` }]);
        return { ...m, [k]: { messages: msgs, streaming: false } };
      });
    }
  }, [selected, chatMap, summary]);

  const threadForSelected = (() => {
    if (!selected) return null;
    return chatMap[keyOf(selected)] || { messages: [], streaming: false };
  })();

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div>
          <div className="title">ArtLens AI 🎬</div>
          <div className="subtitle">Cinema-dark, responsive, and robust.</div>
        </div>
        <div className="col" style={{ gap: 8 }}>
          <AuthBar user={user} setUser={setUser} />
          <div className="row">
            <button className="btn btn-ghost" onClick={flipCamera} title="Flip camera">
              🔄 Flip
            </button>
            <button className="btn btn-ghost" onClick={() => window.location.reload()}>
              Reset
            </button>
          </div>
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
                    const isLoading = !!sumLoading[k];
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

                  {/* Summary panel */}
                  <div className="summary">{summary || "..."}</div>

                  {/* ---------- NEW: Chat panel appears after summary ---------- */}
                  <div style={{ marginTop: 12 }}>
                    <ChatPanel
                      artwork={selected}
                      thread={threadForSelected}
                      streaming={threadForSelected?.streaming}
                      onSend={askStream}
                    />
                  </div>
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

