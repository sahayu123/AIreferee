"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";

type Mode = "video" | "image";
type JobState = "idle" | "ready" | "uploading" | "running" | "complete" | "error";

type TimelinePoint = {
  time: number;
  scene_foul: number;
  pair_crop_foul: number;
  evidence: number;
};

type AnalysisResult = {
  verdict: string;
  confidence: number;
  reason: string;
  peak_time?: number;
  frames_analyzed?: number;
  player_tracks?: number;
  ball_tracks?: number;
  timeline?: TimelinePoint[];
  metrics?: {
    roboflow_foul?: number;
    limb_gap?: number | null;
    contact_type?: string;
    contact_probability?: number;
    tackle_type?: string;
    tackle_probability?: number;
    mlp_foul?: number;
    metric_contact?: string;
    handball_probability?: number;
    handball_quality?: number;
    handball_status?: string;
    handball_prediction?: string;
    handball_ball_rate?: number;
    handball_player_rate?: number;
    handball_pose_rate?: number;
    handball_min_arm_distance?: number | null;
    handball_proximity_override?: boolean;
    handball_proximity_threshold?: number;
    handball_frames?: number;
  };
  peak_frame_url?: string;
  annotated_video_url?: string;
  handball_overlay_url?: string;
  report?: string;
};

const API =
  typeof window !== "undefined" &&
  !["localhost", "127.0.0.1", "::1"].includes(window.location.hostname)
    ? ""
    : "http://localhost:8200";

const modelStack = [
  ["Handball Project", "Ball trajectory + arm-angle geometry"],
  ["YOLO11m", "Metal-accelerated tracking"],
  ["RTMW-X", "CoreML player pose"],
  ["Roboflow", "Every-frame foul detector"],
  ["ResNet34 MLP", "MPS scene classification"],
  ["SAM 2 + MoGe-2", "MPS metric 3D verification"],
];

function formatPercent(value?: number) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

function verdictTone(verdict = "") {
  if (verdict.startsWith("FOUL")) return "foul";
  if (verdict.startsWith("NO FOUL")) return "clear";
  return "review";
}

function EvidenceChart({ points }: { points: TimelinePoint[] }) {
  const width = 900;
  const height = 180;
  const pad = 18;
  const maxT = Math.max(points.at(-1)?.time ?? 1, 1);
  const path = (key: keyof Pick<TimelinePoint, "scene_foul" | "pair_crop_foul" | "evidence">) =>
    points
      .map((point, index) => {
        const x = pad + (point.time / maxT) * (width - pad * 2);
        const y = height - pad - point[key] * (height - pad * 2);
        return `${index ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <div className="chart-wrap" aria-label="Frame-by-frame foul evidence chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {[0.25, 0.5, 0.75].map((value) => (
          <line
            key={value}
            x1={pad}
            x2={width - pad}
            y1={height - pad - value * (height - pad * 2)}
            y2={height - pad - value * (height - pad * 2)}
            className="grid-line"
          />
        ))}
        <line
          x1={pad}
          x2={width - pad}
          y1={height - pad - 0.97 * (height - pad * 2)}
          y2={height - pad - 0.97 * (height - pad * 2)}
          className="override-line"
        />
        <path d={path("scene_foul")} className="chart-line scene" />
        <path d={path("pair_crop_foul")} className="chart-line crop" />
        <path d={path("evidence")} className="chart-line evidence" />
      </svg>
      <div className="chart-legend">
        <span><i className="dot scene" /> Full frame</span>
        <span><i className="dot crop" /> Pair crop</span>
        <span><i className="dot evidence" /> Fused evidence</span>
        <span><i className="dash" /> 97% override</span>
      </div>
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("video");
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [state, setState] = useState<JobState>("idle");
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("Waiting for media");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [fileUrl]);

  const accept = mode === "video" ? "video/*" : "image/*";
  const working = state === "uploading" || state === "running";
  const tone = verdictTone(result?.verdict);

  const mediaMeta = useMemo(() => {
    if (!file) return "";
    const size = file.size / 1024 / 1024;
    return `${file.name} · ${size < 1 ? `${Math.round(size * 1024)} KB` : `${size.toFixed(1)} MB`}`;
  }, [file]);

  function chooseFile(next: File | null) {
    if (!next) return;
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFile(next);
    setFileUrl(URL.createObjectURL(next));
    setResult(null);
    setProgress(0);
    setStage("Ready to analyze");
    setState("ready");
  }

  function handleInput(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function pollJob(jobId: string) {
    const response = await fetch(`${API}/api/jobs/${jobId}`);
    if (!response.ok) throw new Error("The local analysis service stopped responding.");
    const job = await response.json();
    setProgress(job.progress ?? 0);
    setStage(job.stage ?? "Analyzing");
    if (job.status === "complete") {
      if (pollRef.current) clearInterval(pollRef.current);
      setResult(job.result);
      setState("complete");
      setProgress(100);
    } else if (job.status === "error") {
      if (pollRef.current) clearInterval(pollRef.current);
      setStage(job.error || "Analysis failed");
      setState("error");
    }
  }

  async function analyze() {
    if (!file || !apiKey.trim()) return;
    setState("uploading");
    setResult(null);
    setProgress(3);
    setStage("Sending media to the local referee");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("api_key", apiKey.trim());
      const response = await fetch(`${API}/api/analyze/${mode}`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Could not begin analysis.");
      }
      const { job_id: jobId } = await response.json();
      setState("running");
      await pollJob(jobId);
      pollRef.current = setInterval(() => {
        pollJob(jobId).catch((error) => {
          if (pollRef.current) clearInterval(pollRef.current);
          setStage(error.message);
          setState("error");
        });
      }, 1200);
    } catch (error) {
      setState("error");
      setStage(error instanceof Error ? error.message : "Analysis failed");
    }
  }

  function reset() {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    if (pollRef.current) clearInterval(pollRef.current);
    setFile(null);
    setFileUrl("");
    setResult(null);
    setState("idle");
    setProgress(0);
    setStage("Waiting for media");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
          </div>
          <div>
            <p className="brand-name">VAIR</p>
            <p className="brand-subtitle">Virtual Artificial Intelligence Referee</p>
          </div>
        </div>
        <div className="topbar-center">
          <span className="version-pill">Handball Project + Foul · GPU</span>
          <span className="local-pill"><i /> Apple Metal</span>
        </div>
        <button className="text-button" onClick={() => setDetailsOpen(!detailsOpen)}>
          {detailsOpen ? "Close system details" : "System details"}
        </button>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="side-section">
            <p className="eyebrow">Analysis type</p>
            <div className="mode-switch" role="group" aria-label="Analysis type">
              <button className={mode === "video" ? "active" : ""} onClick={() => { setMode("video"); reset(); }}>
                <span className="mode-icon">▶</span> Video
              </button>
              <button className={mode === "image" ? "active" : ""} onClick={() => { setMode("image"); reset(); }}>
                <span className="mode-icon">▣</span> Image
              </button>
            </div>
          </div>

          <div className="side-section">
            <p className="eyebrow">Model stack</p>
            <div className="model-list">
              {modelStack.map(([name, description]) => (
                <div className="model-row" key={name}>
                  <i className="status-light" />
                  <div>
                    <strong>{name}</strong>
                    <span>{description}</span>
                  </div>
                  <b>READY</b>
                </div>
              ))}
            </div>
          </div>

          <div className="side-section secure-section">
            <div className="secure-title">
              <p className="eyebrow">Roboflow connection</p>
              <span>PRIVATE</span>
            </div>
            <label htmlFor="api-key">API key</label>
            <div className="key-field">
              <input
                id="api-key"
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                placeholder="Paste private key"
                autoComplete="off"
              />
              <button onClick={() => setShowKey(!showKey)} aria-label={showKey ? "Hide API key" : "Show API key"}>
                {showKey ? "Hide" : "Show"}
              </button>
            </div>
            <p>The key is passed to your local process for this analysis and is not stored.</p>
          </div>

          <div className="system-note">
            <span className="shield">✓</span>
            <p><strong>Parallel referee active</strong>Ball/arm handball geometry · general-foul detector · metric 3D verification</p>
          </div>
        </aside>

        <section className="main-panel">
          {!file ? (
            <div
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <div className="pitch-mark">
                <span className="pitch-ring" />
                <span className="upload-arrow">↑</span>
              </div>
              <p className="kicker">NEW REVIEW</p>
              <h1>Bring the incident into focus.</h1>
              <p className="drop-copy">
                Upload {mode === "video" ? "a match clip" : "a match frame"} for handball and general-foul analysis using the integrated GitHub referee stack.
              </p>
              <button className="primary-button" onClick={() => inputRef.current?.click()}>
                Choose {mode === "video" ? "video clip" : "image"}
              </button>
              <p className="file-hint">
                or drag and drop · {mode === "video" ? "MP4, MOV, M4V or AVI" : "JPG, PNG or WEBP"}
              </p>
              <input ref={inputRef} className="hidden-input" type="file" accept={accept} onChange={handleInput} />
            </div>
          ) : (
            <div className="review-grid">
              <section className="review-stage">
                <div className="stage-header">
                  <div>
                    <p className="kicker">INCIDENT REVIEW</p>
                    <p className="media-name">{mediaMeta}</p>
                  </div>
                  <button className="text-button" onClick={reset}>Replace media</button>
                </div>

                <div className={`media-frame ${mode === "image" ? "image-frame" : ""}`}>
                  {mode === "video" ? (
                    <video src={result?.annotated_video_url ? `${API}${result.annotated_video_url}` : fileUrl} controls />
                  ) : (
                    <img src={result?.peak_frame_url ? `${API}${result.peak_frame_url}` : fileUrl} alt="Uploaded incident" />
                  )}
                  {working && (
                    <div className="analysis-scrim">
                      <div className="scan-line" />
                      <div className="processing-card">
                        <span className="spinner" />
                        <div>
                          <strong>{stage}</strong>
                          <span>{Math.round(progress)}% complete</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {result?.timeline?.length ? (
                  <div className="evidence-section">
                    <div className="section-heading">
                      <div>
                        <p className="eyebrow">Temporal evidence</p>
                        <h2>Frame-by-frame confidence</h2>
                      </div>
                      <span>Peak at {result.peak_time?.toFixed(2)}s</span>
                    </div>
                    <EvidenceChart points={result.timeline} />
                  </div>
                ) : (
                  <div className="preflight-row">
                    <div><span>01</span><p><strong>Track</strong>Players + ball</p></div>
                    <div><span>02</span><p><strong>Inspect</strong>Every frame</p></div>
                    <div><span>03</span><p><strong>Verify</strong>Metric 3D</p></div>
                  </div>
                )}
              </section>

              <aside className="decision-panel">
                {result ? (
                  <>
                    <div className={`verdict-card ${tone}`}>
                      <p className="eyebrow">Final decision</p>
                      <div className="verdict-score">
                        <strong>{formatPercent(result.confidence)}</strong>
                        <span>confidence</span>
                      </div>
                      <h2>{result.verdict.split("—")[0].trim()}</h2>
                      <p>{result.reason || result.verdict.split("—")[1]?.trim()}</p>
                    </div>

                    <div className="metric-grid">
                      <div><span>Peak frame</span><strong>{result.peak_time?.toFixed(2) ?? "—"}s</strong></div>
                      <div><span>Frames analyzed</span><strong>{result.frames_analyzed ?? "1"}</strong></div>
                      <div><span>Player tracks</span><strong>{result.player_tracks ?? "—"}</strong></div>
                      <div><span>Ball tracks</span><strong>{result.ball_tracks ?? "—"}</strong></div>
                    </div>

                    <div className="evidence-list">
                      <p className="eyebrow">Evidence breakdown</p>
                      {[
                        ["Handball Project", result.metrics?.handball_probability],
                        ["Image MLP", result.metrics?.mlp_foul],
                        ["Roboflow detector", result.metrics?.roboflow_foul],
                        ["Contact classifier", result.metrics?.contact_probability],
                        ["Tackle classifier", result.metrics?.tackle_probability],
                      ].map(([label, value]) => (
                        <div className="evidence-row" key={String(label)}>
                          <span>{label}</span>
                          <div><i style={{ width: formatPercent(value as number) }} /></div>
                          <strong>{formatPercent(value as number)}</strong>
                        </div>
                      ))}
                      <div className="classification-row">
                        <span>Handball</span>
                        <strong>{result.metrics?.handball_prediction?.replaceAll("_", " ") ?? "video only"}</strong>
                      </div>
                      <div className="classification-row">
                        <span>Handball quality</span>
                        <strong>{formatPercent(result.metrics?.handball_quality)}</strong>
                      </div>
                      <div className="classification-row">
                        <span>Ball–arm gap</span>
                        <strong>
                          {formatPercent(result.metrics?.handball_min_arm_distance)}
                          {result.metrics?.handball_proximity_override ? " · AUTO-CALL" : ""}
                        </strong>
                      </div>
                      <div className="classification-row">
                        <span>Contact</span>
                        <strong>{result.metrics?.contact_type?.replaceAll("_", " ") ?? "—"}</strong>
                      </div>
                      <div className="classification-row">
                        <span>Metric 3D</span>
                        <strong>{result.metrics?.metric_contact ?? "NOT RUN"}</strong>
                      </div>
                    </div>

                    {result.handball_overlay_url && (
                      <div className="handball-evidence">
                        <span>
                          HANDBALL EVIDENCE · {result.metrics?.handball_frames ?? 0} DIRECTION-CHANGE
                          {(result.metrics?.handball_frames ?? 0) === 1 ? " MOMENT" : " MOMENTS"}
                        </span>
                        <img src={`${API}${result.handball_overlay_url}`} alt="Handball detector evidence frames" />
                      </div>
                    )}

                    <button className="secondary-button" onClick={reset}>Review another incident</button>
                  </>
                ) : (
                  <>
                    <div className="decision-empty">
                      <span className="decision-glyph">◇</span>
                      <p className="eyebrow">Decision pending</p>
                      <h2>Ready for the whistle.</h2>
                      <p>The referee stack will report a foul, no foul, or request review—with the evidence behind it.</p>
                    </div>
                    <div className="run-settings">
                      <div><span>Handball method</span><strong>Ball + arm geometry</strong></div>
                      <div><span>Handball auto-call</span><strong>≥70% or ≤4% arm gap</strong></div>
                      <div><span>Required persistence</span><strong>0.12s</strong></div>
                      <div><span>Frame coverage</span><strong>Every frame</strong></div>
                    </div>
                    {state === "error" && <div className="error-box">{stage}</div>}
                    <button
                      className="primary-button analyze-button"
                      onClick={analyze}
                      disabled={!apiKey.trim() || working}
                    >
                      {working ? "Analysis running…" : `Analyze ${mode}`}
                    </button>
                    {!apiKey.trim() && <p className="button-help">Add your private Roboflow key to begin.</p>}
                  </>
                )}
              </aside>
            </div>
          )}
        </section>
      </div>

      {detailsOpen && (
        <div className="drawer-backdrop" onClick={() => setDetailsOpen(false)}>
          <aside className="details-drawer" onClick={(event) => event.stopPropagation()}>
            <div className="drawer-head">
              <div><p className="kicker">SYSTEM DETAILS</p><h2>Handball Project + foul pipeline</h2></div>
              <button onClick={() => setDetailsOpen(false)}>×</button>
            </div>
            <div className="pipeline-list">
              {modelStack.map(([name, description], index) => (
                <div key={name}><span>{String(index + 1).padStart(2, "0")}</span><p><strong>{name}</strong>{description}</p><i /></div>
              ))}
            </div>
            <div className="drawer-note">
              Handball source: the supplied nadimra Handball Detection Project. Its ball-direction, arm-collision, and arm-angle logic runs with modern MPS-compatible YOLO models. The existing every-frame foul pipeline remains GPU accelerated and supplies the general-foul decision.
            </div>
          </aside>
        </div>
      )}
    </main>
  );
}
