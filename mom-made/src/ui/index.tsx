import React, { useEffect, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import type { API, State, Action } from "../core/contracts";
declare global {
  interface Window {
    mom: API;
  }
}
function App() {
  const [state, setState] = useState<State>({
      apps: [],
      messages: [],
      runtimes: {},
      account: "Connecting…",
      models: [],
    }),
    [text, setText] = useState(""),
    [name, setName] = useState(""),
    [creating, setCreating] = useState(false),
    [model, setModel] = useState(""),
    [effort, setEffort] = useState(""),
    [logs, setLogs] = useState(false),
    [capture, setCapture] = useState<string>(),
    [localError, setLocalError] = useState("");
  const area = useRef<HTMLDivElement>(null),
    end = useRef<HTMLDivElement>(null);
  const selected = state.apps.find((a) => a.id === state.selected),
    runtime = selected && state.runtimes[selected.id],
    chosen = state.models.find((m) => m.model === model);
  async function command(c: Action) {
    try {
      setLocalError("");
      return await window.mom.command(c);
    } catch (e) {
      setLocalError(String(e));
    }
  }
  useEffect(() => {
    window.mom.state().then(setState);
    return window.mom.subscribe(setState);
  }, []);
  useEffect(() => {
    if (!model && state.models.length) {
      const m = state.models.find((m) => m.isDefault) ?? state.models[0];
      setModel(m.model);
      setEffort(m.defaultReasoningEffort);
    }
  }, [state.models]);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.messages]);
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    const update = () => {
      const b = el.getBoundingClientRect();
      void command({
        type: "bounds",
        x: Math.round(b.x),
        y: Math.round(b.y),
        width: Math.floor(b.width),
        height: Math.floor(b.height),
        visible: !logs && !capture && !creating,
      });
    };
    const observer = new ResizeObserver(update);
    observer.observe(el);
    update();
    return () => observer.disconnect();
  }, [selected?.id, logs, capture, creating]);
  async function create(e: React.FormEvent) {
    e.preventDefault();
    const id = await command({ type: "create", name });
    if (id) {
      setCreating(false);
      setName("");
    }
  }
  return (
    <>
      <header>
        <div className="brand">mom-made</div>
        <span className="tagline">Little things, made real.</span>
        <button onClick={() => setCreating(true)}>＋ New app</button>
      </header>
      <div className="workspace">
        <aside>
          <div className="chat-title">
            <span>YOUR WORKSPACE</span>
            <button
              disabled={!!state.busy || !selected?.threadId}
              onClick={() => command({ type: "compact" })}
            >
              Compact
            </button>
          </div>
          <div className="messages">
            {state.messages.length === 0 ? (
              <div className="welcome">
                <h1>
                  What shall
                  <br />
                  we make?
                </h1>
                <p>
                  A little tool. A place for your ideas. Something the internet
                  needs.
                </p>
                <p>
                  Create an app, then tell your agent what it should become.
                </p>
              </div>
            ) : (
              state.messages.map((m) => (
                <div key={m.id} className={"message " + m.role}>
                  <div className="message-meta">
                    {m.role === "user"
                      ? "You"
                      : m.role === "assistant"
                        ? "Codex"
                        : m.role === "error"
                          ? "Error"
                          : "Activity"}{" "}
                    · {state.apps.find((a) => a.id === m.appId)?.name}
                  </div>
                  <div>{m.text}</div>
                </div>
              ))
            )}
            <div ref={end} />
          </div>
          {state.approval && (
            <div className="approval" role="alert">
              <strong>Permission requested</strong>
              <pre>
                {String(
                  state.approval.params.command ??
                    state.approval.params.reason ??
                    state.approval.method,
                )}
              </pre>
              <button
                onClick={() =>
                  command({
                    type: "answer",
                    id: state.approval!.id,
                    accept: true,
                  })
                }
              >
                Allow once
              </button>
              <button
                onClick={() =>
                  command({
                    type: "answer",
                    id: state.approval!.id,
                    accept: false,
                  })
                }
              >
                Decline
              </button>
            </div>
          )}
          {state.busy && (
            <div className="working" role="status">
              Working on{" "}
              {state.apps.find((a) => a.id === state.busy?.appId)?.name}
              <button onClick={() => command({ type: "interrupt" })}>
                Stop
              </button>
            </div>
          )}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              if (selected && text.trim()) {
                void command({
                  type: "send",
                  appId: selected.id,
                  text,
                  model,
                  effort,
                });
                setText("");
              }
            }}
          >
            <div className="target">
              ↗ {selected?.name ?? "Create an app to begin"}
            </div>
            <textarea
              aria-label="Message to Codex"
              placeholder="What should we make next?"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey))
                  e.currentTarget.form?.requestSubmit();
              }}
            />
            <div className="composer-controls">
              <select
                aria-label="Model"
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  setEffort(
                    state.models.find((m) => m.model === e.target.value)
                      ?.defaultReasoningEffort ?? "",
                  );
                }}
              >
                {state.models.map((m) => (
                  <option key={m.model} value={m.model}>
                    {m.displayName}
                  </option>
                ))}
              </select>
              {chosen && (
                <select
                  aria-label="Reasoning effort"
                  value={effort}
                  onChange={(e) => setEffort(e.target.value)}
                >
                  {chosen.supportedReasoningEfforts.map((e) => (
                    <option key={e.reasoningEffort}>{e.reasoningEffort}</option>
                  ))}
                </select>
              )}
              <button
                className="primary"
                disabled={!selected || !!state.busy || !text.trim()}
              >
                Send ↑
              </button>
            </div>
          </form>
          <div className="account">
            <span>{state.account}</span>
            <button
              onClick={() =>
                command({
                  type:
                    state.account === "Login required" ? "login" : "connect",
                })
              }
            >
              {state.account === "Login required" ? "Sign in" : "Reconnect"}
            </button>
          </div>
        </aside>
        <main>
          <nav className="tabs" aria-label="App previews">
            {state.apps
              .filter((a) => a.open)
              .map((a) => (
                <div
                  key={a.id}
                  className={a.id === selected?.id ? "tab selected" : "tab"}
                >
                  <button
                    onClick={() => {
                      setCapture(undefined);
                      setLogs(false);
                      void command({ type: "select", appId: a.id });
                    }}
                  >
                    {a.name}
                  </button>
                  <button
                    aria-label={"Close " + a.name}
                    onClick={() => command({ type: "close", appId: a.id })}
                  >
                    ×
                  </button>
                </div>
              ))}
          </nav>
          <div className="toolbar">
            <span>{runtime?.url ?? "Local preview"}</span>
            {selected && (
              <>
                <button
                  onClick={() =>
                    command({
                      type: runtime?.status === "ready" ? "reload" : "start",
                      appId: selected.id,
                    })
                  }
                >
                  {runtime?.status === "ready" ? "Reload" : "Start"}
                </button>
                <button onClick={() => setLogs(!logs)} aria-pressed={logs}>
                  Logs
                </button>
                <button
                  disabled={runtime?.status !== "ready"}
                  onClick={async () => {
                    if (capture) setCapture(undefined);
                    else {
                      const r = (await command({
                        type: "capture",
                        appId: selected.id,
                      })) as { data: string } | undefined;
                      if (r) setCapture(r.data);
                    }
                  }}
                >
                  {capture ? "Back to live" : "Capture"}
                </button>
                <button
                  onClick={() =>
                    command({ type: "reveal", appId: selected.id })
                  }
                >
                  Files ↗
                </button>
              </>
            )}
          </div>
          <div className="preview-space" ref={area}>
            {capture ? (
              <div className="capture">
                <img src={capture} alt="Frozen capture of your app" />
              </div>
            ) : logs ? (
              <pre className="logs">
                {runtime?.logs || "No server logs yet."}
              </pre>
            ) : runtime?.status === "ready" ? null : (
              <div className="empty">
                <span className="seed">m</span>
                <h2>
                  {runtime?.status === "starting"
                    ? "Warming things up…"
                    : runtime?.status === "failed"
                      ? "A little help needed."
                      : selected
                        ? "Ready when you are."
                        : "A home for your next idea."}
                </h2>
                <p>
                  {runtime?.error ??
                    (selected
                      ? "Start this app to see it here."
                      : "Create an app, or reopen one from your library.")}
                </p>
                {selected && runtime?.status === "failed" && (
                  <button onClick={() => setLogs(true)}>View logs</button>
                )}
                {!selected && (
                  <button className="primary" onClick={() => setCreating(true)}>
                    Make something
                  </button>
                )}
                {state.apps
                  .filter((a) => !a.open)
                  .map((a) => (
                    <button
                      key={a.id}
                      onClick={async () => {
                        await command({ type: "select", appId: a.id });
                        await command({ type: "start", appId: a.id });
                      }}
                    >
                      Reopen {a.name}
                    </button>
                  ))}
              </div>
            )}
          </div>
          <footer>
            <span>
              {runtime?.status ?? "No app selected"} · local workspace
            </span>
            <span>React · Worker · D1</span>
          </footer>
        </main>
      </div>
      {(state.error || localError) && (
        <div className="error" role="alert">
          {localError || state.error}
        </div>
      )}
      {creating && (
        <div className="modal">
          <form onSubmit={create}>
            <div className="eyebrow">START WITH A LITTLE IDEA</div>
            <h2>What are we making?</h2>
            <label htmlFor="appname">App name</label>
            <input
              id="appname"
              autoFocus
              required
              maxLength={80}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Supper Club"
            />
            <p>Its own folder, live preview, and local database.</p>
            <div className="actions">
              <button type="button" onClick={() => setCreating(false)}>
                Cancel
              </button>
              <button className="primary">Create app ↗</button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
