import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  API,
  State,
  Action,
  Message,
  Mark,
  AppRecord,
  AgentId,
  ModelChoice,
  SettingsSection,
} from "../core/contracts";
import { Settings, effortLabels } from "./settings";
import { AppLibrary } from "./app-library";
import { ChatDivider } from "./chat-divider";
import { ChatMarkdown } from "./chat-markdown";
import { resolveModel } from "../core/model-routing";
import { CHAT_PANE, WORKSPACE_CHAT_ID, defaultPiLocalSettings } from "../core/contracts";
declare global {
  interface Window {
    mom: API;
  }
}
const EMPTY: State = {
  apps: [],
  messages: [],
  chatWidth: CHAT_PANE.initial,
  chatSessions: {},
  runtimes: {},
  account: "Connecting…",
  feedback: [],
  publish: {},
  cloudflare: { state: "unknown" },
  version: "",
  agent: "codex",
  agents: [],
  modelDefaults: {},
  openRouterKeySaved: false,
  piLocal: defaultPiLocalSettings,
};
const IDEAS = [
  "A recipe box for my family",
  "A page to plan our trip",
  "A little shop for my crafts",
  "A guest book for the party",
];
function nameFrom(text: string) {
  const w = text
    .replace(/[^\p{L}\p{N}\s'-]/gu, " ")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 4)
    .join(" ");
  return (w || "My app").slice(0, 40);
}
type Group =
  | { kind: "msg"; m: Message }
  | { kind: "work"; items: Message[]; key: string };
function group(ms: Message[]): Group[] {
  const out: Group[] = [];
  for (const m of ms) {
    if (m.role === "activity" && m.kind !== "divider" && m.kind !== "note") {
      const last = out[out.length - 1];
      if (last?.kind === "work") last.items.push(m);
      else out.push({ kind: "work", items: [m], key: m.id });
    } else out.push({ kind: "msg", m });
  }
  return out;
}
function summarize(items: Message[]) {
  const n = (k: string) => items.filter((i) => i.kind === k);
  const edits = n("edit").reduce(
    (s, i) => s + (Number(/\d+/.exec(i.text)?.[0]) || 1),
    0,
  );
  const parts: string[] = [];
  if (edits) parts.push(edits === 1 ? "Changed 1 file" : `Changed ${edits} files`);
  if (n("look").length) parts.push("Looked things over");
  if (n("command").length) parts.push("Ran a few steps");
  return parts.join(" · ") || "Worked on it";
}
function Spinner() {
  return <span className="spin" aria-hidden />;
}
function Work({ items, live }: { items: Message[]; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="work">
      <button
        className="work-head"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        {live ? <Spinner /> : <span className="tick">✓</span>}
        <span>{live ? "Working… " : ""}{summarize(items)}</span>
        <span className="chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <ul>
          {items.map((i) => (
            <li key={i.id}>
              {i.text}
              {i.detail && <pre>{i.detail}</pre>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
function Markup({
  src,
  appName,
  onCancel,
  onSend,
}: {
  src: string;
  appName: string;
  onCancel: () => void;
  onSend: (image: string, marks: Mark[], comment: string, w: number, h: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const img = useRef<HTMLImageElement>(null);
  const [tool, setTool] = useState<Mark["type"]>("pen");
  const [marks, setMarks] = useState<Mark[]>([]);
  const [draft, setDraft] = useState<Mark | null>(null);
  const [comment, setComment] = useState("");
  const [size, setSize] = useState({ w: 0, h: 0 });
  function paint(ctx: CanvasRenderingContext2D, list: Mark[], scale: number) {
    ctx.lineWidth = 4 * scale;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.strokeStyle = ctx.fillStyle = "#e0452b";
    list.forEach((m, idx) => {
      const [a, b] = [m.points[0], m.points[m.points.length - 1]];
      if (!a) return;
      ctx.beginPath();
      if (m.type === "pen") {
        ctx.moveTo(a[0], a[1]);
        m.points.forEach((p) => ctx.lineTo(p[0], p[1]));
        ctx.stroke();
      } else if (m.type === "rect") ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      else if (m.type === "ellipse") {
        ctx.ellipse((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, Math.abs(b[0] - a[0]) / 2 || 1, Math.abs(b[1] - a[1]) / 2 || 1, 0, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.arc(a[0], a[1], 15 * scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.font = `bold ${16 * scale}px -apple-system, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(list.slice(0, idx + 1).filter((x) => x.type === "pin").length), a[0], a[1] + 1);
        ctx.fillStyle = "#e0452b";
      }
    });
  }
  useEffect(() => {
    const c = canvas.current;
    if (!c || !size.w) return;
    c.width = size.w;
    c.height = size.h;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    paint(ctx, draft ? [...marks, draft] : marks, size.w / 800);
  }, [marks, draft, size]);
  function pt(e: React.PointerEvent): [number, number] {
    const r = canvas.current!.getBoundingClientRect();
    return [
      ((e.clientX - r.left) / r.width) * size.w,
      ((e.clientY - r.top) / r.height) * size.h,
    ];
  }
  function down(e: React.PointerEvent) {
    (e.target as Element).setPointerCapture(e.pointerId);
    const p = pt(e);
    if (tool === "pin") setMarks([...marks, { type: "pin", points: [p] }]);
    else setDraft({ type: tool, points: [p] });
  }
  function move(e: React.PointerEvent) {
    if (!draft) return;
    const p = pt(e);
    setDraft(
      draft.type === "pen"
        ? { ...draft, points: [...draft.points, p] }
        : { ...draft, points: [draft.points[0], p] },
    );
  }
  function up() {
    if (draft && draft.points.length > 1) setMarks([...marks, draft]);
    setDraft(null);
  }
  function submit() {
    const out = document.createElement("canvas");
    out.width = size.w;
    out.height = size.h;
    const ctx = out.getContext("2d")!;
    ctx.drawImage(img.current!, 0, 0, size.w, size.h);
    paint(ctx, marks, size.w / 800);
    onSend(out.toDataURL("image/png"), marks, comment, size.w, size.h);
  }
  const tools: [Mark["type"], string, string][] = [
    ["pen", "✎", "Scribble"],
    ["ellipse", "◯", "Circle"],
    ["rect", "▭", "Box"],
    ["pin", "●", "Pin"],
  ];
  return (
    <div className="markup">
      <div className="markup-bar">
        <div className="seg" role="group" aria-label="Drawing tool">
          {tools.map(([t, g, l]) => (
            <button key={t} aria-pressed={tool === t} onClick={() => setTool(t)} title={l}>
              <span aria-hidden>{g}</span> {l}
            </button>
          ))}
        </div>
        <button className="quiet" disabled={!marks.length} onClick={() => setMarks(marks.slice(0, -1))}>
          Undo
        </button>
        <button className="quiet" onClick={onCancel}>Cancel</button>
      </div>
      <div className="markup-stage">
        <div className="markup-frame">
          <img
            ref={img}
            src={src}
            alt={`Frozen picture of ${appName}`}
            onLoad={(e) =>
              setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
            }
          />
          <canvas ref={canvas} onPointerDown={down} onPointerMove={move} onPointerUp={up} aria-label="Draw on the picture" />
        </div>
      </div>
      <form
        className="markup-ask"
        onSubmit={(e) => {
          e.preventDefault();
          if (comment.trim()) submit();
        }}
      >
        <input
          autoFocus
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Circle something, then tell me what to change…"
          aria-label="What should change?"
        />
        <button className="cta" disabled={!comment.trim()}>
          Make this change
        </button>
      </form>
    </div>
  );
}
function PublishSheet({
  app,
  state,
  command,
  onClose,
}: {
  app: AppRecord;
  state: State;
  command: (c: Action) => Promise<unknown>;
  onClose: () => void;
}) {
  const p = state.publish[app.id];
  const cf = state.cloudflare;
  const [copied, setCopied] = useState(false);
  const steps = [
    "Checking your Cloudflare account",
    "Saving a snapshot of your app",
    "Preparing your app's database",
    "Building",
    "Updating the live database",
    "Going live",
    "Checking that it works",
  ];
  const at = Math.max(0, steps.indexOf(p?.step ?? ""));
  const last = p?.history.find((h) => h.ok && h.url);
  const live = p?.status === "done" ? p.url : last?.url;
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Publish">
      <div className="sheet">
        {p?.status === "working" ? (
          <>
            <Spinner />
            <h2>Putting {app.name} on the internet…</h2>
            <p className="step">{p.step}</p>
            <div className="bar"><i style={{ width: `${((at + 1) / steps.length) * 100}%` }} /></div>
          </>
        ) : p?.status === "done" ? (
          <>
            <div className="big-tick">✓</div>
            <h2>{app.name} is live!</h2>
            {p.url ? (
              <>
                <a className="url" href={p.url} onClick={(e) => { e.preventDefault(); void command({ type: "open", url: p.url! }); }}>
                  {p.url.replace("https://", "")}
                </a>
                <p>Anyone with this link can visit it. Send it to your family!</p>
                <div className="actions">
                  <button
                    onClick={() => {
                      void navigator.clipboard.writeText(p.url!).then(() => setCopied(true));
                    }}
                  >
                    {copied ? "Copied ✓" : "Copy link"}
                  </button>
                  <button className="cta" onClick={() => void command({ type: "open", url: p.url! })}>
                    Open it
                  </button>
                </div>
              </>
            ) : (
              <p>Everything checked out.</p>
            )}
          </>
        ) : cf.state !== "ready" ? (
          <>
            <div className="glyph">☁︎</div>
            <h2>Connect to Cloudflare</h2>
            <p>
              Cloudflare is the free service that hosts your app on the internet. Sign in (or make a free account) once and you're set.
            </p>
            {cf.state === "unknown" && <p className="soft">Checking…</p>}
            <div className="actions">
              <button className="quiet" onClick={onClose}>Not now</button>
              <button className="cta" onClick={() => void command({ type: "cloudflareLogin" })}>
                Sign in to Cloudflare
              </button>
            </div>
            <button className="quiet small" onClick={() => void command({ type: "cloudflareCheck" })}>
              I've signed in
            </button>
          </>
        ) : (
          <>
            <div className="glyph">↗</div>
            <h2>{live ? `Update ${app.name}` : `Put ${app.name} on the internet`}</h2>
            <p>
              {live
                ? `Your published copy will be replaced with what you see now. Its data stays safe.`
                : `We'll give it its own web address that anyone can visit. You can update it any time.`}
            </p>
            {p?.status === "failed" && (
              <div className="oops" role="alert">
                <strong>That didn't work.</strong>
                <pre>{p.error}</pre>
                <button className="quiet" onClick={() => { onClose(); void command({ type: "fix", appId: app.id, agent: state.agent, model: "", effort: "" }); }}>
                  Ask the assistant to fix it
                </button>
              </div>
            )}
            {live && (
              <p className="soft">Currently live at {live.replace("https://", "")}</p>
            )}
            <div className="actions">
              <button className="quiet" onClick={onClose}>Cancel</button>
              <button className="cta" onClick={() => void command({ type: "publish", appId: app.id, model: "", effort: "" })}>
                {p?.status === "failed" ? "Try again" : live ? "Update it" : "Publish"}
              </button>
            </div>
            <p className="soft small">Signed in as {cf.account ?? "your Cloudflare account"}</p>
          </>
        )}
        {p?.status !== "working" && p?.status !== "idle" && p?.status !== undefined && (
          <button className="close" aria-label="Close" onClick={onClose}>×</button>
        )}
      </div>
    </div>
  );
}
function App() {
  const [state, setState] = useState<State>(EMPTY);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [creating, setCreating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [chatWidth, setChatWidth] = useState(CHAT_PANE.initial);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [dragging, setDragging] = useState(false);
  const [override, setOverride] = useState<(ModelChoice & { agent: AgentId })>();
  const [logs, setLogs] = useState(false);
  const [capture, setCapture] = useState<string>();
  const [sheet, setSheet] = useState(false);
  const [library, setLibrary] = useState(false);
  const [settings, setSettings] = useState<SettingsSection>();
  const settingsButton = useRef<HTMLButtonElement>(null);
  const appsButton = useRef<HTMLButtonElement>(null);
  const [more, setMore] = useState(false);
  const [localError, setLocalError] = useState("");
  const end = useRef<HTMLDivElement>(null);
  const area = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const selected = state.apps.find((a) => a.id === state.selected);
  const runtime = selected && state.runtimes[selected.id];
  const openApps = state.apps.filter((a) => a.open);
  const busyApp = state.apps.find((a) => a.id === state.busy?.appId);
  const buildingNew = creating && !selected;
  const chatId = selected?.id ?? WORKSPACE_CHAT_ID;
  const draftId = buildingNew ? "new-app" : chatId;
  const text = drafts[draftId] ?? "";
  const maxChatWidth = Math.max(CHAT_PANE.min, Math.min(CHAT_PANE.max, windowWidth - CHAT_PANE.previewMin - CHAT_PANE.divider));
  const paneWidth = Math.min(chatWidth, maxChatWidth);
  const session = selected
    ? selected.sessions?.[state.agent] ?? (state.agent === "codex" ? selected.threadId : undefined)
    : state.chatSessions[state.agent];
  function setText(value: string) {
    setDrafts((items) => ({ ...items, [draftId]: value }));
  }
  const agent = state.agents.find((a) => a.id === state.agent);
  const models = agent?.models ?? [];
  const saved = state.modelDefaults[state.agent];
  const activeOverride = override?.agent === state.agent && models.some((m) => m.model === override.model) ? override : undefined;
  const { model, effort } = resolveModel(models, saved, activeOverride);
  const requestModel = { model: activeOverride?.model ?? "", effort: activeOverride?.effort ?? "" };
  const chosen = models.find((m) => m.model === model);
  async function command(c: Action, onError?: () => void) {
    try {
      setLocalError("");
      return await window.mom.command(c);
    } catch (e) {
      setLocalError(String(e).replace(/^Error: (Error invoking remote method '\w+': )?(Error: )?/, ""));
      onError?.();
    }
  }
  async function run(c: Action) {
    let ok = true;
    await command(c, () => { ok = false; });
    return ok;
  }
  const openChatLink = useCallback((url: string) => {
    void window.mom.command({ type: "openChatLink", url }).catch(() => {
      setLocalError("This link couldn’t be opened.");
    });
  }, []);
  function commitWidth(width: number) {
    setChatWidth(width);
    void command({ type: "chatWidth", width });
  }
  function showChat(newApp = false) {
    setCapture(undefined);
    setLogs(false);
    setSheet(false);
    setMore(false);
    setCreating(newApp);
    void command({ type: "showChat" }).then(() => composer.current?.focus());
  }
  function showSettings(section: SettingsSection = "assistants") {
    setLocalError("");
    setLibrary(false);
    setMore(false);
    setSettings(section);
  }
  function closeSettings() {
    setLocalError("");
    setSettings(undefined);
    settingsButton.current?.focus();
  }
  function showLibrary() {
    setLocalError("");
    setMore(false);
    setLibrary(true);
  }
  async function openApp(appId: string) {
    const opened = await run({ type: "openApp", appId });
    if (opened) {
      setSettings(undefined);
      if (appId !== selected?.id) {
        setCapture(undefined);
        setLogs(false);
      }
    }
    return opened;
  }
  useEffect(() => {
    window.mom.state().then(setState);
    const unsubscribe = window.mom.subscribe(setState);
    const removeMenu = window.mom.onSettings(() => showSettings());
    return () => { unsubscribe(); removeMenu(); };
  }, []);
  useEffect(() => setOverride(undefined), [state.agent, saved?.model, saved?.effort]);
  useEffect(() => setChatWidth(state.chatWidth), [state.chatWidth]);
  useEffect(() => {
    const resize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);
  useEffect(() => {
    if (selected) setCreating(false);
    else setDragging(false);
  }, [selected?.id]);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatId, state.messages.length, state.busy, state.approval]);
  const overlay = !!capture || logs || sheet || more || !!settings || library || dragging;
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    const update = () => {
      const b = el.getBoundingClientRect();
      if (!b.width || !b.height) {
        void window.mom.command({ type: "previewVisibility", visible: false });
        return;
      }
      void window.mom.command({
        type: "bounds",
        x: Math.round(b.x),
        y: Math.round(b.y),
        width: Math.floor(b.width),
        height: Math.floor(b.height),
        visible: !overlay,
      });
    };
    const ro = new ResizeObserver(update);
    ro.observe(el);
    update();
    return () => ro.disconnect();
  }, [selected?.id, overlay]);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (library) return;
        if (settings) { closeSettings(); return; }
        setCapture(undefined);
        setLogs(false);
        setSheet(false);
        setMore(false);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [settings, library]);
  const groups = useMemo(() => group(state.messages), [state.messages]);
  const feedbackFor = (m: Message) => state.feedback.find((f) => f.image === m.image && m.image);
  async function submit() {
    const t = text.trim();
    if (!t || state.busy || submitting) return;
    if (!agent?.ready) { showSettings(); return; }
    setSubmitting(true);
    setText("");
    try {
      let target = chatId;
      if (buildingNew) {
        const id = (await command({ type: "create", name: nameFrom(t) })) as string | undefined;
        if (!id) { setText(t); return; }
        target = id;
      }
      await command({ type: "send", appId: target, text: t, agent: state.agent, ...requestModel }, () => {
        setDrafts((items) => ({ ...items, [target]: items[target] || t }));
      });
    } finally {
      setSubmitting(false);
    }
  }
  async function startMarkup() {
    if (!selected || state.agent === "pi-local") return;
    const r = (await command({ type: "capture", appId: selected.id })) as { data: string } | undefined;
    if (r) {
      setLogs(false);
      setCapture(r.data);
    }
  }
  const status = runtime?.status;
  const ready = status === "ready";
  const live = selected && state.publish[selected.id]?.history.find((h) => h.ok && h.url);
  const justOpened = buildingNew || state.messages.length === 0;
  const busyHere = state.busy?.appId === chatId && !buildingNew;
  return (
    <>
      <header>
        <div className="brand">mom-made</div>
        <button
          ref={appsButton}
          className="library-toggle"
          aria-haspopup="dialog"
          aria-expanded={library}
          onClick={showLibrary}
        >
          Apps
        </button>
        <nav hidden={!!settings} className="tabs" aria-label="Your apps">
          <button
            className="chat-tab"
            aria-pressed={!selected && !buildingNew}
            onClick={() => showChat()}
          >
            Chat
          </button>
          {openApps.map((a) => (
            <div key={a.id} className={a.id === selected?.id ? "tab selected" : "tab"}>
              <button
                className="tab-name"
                onClick={() => {
                  setCapture(undefined);
                  setLogs(false);
                  setSheet(false);
                  setMore(false);
                  setCreating(false);
                  void command({ type: "select", appId: a.id });
                }}
              >
                <i className={"dot " + (state.runtimes[a.id]?.status ?? "stopped")} aria-hidden />
                {a.name}
              </button>
              <button className="tab-x" aria-label={"Close " + a.name} onClick={() => command({ type: "close", appId: a.id })}>
                ×
              </button>
            </div>
          ))}
        </nav>
        <button
          className="new"
          hidden={!!settings}
          onClick={() => showChat(true)}
        >
          ＋ New App
        </button>
        <button ref={settingsButton} className="settings-toggle" aria-label="Settings" title="Settings (⌘,)" aria-pressed={!!settings} onClick={() => settings ? closeSettings() : showSettings()}>⚙ <span>Settings</span></button>
      </header>
      <div
        className={"workspace" + (selected ? "" : " chat-only") + (dragging ? " resizing" : "")}
        style={selected ? { "--chat-width": `${paneWidth}px` } as React.CSSProperties : undefined}
        hidden={!!settings}
      >
        <aside id="chat-pane" aria-label={selected ? `Chat for ${selected.name}` : buildingNew ? "New app" : "Chat"}>
          <div className="messages" aria-live="polite">
            {justOpened ? (
              <div className="welcome">
                <h1>{buildingNew ? "What would you like to make?" : selected ? `What should we do with ${selected.name}?` : "What’s on your mind?"}</h1>
                {buildingNew && (
                  <>
                    <p>Describe your app to start building.</p>
                    <div className="ideas">
                      {IDEAS.map((i) => (
                        <button key={i} onClick={() => { setText(i); composer.current?.focus(); }}>{i}</button>
                      ))}
                    </div>
                  </>
                )}
                {!selected && !buildingNew && <p>Ask a question or plan an app. Choose <b>New App</b> when you’re ready to build.</p>}
              </div>
            ) : (
              groups.map((g, i) => {
                if (g.kind === "work")
                  return <Work key={g.key} items={g.items} live={busyHere && i === groups.length - 1} />;
                const m = g.m;
                if (m.kind === "divider")
                  return <div key={m.id} className="divider"><span>New conversation</span></div>;
                if (m.role === "activity")
                  return <div key={m.id} className="note">{m.text}</div>;
                const f = feedbackFor(m);
                return (
                  <div key={m.id} className={"message " + m.role}>
                    {m.role !== "user" && (
                      <div className="who">
                        {m.role === "error" ? "Something went wrong" : "mom-made"}
                      </div>
                    )}
                    {m.image && <div className="chip">✎ Marked-up picture</div>}
                    <div className={m.role === "assistant" ? "body markdown" : "body"}>
                      {m.role === "assistant" ? <ChatMarkdown id={m.id} text={m.text} openLink={openChatLink} /> : m.text}
                    </div>
                    {f && (
                      <div className={"fb " + f.status}>
                        {f.status === "open" && <span>Working on it…</span>}
                        {f.status === "addressed" && (
                          <>
                            <span>Done — does that look right?</span>
                            <button onClick={() => command({ type: "feedback", id: f.id, status: "resolved" })}>Looks good</button>
                            {state.agent !== "pi-local" && <button onClick={startMarkup}>Not quite</button>}
                          </>
                        )}
                        {f.status === "resolved" && (
                          <>
                            <span>✓ Sorted</span>
                            <button className="quiet" onClick={() => command({ type: "feedback", id: f.id, status: "open" })}>Reopen</button>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
            {state.approval && (
              <div className="ask" role="alert">
                <strong>
                  {state.approval.method.includes("fileChange")
                    ? "Okay to change some files?"
                    : "Okay to run a quick step?"}
                </strong>
                <p>{String(state.approval.params.reason ?? "The assistant needs your OK to keep going.")}</p>
                {!!state.approval.params.command && (
                  <details>
                    <summary>See details</summary>
                    <pre>{String(state.approval.params.command)}</pre>
                  </details>
                )}
                <div className="actions">
                  <button className="quiet" onClick={() => command({ type: "answer", id: state.approval!.id, accept: false })}>
                    Don't allow
                  </button>
                  <button onClick={() => command({ type: "answer", id: state.approval!.id, accept: true, always: true })}>
                    Always okay
                  </button>
                  <button className="cta" onClick={() => command({ type: "answer", id: state.approval!.id, accept: true })}>
                    Allow
                  </button>
                </div>
              </div>
            )}
            {busyHere && !state.approval && state.messages.at(-1)?.role !== "activity" && (
              <div className="thinking"><Spinner /> Thinking…</div>
            )}
            <div ref={end} />
          </div>
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            {state.busy && (
              <div className="working" role="status">
                <Spinner /> {busyApp ? <>Working on <b>{busyApp.name}</b></> : "Replying in Chat"}
                <button type="button" onClick={() => command({ type: "interrupt" })}>Stop</button>
              </div>
            )}
            <textarea
              ref={composer}
              aria-label={buildingNew ? "Describe your app" : "Message"}
              placeholder={selected ? `What should change in ${selected.name}?` : buildingNew ? "Describe what you’d like to make…" : "Ask a question or talk through an idea…"}
              value={text}
              rows={2}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void submit();
                }
              }}
            />
            <div className="composer-row">
              <details className="opts">
                <summary title="Chat options" aria-label="Chat options">⚙</summary>
                <div className="pop">
                  <label>
                    Model
                    <select value={activeOverride?.model ?? ""} disabled={!models.length} onChange={(e) => {
                      const next = e.target.value;
                      setOverride(next ? { agent: state.agent, ...resolveModel(models, undefined, { model: next, effort: "" }) } : undefined);
                    }}>
                      <option value="">{models.find((m) => m.model === resolveModel(models, saved).model)?.displayName ?? "Assistant default"} · Default</option>
                      {models.map((m) => <option key={m.model} value={m.model}>{m.displayName}</option>)}
                    </select>
                  </label>
                  {chosen && chosen.supportedReasoningEfforts.length > 0 && (
                    <label>
                      Reasoning
                      <select value={effort} onChange={(e) => setOverride({ agent: state.agent, model, effort: e.target.value })}>
                        {chosen.supportedReasoningEfforts.map((x) => (
                          <option key={x.reasoningEffort} value={x.reasoningEffort}>{effortLabels[x.reasoningEffort] ?? x.reasoningEffort}</option>
                        ))}
                      </select>
                    </label>
                  )}
                  <button type="button" onClick={(e) => {
                    e.currentTarget.closest("details")?.removeAttribute("open");
                    showSettings();
                  }}>{agent?.label ?? "Assistant settings"} ▸</button>
                  {agent?.compact && !buildingNew && (
                    <button type="button" className="quiet" disabled={!!state.busy || !session} onClick={() => command({ type: "compact" })}>Tidy up this chat</button>
                  )}
                  {!buildingNew && <button type="button" className="quiet" disabled={!!state.busy || !state.messages.length} onClick={() => command({ type: "newConversation" })}>Start a fresh chat</button>}
                  <span className="soft small">{state.account}</span>
                </div>
              </details>
              <span className="hint">⏎ to send</span>
              <button className="cta send" disabled={!!state.busy || submitting || !text.trim()} aria-label={buildingNew ? "Build app" : "Send"}>↑</button>
            </div>
          </form>
        </aside>
        {selected && <ChatDivider width={paneWidth} max={maxChatWidth} onWidth={setChatWidth} onCommit={commitWidth} onDragging={setDragging} />}
        <main hidden={!selected}>
          <div className="toolbar">
            <div className="title">
              {selected ? (
                <input
                  className="rename"
                  aria-label="App name"
                  defaultValue={selected.name}
                  key={selected.id}
                  onBlur={(e) => {
                    const v = e.target.value.trim();
                    if (v && v !== selected.name) void command({ type: "rename", appId: selected.id, name: v });
                  }}
                  onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                />
              ) : null}
            </div>
            {selected && (
              <>
                <button className="tool" disabled={!ready || !!capture || state.agent === "pi-local"} onClick={startMarkup}
                  title={state.agent === "pi-local" ? "Local Pi accepts text. Switch assistants for visual markup." : "Point at what you'd like changed"}>
                  <span className="markup-full">✎ Show me what to change</span>
                  <span className="markup-short">✎ Mark up</span>
                </button>
                <button className="tool icon" disabled={!ready} onClick={() => command({ type: "reload", appId: selected.id })} title="Reload" aria-label="Reload">↻</button>
                <div className="menu">
                  <button className="tool icon" aria-label="More" aria-expanded={more} onClick={() => setMore(!more)}>⋯</button>
                  {more && (
                    <div className="dropdown" role="menu" onClick={() => setMore(false)}>
                      <button role="menuitem" onClick={() => setLogs(true)}>See what's happening behind the scenes</button>
                      <button role="menuitem" onClick={() => command({ type: "reveal", appId: selected.id })}>Show files in Finder</button>
                      <button role="menuitem" onClick={() => command({ type: ready ? "stop" : "start", appId: selected.id })}>{ready ? "Pause this app" : "Start this app"}</button>
                    </div>
                  )}
                </div>
                <button className="publish" disabled={!ready && !live} onClick={() => { setSheet(true); void command({ type: "cloudflareCheck" }); }}>
                  <span aria-hidden>↗</span> {live ? "Update" : "Publish"}
                </button>
              </>
            )}
          </div>
          <div className="preview-space" ref={area}>
            {capture && selected ? (
              <Markup
                src={capture}
                appName={selected.name}
                onCancel={() => setCapture(undefined)}
                onSend={(image, marks, comment, w, h) => {
                  setCapture(undefined);
                  void command({ type: "markup", appId: selected.id, comment, image, marks, viewport: { width: w, height: h }, agent: state.agent, ...requestModel });
                }}
              />
            ) : logs ? (
              <div className="logs-wrap">
                <div className="logs-head"><b>Behind the scenes</b><button className="quiet" onClick={() => setLogs(false)}>Close</button></div>
                <pre className="logs">{runtime?.logs || "Nothing to show yet."}</pre>
              </div>
            ) : ready ? null : (
              <div className="empty">
                {status === "starting" ? (
                  <>
                    <Spinner />
                    <h2>Getting your app ready…</h2>
                    <p>This takes a few seconds the first time.</p>
                  </>
                ) : status === "failed" && selected ? (
                  <>
                    <div className="glyph">✦</div>
                    <h2>This app hit a snag</h2>
                    <p>{(runtime?.error ?? "It stopped unexpectedly.").split("\n")[0].slice(0, 160)}</p>
                    <div className="actions">
                      <button onClick={() => command({ type: "start", appId: selected.id })}>Try again</button>
                      <button className="cta" disabled={!!state.busy} onClick={() => command({ type: "fix", appId: selected.id, agent: state.agent, ...requestModel })}>Ask the assistant to fix it</button>
                    </div>
                    <button className="quiet small" onClick={() => setLogs(true)}>See details</button>
                  </>
                ) : selected ? (
                  <>
                    <div className="glyph">◌</div>
                    <h2>{selected.name} is resting</h2>
                    <button className="cta" onClick={() => command({ type: "start", appId: selected.id })}>Wake it up</button>
                  </>
                ) : null}
              </div>
            )}
          </div>
        </main>
      </div>
      {!library && (state.error || localError) && (
        <div className="toast" role="alert">
          <span>{(localError || state.error || "").split("\n")[0].slice(0, 200)}</span>
          <button className="quiet" onClick={() => setLocalError("")}>OK</button>
        </div>
      )}
      {settings && <Settings state={state} section={settings} setSection={setSettings} run={run} onClose={closeSettings} />}
      {library && (
        <AppLibrary
          state={state}
          error={localError}
          onOpen={openApp}
          onClose={() => {
            setLocalError("");
            setLibrary(false);
            requestAnimationFrame(() => appsButton.current?.focus());
          }}
        />
      )}
      {!settings && sheet && selected && (
        <PublishSheet app={selected} state={state} command={command} onClose={() => setSheet(false)} />
      )}
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
