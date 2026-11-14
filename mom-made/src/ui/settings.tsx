import React, { useEffect, useRef, useState } from "react";
import type {
  Action,
  AgentId,
  PiLocalSettings,
  SettingsSection,
  State,
} from "../core/contracts";
import { resolveModel } from "../core/model-routing";
import { runtimeLabels } from "./app-library";

type Run = (action: Action) => Promise<boolean>;
const sections = {
  assistants: "Assistants",
  cloudflare: "Cloudflare",
  admin: "Admin",
} satisfies Record<SettingsSection, string>;
const assistantNames = {
  codex: "Codex",
  claude: "Claude Code",
  pi: "Pi + OpenRouter",
  "pi-local": "Pi + Local model",
} satisfies Record<AgentId, string>;
export const effortLabels: Record<string, string> = {
  none: "None",
  off: "Off",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Very high",
  max: "Maximum",
};

function Assistants({ state, run }: { state: State; run: Run }) {
  const [id, setId] = useState(state.agent);
  const [key, setKey] = useState("");
  const [pending, setPending] = useState(false);
  const [copied, setCopied] = useState("");
  const agent = state.agents.find((a) => a.id === id);
  const saved = state.modelDefaults[id];
  const models = agent?.models ?? [];
  const savedModel = models.find((m) => m.model === saved?.model);
  const defaults = resolveModel(models, saved);
  const fallback = models.find((m) => m.isDefault) ?? models[0];
  const active = id === state.agent;

  async function update(action: Action) {
    setPending(true);
    try {
      if ((await run(action)) && action.type === "agentKey") setKey("");
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <div className="settings-intro">
        <div>
          <h1>Assistants</h1>
          <p>Connect an assistant and choose the model it starts with.</p>
        </div>
        <button
          className="settings-button"
          disabled={pending}
          onClick={() => void update({ type: "recheck" })}
        >
          {pending ? "Checking…" : "Refresh connections"}
        </button>
      </div>
      <div className="assistant-picker" aria-label="Inspect an assistant">
        {(Object.keys(assistantNames) as AgentId[]).map((value) => {
          const a = state.agents.find((item) => item.id === value);
          const status = !a
            ? "Checking…"
            : a.ready
              ? "Ready"
              : !a.installed
                ? "Not installed"
                : a.help.needsKey || value === "pi-local"
                  ? "Needs setup"
                  : "Needs sign-in";
          return (
            <button
              key={value}
              aria-pressed={id === value}
              onClick={() => {
                setId(value);
                setKey("");
                setCopied("");
              }}
            >
              <strong>{assistantNames[value]}</strong>
              <span className={a?.ready ? "connected" : "soft"}>{status}</span>
              {value === state.agent && (
                <span
                  className="assistant-active"
                  aria-label="Active assistant"
                >
                  Active
                </span>
              )}
            </button>
          );
        })}
      </div>
      {agent && (
        <>
          <section
            className="settings-card"
            aria-label={agent.label + " connection"}
          >
            <div className="settings-row">
              <div>
                <h2>{agent.label}</h2>
                <p>
                  {agent.account ??
                    (agent.installed
                      ? agent.help.loginHint
                      : "Install this assistant to use it in the studio.")}
                </p>
              </div>
              {!active && (
                <button
                  className="settings-button"
                  disabled={!!state.busy || pending || !agent.ready}
                  onClick={() => void update({ type: "agent", id })}
                >
                  Use assistant
                </button>
              )}
            </div>
            {!agent.installed &&
              agent.help.install.map((install) => (
                <div className="cmd" key={install}>
                  <code>{install}</code>
                  <button
                    className="quiet"
                    onClick={() =>
                      void navigator.clipboard
                        .writeText(install)
                        .then(() => setCopied(install))
                    }
                  >
                    {copied === install ? "Copied" : "Copy"}
                  </button>
                </div>
              ))}
            {id !== "pi-local" &&
              agent.installed &&
              !agent.help.needsKey &&
              !agent.ready && (
                <button
                  className="settings-button"
                  disabled={pending}
                  onClick={() => void update({ type: "login", id })}
                >
                  Sign in
                </button>
              )}
            {agent.help.needsKey && (
              <form
                className="keyrow"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (key.trim())
                    void update({ type: "agentKey", key: key.trim() });
                }}
              >
                <input
                  aria-label="OpenRouter API key"
                  type="password"
                  autoComplete="off"
                  maxLength={400}
                  placeholder={
                    state.openRouterKeySaved
                      ? "Replace saved API key"
                      : "OpenRouter API key"
                  }
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                />
                <button
                  className="settings-button"
                  disabled={!key.trim() || pending || (active && !!state.busy)}
                >
                  {state.openRouterKeySaved ? "Replace" : "Save key"}
                </button>
                {state.openRouterKeySaved && (
                  <button
                    type="button"
                    className="quiet"
                    disabled={pending || (active && !!state.busy)}
                    onClick={() => void update({ type: "agentKey", key: "" })}
                  >
                    Remove
                  </button>
                )}
              </form>
            )}
            {id === "pi-local" && (
              <LocalModelFields
                settings={state.piLocal}
                disabled={pending || (active && !!state.busy)}
                save={(settings) =>
                  void update({ type: "piLocalSettings", settings })
                }
              />
            )}
            {agent.detail && !agent.ready && (
              <p role="status">{agent.detail.slice(0, 240)}</p>
            )}
            {agent.help.note && <p>{agent.help.note}</p>}
          </section>
          {id !== "pi-local" && (
            <section className="settings-card" aria-label="Model routing">
              <div className="settings-row">
                <div>
                  <h2>Model defaults</h2>
                  <p>
                    Applies to new requests. You can pick a different model in
                    chat.
                  </p>
                </div>
              </div>
              <div className="settings-fields">
                <label>
                  Default model
                  <select
                    value={savedModel?.model ?? ""}
                    disabled={!models.length || pending}
                    onChange={(e) =>
                      void update({
                        type: "modelDefault",
                        id,
                        model: e.target.value,
                        effort: "",
                      })
                    }
                  >
                    <option value="">
                      {fallback
                        ? `Assistant default · ${fallback.displayName}`
                        : "No models available"}
                    </option>
                    {models.map((m) => (
                      <option key={m.model} value={m.model}>
                        {m.displayName}
                      </option>
                    ))}
                  </select>
                </label>
                {savedModel?.supportedReasoningEfforts.length ? (
                  <label>
                    Default reasoning
                    <select
                      value={defaults.effort}
                      disabled={pending}
                      onChange={(e) =>
                        void update({
                          type: "modelDefault",
                          id,
                          model: savedModel.model,
                          effort: e.target.value,
                        })
                      }
                    >
                      {savedModel.supportedReasoningEfforts.map((effort) => (
                        <option
                          key={effort.reasoningEffort}
                          value={effort.reasoningEffort}
                        >
                          {effortLabels[effort.reasoningEffort] ??
                            effort.reasoningEffort}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </div>
              {saved?.model && !savedModel && (
                <p>
                  The saved model is unavailable. New requests use the
                  assistant’s default.
                </p>
              )}
            </section>
          )}
        </>
      )}
    </>
  );
}

function LocalModelFields({
  settings,
  disabled,
  save,
}: {
  settings: PiLocalSettings;
  disabled: boolean;
  save: (settings: PiLocalSettings) => void;
}) {
  const [draft, setDraft] = useState(settings);
  useEffect(
    () => setDraft(settings),
    [settings.baseUrl, settings.model, settings.contextWindow],
  );
  return (
    <form
      className="settings-fields local-model-fields"
      onSubmit={(event) => {
        event.preventDefault();
        save(draft);
      }}
    >
      <label>
        Server URL
        <input
          value={draft.baseUrl}
          required
          maxLength={200}
          onChange={(event) =>
            setDraft({ ...draft, baseUrl: event.target.value })
          }
        />
      </label>
      <label>
        Model ID
        <input
          value={draft.model}
          required
          maxLength={200}
          onChange={(event) =>
            setDraft({ ...draft, model: event.target.value })
          }
        />
      </label>
      <label>
        Context tokens
        <input
          type="number"
          min={16384}
          max={262144}
          step={1}
          required
          value={
            Number.isFinite(draft.contextWindow) ? draft.contextWindow : ""
          }
          onChange={(event) =>
            setDraft({ ...draft, contextWindow: event.target.valueAsNumber })
          }
        />
      </label>
      <button className="settings-button" disabled={disabled}>
        Save and check
      </button>
      <details>
        <summary>Run Flint on Apple Silicon</summary>
        <p>
          Flint INT4 needs about 26 GB of disk space and at least 48 GB of
          unified memory. Install a recent llama.cpp release with Flint support,
          then run this in Terminal and wait for the server to load.
        </p>
        <div className="cmd">
          <code>{FLINT_COMMAND}</code>
          <button
            type="button"
            className="quiet"
            onClick={() => void navigator.clipboard.writeText(FLINT_COMMAND)}
          >
            Copy
          </button>
        </div>
      </details>
    </form>
  );
}

const FLINT_COMMAND =
  "llama-server -hf RecursiveSI/flint-int4-gguf:Q4_1 --alias flint -ngl all -c 65536 -np 1 --cache-ram 0 -b 256 -ub 128 -t 8 -fa on -ctk f16 -ctv f16 --jinja --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080 --no-webui --cors-origins http://127.0.0.1:8080";

function CloudflareSettings({ state, run }: { state: State; run: Run }) {
  const [pending, setPending] = useState<
    "cloudflareLogin" | "cloudflareCheck"
  >();
  const cf = state.cloudflare;
  async function check(type: "cloudflareLogin" | "cloudflareCheck") {
    setPending(type);
    try {
      await run({ type });
    } finally {
      setPending(undefined);
    }
  }
  return (
    <>
      <div className="settings-intro">
        <div>
          <h1>Cloudflare</h1>
          <p>Connect the account that publishes your apps to workers.dev.</p>
        </div>
      </div>
      <section className="settings-card">
        <div className="settings-row">
          <div>
            <h2>
              {cf.state === "ready"
                ? "Connected"
                : cf.state === "login-required"
                  ? "Sign-in required"
                  : "Connection unverified"}
            </h2>
            <p role="status">
              {pending === "cloudflareLogin"
                ? "Finish signing in in your browser."
                : pending === "cloudflareCheck"
                  ? "Checking your connection…"
                  : (cf.account ??
                    (cf.state === "ready"
                      ? "Connected through Wrangler."
                      : "Sign in through Cloudflare in your browser."))}
            </p>
          </div>
          <div className="settings-actions">
            <button
              className="settings-button"
              disabled={!!pending}
              onClick={() => void check("cloudflareCheck")}
            >
              Check connection
            </button>
            {cf.state !== "ready" && (
              <button
                className="cta"
                disabled={!!pending}
                onClick={() => void check("cloudflareLogin")}
              >
                Sign in
              </button>
            )}
          </div>
        </div>
      </section>
    </>
  );
}

function Admin({
  state,
  run,
  onClose,
}: {
  state: State;
  run: Run;
  onClose: () => void;
}) {
  const [opening, setOpening] = useState<string>();
  return (
    <>
      <div className="settings-intro">
        <div>
          <h1>Admin</h1>
          <p>Manage this Mac’s apps and review their Cloudflare deployments.</p>
        </div>
        <button
          className="settings-button"
          onClick={() => void run({ type: "revealWorkspace" })}
        >
          Show workspace in Finder
        </button>
      </div>
      <div className="admin-apps">
        {!state.apps.length && (
          <div className="settings-card">
            <p>Your apps will appear here once you create one.</p>
          </div>
        )}
        {state.apps.map((app) => {
          const runtime = state.runtimes[app.id];
          const deploy = state.publish[app.id];
          const history = deploy?.history ?? [];
          const live = history.find((h) => h.ok && h.url)?.url;
          return (
            <section key={app.id} className="settings-card">
              <div className="settings-row">
                <div>
                  <h2>{app.name}</h2>
                  <p>
                    {app.open
                      ? runtimeLabels[runtime?.status ?? "stopped"]
                      : "Closed"}{" "}
                    ·{" "}
                    {deploy?.status === "working"
                      ? (deploy.step ?? "Publishing")
                      : live
                        ? "Published"
                        : "Unpublished"}
                  </p>
                </div>
                <div className="settings-actions">
                  {live && (
                    <button
                      className="settings-button"
                      onClick={() => void run({ type: "open", url: live })}
                    >
                      Open live site
                    </button>
                  )}
                  <button
                    className="settings-button"
                    onClick={() => void run({ type: "reveal", appId: app.id })}
                  >
                    Files
                  </button>
                  <button
                    className="settings-button"
                    disabled={!!opening}
                    onClick={async () => {
                      setOpening(app.id);
                      try {
                        if (await run({ type: "openApp", appId: app.id }))
                          onClose();
                      } finally {
                        setOpening(undefined);
                      }
                    }}
                  >
                    Open in Studio
                  </button>
                </div>
              </div>
              {history.length > 0 && (
                <details className="deployment-history">
                  <summary>Deployment history</summary>
                  <ul>
                    {history.map((entry, i) => (
                      <li key={entry.at + i}>
                        <div>
                          <time dateTime={entry.at}>
                            {new Date(entry.at).toLocaleString()}
                          </time>
                          <span
                            className={entry.ok ? "soft" : "deployment-failed"}
                          >
                            {entry.ok ? entry.note : "Failed"}
                          </span>
                        </div>
                        {!entry.ok && <p>{entry.note}</p>}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </section>
          );
        })}
      </div>
    </>
  );
}

export function Settings({
  section,
  setSection,
  state,
  run,
  onClose,
}: {
  section: SettingsSection;
  setSection: (section: SettingsSection) => void;
  state: State;
  run: Run;
  onClose: () => void;
}) {
  const back = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    back.current?.focus();
  }, []);
  return (
    <div className="settings-page">
      <nav className="settings-nav" aria-label="Settings">
        <button ref={back} className="settings-back" onClick={onClose}>
          ← Back to Studio
        </button>
        <h2>Settings</h2>
        {(Object.keys(sections) as SettingsSection[]).map((item) => (
          <button
            key={item}
            aria-current={item === section ? "page" : undefined}
            onClick={() => setSection(item)}
          >
            {sections[item]}
          </button>
        ))}
        {state.version && (
          <span className="settings-version">mom-made {state.version}</span>
        )}
      </nav>
      <main className="settings-content" key={section}>
        {section === "assistants" && <Assistants state={state} run={run} />}
        {section === "cloudflare" && (
          <CloudflareSettings state={state} run={run} />
        )}
        {section === "admin" && (
          <Admin state={state} run={run} onClose={onClose} />
        )}
      </main>
    </div>
  );
}
