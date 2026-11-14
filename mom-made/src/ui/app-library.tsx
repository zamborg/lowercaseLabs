import React, { useEffect, useRef, useState } from "react";
import type { Runtime, State } from "../core/contracts";

export const runtimeLabels: Record<Runtime["status"], string> = {
  ready: "Running",
  starting: "Starting",
  stopping: "Stopping",
  stopped: "Stopped",
  failed: "Needs attention",
};

export function AppLibrary({
  state,
  error,
  onOpen,
  onClose,
}: {
  state: State;
  error: string;
  onOpen: (appId: string) => Promise<boolean>;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [opening, setOpening] = useState<string>();
  const filter = query.trim().toLocaleLowerCase();
  const apps = state.apps
    .filter((app) => app.name.toLocaleLowerCase().includes(filter))
    .sort((a, b) => a.name.localeCompare(b.name));

  useEffect(() => {
    const el = dialog.current;
    el?.showModal();
    search.current?.focus();
    return () => el?.close();
  }, []);

  return (
    <dialog
      ref={dialog}
      className="app-library"
      aria-labelledby="app-library-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <div className="app-library-heading">
        <h2 id="app-library-title">Your apps</h2>
        <button className="quiet" onClick={onClose}>
          Close
        </button>
      </div>
      {state.apps.length > 0 && (
        <input
          ref={search}
          type="search"
          aria-label="Search apps"
          placeholder="Search apps"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      )}
      {apps.length > 0 ? (
        <ul className="app-library-list">
          {apps.map((app) => {
            const status = state.runtimes[app.id]?.status ?? "stopped";
            return (
              <li key={app.id}>
                <button
                  aria-current={app.id === state.selected ? "true" : undefined}
                  disabled={!!opening}
                  onClick={async () => {
                    setOpening(app.id);
                    try {
                      if (await onOpen(app.id)) onClose();
                    } finally {
                      setOpening(undefined);
                    }
                  }}
                >
                  <i
                    className={"dot " + (app.open ? status : "stopped")}
                    aria-hidden
                  />
                  <strong>{app.name}</strong>
                  <span>
                    {opening === app.id
                      ? "Opening…"
                      : app.open
                        ? runtimeLabels[status]
                        : "Closed"}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p>
          {state.apps.length
            ? "No apps match your search."
            : "Apps you build in Studio will appear here."}
        </p>
      )}
      {error && <p role="alert">{error}</p>}
    </dialog>
  );
}
