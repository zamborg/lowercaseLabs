import React, { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
function App() {
  const [notes, setNotes] = useState<{ id: number; text: string }[]>([]),
    [text, setText] = useState(""),
    [error, setError] = useState("");
  async function load() {
    const r = await fetch("/api/notes");
    if (!r.ok) throw Error("Could not load notes");
    setNotes(await r.json());
  }
  useEffect(() => {
    load().catch((e) => setError(e.message));
  }, []);
  return (
    <main>
      <div className="label">A LITTLE SOMETHING, MADE BY YOU</div>
      <h1>
        Good things
        <br />
        start small.
      </h1>
      <p>
        Your app is running. Tell mom-made what you want to make, or leave a
        little note to try the database.
      </p>
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          try {
            const r = await fetch("/api/notes", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text }),
            });
            if (!r.ok) throw Error("Could not save note");
            setText("");
            setError("");
            await load();
          } catch (e) {
            setError(String(e));
          }
        }}
      >
        <label htmlFor="note">A note for later</label>
        <div className="form">
          <input
            id="note"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Something worth keeping…"
            required
            maxLength={2000}
          />
          <button>Keep it ↗</button>
        </div>
      </form>
      <div role="alert">{error}</div>
      <ul>
        {notes.map((n) => (
          <li key={n.id}>{n.text}</li>
        ))}
      </ul>
      <footer>Saved on your Mac · ready for your next idea</footer>
    </main>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
