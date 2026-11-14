import { Hono } from "hono";
const app = new Hono<{ Bindings: { DB: D1Database } }>();
app.get("/api/health", (c) => c.json({ ok: true }));
app.get("/api/notes", async (c) => {
  await c.env.DB.prepare(
    "CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, text TEXT NOT NULL)",
  ).run();
  return c.json(
    (await c.env.DB.prepare("SELECT * FROM notes ORDER BY id DESC").all())
      .results,
  );
});
app.post("/api/notes", async (c) => {
  const body = await c.req.json();
  if (
    typeof body.text !== "string" ||
    !body.text.trim() ||
    body.text.length > 2000
  )
    return c.json({ error: "Write a note under 2,000 characters." }, 400);
  await c.env.DB.prepare(
    "CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, text TEXT NOT NULL)",
  ).run();
  await c.env.DB.prepare("INSERT INTO notes(text) VALUES(?)")
    .bind(body.text.trim())
    .run();
  return c.json({ ok: true }, 201);
});
export default app;
