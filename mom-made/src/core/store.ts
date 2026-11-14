import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type {
  AppRecord,
  Feedback,
  Message,
  Publish,
  ActivityKind,
} from "./contracts";

type Extra = { kind?: ActivityKind; detail?: string; image?: string };

export class Store {
  db: DatabaseSync;
  constructor(readonly root: string) {
    mkdirSync(root, { recursive: true });
    this.db = new DatabaseSync(join(root, "studio.sqlite"));
    this.db.exec(
      `PRAGMA journal_mode=WAL;
       CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value TEXT NOT NULL);
       CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,at TEXT NOT NULL,app TEXT NOT NULL,type TEXT NOT NULL,payload TEXT NOT NULL);`,
    );
    this.migrate();
  }
  private migrate() {
    const v = (this.db.prepare("PRAGMA user_version").get() as {
      user_version: number;
    }).user_version;
    if (v < 1) {
      this.db.exec("BEGIN");
      try {
        this.db.exec(
          `CREATE TABLE IF NOT EXISTS messages(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,role TEXT NOT NULL,text TEXT NOT NULL,app TEXT NOT NULL,turn TEXT,kind TEXT,detail TEXT,image TEXT);
           CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY,app TEXT NOT NULL,body TEXT NOT NULL);
           CREATE TABLE IF NOT EXISTS deployments(seq INTEGER PRIMARY KEY AUTOINCREMENT,app TEXT NOT NULL,at TEXT NOT NULL,ok INTEGER NOT NULL,url TEXT,note TEXT NOT NULL);`,
        );
        const old = this.get<Message[]>("messages", []);
        const ins = this.db.prepare(
          "INSERT OR IGNORE INTO messages(id,role,text,app,turn) VALUES(?,?,?,?,?)",
        );
        for (const m of old) ins.run(m.id, m.role, m.text, m.appId, m.turnId ?? null);
        this.db.exec("DELETE FROM kv WHERE key='messages'");
        this.db.exec("PRAGMA user_version=1");
        this.db.exec("COMMIT");
      } catch (e) {
        this.db.exec("ROLLBACK");
        throw e;
      }
    }
  }
  get<T>(key: string, fallback: T): T {
    const r = this.db.prepare("SELECT value FROM kv WHERE key=?").get(key) as
      { value: string } | undefined;
    return r ? JSON.parse(r.value) : fallback;
  }
  set(key: string, value: unknown) {
    this.db
      .prepare("INSERT OR REPLACE INTO kv VALUES (?,?)")
      .run(key, JSON.stringify(value));
  }
  apps() {
    return this.get<AppRecord[]>("apps", []);
  }
  saveApp(app: AppRecord) {
    const apps = this.apps();
    const i = apps.findIndex((a) => a.id === app.id);
    if (i < 0) apps.push(app);
    else apps[i] = app;
    this.set("apps", apps);
  }
  messages(limit = 400, appId?: string): Message[] {
    const rows = this.db
      .prepare(
        `SELECT id,role,text,app AS appId,turn AS turnId,kind,detail,image FROM
         (SELECT * FROM messages ${appId === undefined ? "" : "WHERE app = ?"} ORDER BY seq DESC LIMIT ?) ORDER BY seq`,
      )
      .all(...(appId === undefined ? [limit] : [appId, limit])) as Record<string, string | null>[];
    return rows.map((r) => {
      const m: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(r)) if (v !== null) m[k] = v;
      return m as unknown as Message;
    });
  }
  private tx<T>(fn: () => T): T {
    this.db.exec("BEGIN");
    try {
      const r = fn();
      this.db.exec("COMMIT");
      return r;
    } catch (e) {
      this.db.exec("ROLLBACK");
      throw e;
    }
  }
  message(
    role: Message["role"],
    text: string,
    appId: string,
    turnId?: string,
    extra: Extra = {},
  ) {
    const m: Message = { id: randomUUID(), role, text, appId, turnId, ...extra };
    this.tx(() => {
      this.event(appId, "message", m);
      this.db
        .prepare(
          "INSERT INTO messages(id,role,text,app,turn,kind,detail,image) VALUES(?,?,?,?,?,?,?,?)",
        )
        .run(
          m.id,
          role,
          text,
          appId,
          turnId ?? null,
          extra.kind ?? null,
          extra.detail ?? null,
          extra.image ?? null,
        );
    });
    return m;
  }
  append(id: string, text: string) {
    this.tx(() => {
      const r = this.db
        .prepare("UPDATE messages SET text=text||? WHERE id=? RETURNING app")
        .get(text, id) as { app: string } | undefined;
      if (r) this.event(r.app, "message/delta", { id, text });
    });
  }
  update(id: string, patch: { text?: string; detail?: string }) {
    if (patch.text !== undefined)
      this.db.prepare("UPDATE messages SET text=? WHERE id=?").run(patch.text, id);
    if (patch.detail !== undefined)
      this.db
        .prepare("UPDATE messages SET detail=? WHERE id=?")
        .run(patch.detail, id);
  }
  feedback(): Feedback[] {
    return (
      this.db.prepare("SELECT body FROM feedback").all() as { body: string }[]
    )
      .map((r) => JSON.parse(r.body) as Feedback)
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }
  saveFeedback(f: Feedback) {
    this.db
      .prepare("INSERT OR REPLACE INTO feedback(id,app,body) VALUES(?,?,?)")
      .run(f.id, f.appId, JSON.stringify(f));
  }
  deployments(app: string): Publish["history"] {
    return (
      this.db
        .prepare(
          "SELECT at,ok,url,note FROM deployments WHERE app=? ORDER BY seq DESC LIMIT 20",
        )
        .all(app) as { at: string; ok: number; url: string | null; note: string }[]
    ).map((r) => ({
      at: r.at,
      ok: !!r.ok,
      url: r.url ?? undefined,
      note: r.note,
    }));
  }
  deployment(app: string, ok: boolean, url: string | undefined, note: string) {
    this.db
      .prepare("INSERT INTO deployments(app,at,ok,url,note) VALUES(?,?,?,?,?)")
      .run(app, new Date().toISOString(), ok ? 1 : 0, url ?? null, note);
  }
  event(app: string, type: string, payload: unknown) {
    this.db
      .prepare("INSERT INTO events(at,app,type,payload) VALUES(?,?,?,?)")
      .run(new Date().toISOString(), app, type, JSON.stringify(payload));
  }
  close() {
    this.db.close();
  }
}
