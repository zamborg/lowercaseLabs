import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type { AppRecord, Message } from "./contracts";
export class Store {
  db: DatabaseSync;
  constructor(readonly root: string) {
    mkdirSync(root, { recursive: true });
    this.db = new DatabaseSync(join(root, "studio.sqlite"));
    this.db.exec(
      `PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,at TEXT NOT NULL,app TEXT NOT NULL,type TEXT NOT NULL,payload TEXT NOT NULL);`,
    );
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
  messages() {
    return this.get<Message[]>("messages", []);
  }
  message(role: Message["role"], text: string, appId: string, turnId?: string) {
    const m = { id: randomUUID(), role, text, appId, turnId };
    this.db.exec("BEGIN");
    try {
      this.event(appId, "message", m);
      this.set("messages", [...this.messages(), m]);
      this.db.exec("COMMIT");
    } catch (e) {
      this.db.exec("ROLLBACK");
      throw e;
    }
    return m;
  }
  append(id: string, text: string) {
    const ms = this.messages();
    const m = ms.find((x) => x.id === id);
    if (m) {
      m.text += text;
      this.db.exec("BEGIN");
      try {
        this.event(m.appId, "message/delta", { id, text });
        this.set("messages", ms);
        this.db.exec("COMMIT");
      } catch (e) {
        this.db.exec("ROLLBACK");
        throw e;
      }
    }
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
