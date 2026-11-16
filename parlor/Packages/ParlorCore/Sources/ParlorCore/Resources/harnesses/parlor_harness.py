"""
parlor_harness.py: the helper every parlor harness is built on (standard library only).

A harness is a program that runs an agent's turns. parlor starts it once per agent and keeps it running. For each
message it sends one JSON line on stdin (turn.start); the program answers with JSON lines on stdout and ends the turn
with turn.done. This module owns that loop, the gateway client (streaming, cancel, priorities), tool calls, and the
helpers for history and compaction, so a harness is only "what to do in one turn".

    import parlor_harness as ph

    def turn(t):                       # t is a Turn
        text, _ = t.chat([{"role": "user", "content": "hi"}])
        t.message(text)

    ph.run(turn, actions={"compact": my_compact})

Everything printed to stderr lands in logs/harness-<agent>.log.
"""
import datetime, json, os, queue, re, sys, threading, time, urllib.request

PROTOCOL = 2
GATEWAY = os.environ.get("PARLOR_GATEWAY_URL") or os.environ.get("OPENAI_BASE_URL") or ""
TOKEN = os.environ.get("PARLOR_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""


class Cancelled(Exception):
    """The app cancelled the turn (Stop). End quietly."""


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


_inbox = queue.Queue()
_cancelled = set()


def _reader():
    for line in sys.stdin:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("type") == "turn.cancel":
            _cancelled.add(m.get("turn"))
        _inbox.put(m)
    _inbox.put(None)


threading.Thread(target=_reader, daemon=True).start()


def today():
    d = datetime.date.today()
    return "%s, %s %d, %d" % (d.strftime("%A"), d.strftime("%B"), d.day, d.year)


def strip_think(text):
    """Visible text: no <think> blocks (an unterminated one is cut), and a stray closing tag drops what is before it."""
    out = text
    while "<think>" in out:
        a = out.index("<think>")
        b = out.find("</think>", a)
        out = out[:a] if b < 0 else out[:a] + out[b + 8:]
    if "</think>" in out:
        out = out[out.index("</think>") + 8:]
    return out.strip()


def estimate_tokens(messages):
    """Rough prompt size before the model has reported a real count."""
    n = 0
    for m in messages:
        n += len(m.get("content") or "") // 3 + 8
        for c in m.get("tool_calls") or []:
            n += len(c["function"]["arguments"]) // 3
    return n


class Turn:
    def __init__(self, start):
        self.start = start
        self.id = start["turn"]
        self.agent = start["agent"]
        self.config = start.get("config") or {}
        self.settings = start.get("settings") or {}
        self.hosts = start.get("hosts") or {}
        self.tools = start.get("tools") or []
        self.transcript = start.get("transcript") or []
        self.participants = start.get("participants") or []
        self.memory = start.get("memory") or ""
        self.summary = start.get("summary")
        self.action = start.get("action")
        self.last_prompt_tokens = start.get("lastPromptTokens") or 0
        self.names = {p["agent"]: p.get("name") or p["agent"] for p in self.participants if p.get("agent")}
        self.names.setdefault(self.agent["id"], self.agent["name"])
        self.last_usage = None

    # -- settings --
    def opt(self, key, default=None):
        """A loop setting: the agent's config, else the machine's loop settings (context, compaction, steps), else default. Sampling is not here: send a key only when the agent's config sets it."""
        v = self.config.get(key)
        if v is None:
            v = self.settings.get(key)
        return default if v is None else v

    @property
    def others(self):
        return [(p["agent"], p.get("name") or p["agent"]) for p in self.participants
                if p.get("agent") and p["agent"] != self.agent["id"]]

    def has_host(self, need):
        return bool((self.hosts.get(need) or {}).get("model"))

    def has_tool(self, name):
        return any(t["name"] == name for t in self.tools)

    # -- events --
    @property
    def cancelled(self):
        return self.id in _cancelled

    def check(self):
        if self.cancelled:
            raise Cancelled()

    def status(self, text):
        send({"type": "status", "turn": self.id, "text": text})

    def delta(self, visible_text):
        send({"type": "text.delta", "turn": self.id, "text": visible_text})

    def message(self, text, tool_calls=None):
        m = {"type": "message", "turn": self.id, "role": "assistant", "text": text}
        if tool_calls:
            m["toolCalls"] = tool_calls
        send(m)

    def stats(self, usage, started):
        u = usage or {}
        done = u.get("completion_tokens", 0)
        send({"type": "stats", "turn": self.id, "promptTokens": u.get("prompt_tokens", 0), "completionTokens": done,
              "tokensPerSecond": done / max(time.time() - started, 0.001),
              "cachedTokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)})

    def summary_event(self, through, text, folded):
        send({"type": "summary", "turn": self.id, "through": through, "text": text, "folded": folded})

    # -- tools --
    def call_tool(self, name, arguments, call_id=None):
        """Ask the app to run a tool (allowlist, tier and the log are the app's). Returns the tool.result message."""
        call_id = call_id or "call-%s-%d" % (self.id[:8], int(time.time() * 1000) % 100000000)
        send({"type": "tool.call", "turn": self.id, "callID": call_id, "name": name, "arguments": arguments})
        while True:
            m = _inbox.get()
            if m is None:
                sys.exit(0)
            if m.get("type") == "turn.cancel" and m.get("turn") == self.id:
                raise Cancelled()
            if m.get("type") == "tool.result" and m.get("callID") == call_id:
                return m

    # -- the gateway --
    def chat(self, messages, host="chat", tools=None, stream=True, priority="user", visible=None, emit=True, **sampling):
        """One chat completion through the gateway. Sampling keys that are None are not sent, so the host's own
        defaults apply. Returns (text, tool_calls, usage); tool_calls are {"id", "name", "arguments": str}."""
        h = self.hosts.get(host) or {}
        if not h.get("model"):
            raise RuntimeError("this agent has no %s host" % host)
        body = {"model": h["model"], "messages": messages, "stream": stream}
        for k, v in sampling.items():
            if v is not None:
                body[k] = v
        if stream:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["schema"]}} for t in tools]
        if sampling.get("thinking") is not None:
            body.pop("thinking", None)
            body["chat_template_kwargs"] = {"enable_thinking": bool(sampling["thinking"])}
        req = urllib.request.Request((h.get("baseURL") or GATEWAY) + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json",
                                              "X-Parlor-Priority": priority})
        started = time.time()
        if not stream:
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                     for c in msg.get("tool_calls") or []]
            return strip_think(msg.get("content") or ""), calls, data.get("usage")
        raw, calls, usage = "", {}, None
        resp = urllib.request.urlopen(req, timeout=900)
        try:
            for line in resp:
                self.check()
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"].get("message", "the model failed"))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = ((chunk.get("choices") or [{}])[0]).get("delta", {})
                if delta.get("content"):
                    raw += delta["content"]
                    shown = visible(raw) if visible else strip_think(raw)
                    if emit and shown:
                        self.delta(shown)
                for tc in delta.get("tool_calls", []):
                    c = calls.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    c["name"] += tc.get("function", {}).get("name", "")
                    c["arguments"] += tc.get("function", {}).get("arguments", "")
        finally:
            resp.close()   # a closed connection is how the gateway learns to stop generating
        self.last_usage = usage
        if emit:
            self.stats(usage, started)
        text = visible(raw) if visible else strip_think(raw)
        return text, [calls[i] for i in sorted(calls)], usage

    def complete(self, messages, host="chat", **kw):
        """A small non-streaming call (summaries, art direction), low priority by default."""
        kw.setdefault("priority", "background")
        text, _, _ = self.chat(messages, host=host, stream=False, **kw)
        return text


# ---------------------------------------------------------------- history and compaction helpers

def real(transcript):
    return [m for m in transcript if m.get("role") != "marker"]


def after_summary(transcript, summary):
    live = real(transcript)
    if summary and summary.get("through"):
        for i in range(len(live) - 1, -1, -1):
            if live[i].get("id") == summary["through"]:
                return live[i + 1:]
    return live


def fold(t, keep_recent=6, summary_tokens=500):
    """Fold older messages into the rolling summary. Returns (text, through_id, folded_count) or None."""
    msgs = real(t.transcript)
    if len(msgs) <= keep_recent + 1:
        return None
    cut = len(msgs) - keep_recent
    while cut > 0 and msgs[cut].get("role") != "user":
        cut -= 1
    if cut <= 0:
        return None
    old = msgs[:cut]
    lines = []
    for m in old:
        role = m.get("role")
        if role == "user":
            lines.append("User: %s" % m.get("text", ""))
        elif role == "assistant":
            who = t.names.get(m.get("speaker") or "", t.agent["name"])
            if m.get("text"):
                lines.append("%s: %s" % (who, m["text"]))
            for c in m.get("toolCalls") or []:
                lines.append("(%s called %s)" % (who, c["name"].replace("__", ".", 1)))
        elif role == "tool":
            lines.append("(tool %s returned: %s)" % (m.get("toolName") or "", (m.get("text") or "")[:300]))
    prior = ("Existing summary:\n%s\n\n" % t.summary["text"]) if t.summary else ""
    prompt = ("%sNew conversation to fold in:\n%s\n"
              "Write an updated summary of the whole conversation so far for %s to continue from. "
              "Keep durable facts, decisions, preferences, names, dates, open tasks, and the tone of the relationship. "
              "Drop pleasantries. Under 250 words. Plain prose or short bullets.\n") % (prior, "\n".join(lines) + "\n", t.agent["name"])
    text = t.complete([{"role": "system", "content": "You write faithful, compact conversation summaries."},
                       {"role": "user", "content": prompt}], temperature=0.2, max_tokens=summary_tokens, thinking=False)
    if not text:
        return None
    through = old[-1].get("id")
    return text, through, cut


# ---------------------------------------------------------------- the loop

def run(turn_fn, actions=None):
    """Read turns forever. `actions` maps an action id (declared in harness.json) to a function(Turn)."""
    actions = actions or {}
    while True:
        msg = _inbox.get()
        if msg is None:
            break
        if msg.get("type") != "turn.start":
            continue
        t = Turn(msg)
        try:
            if t.action:
                fn = actions.get(t.action.get("id"))
                if not fn:
                    raise RuntimeError("this harness has no action %r" % t.action.get("id"))
                fn(t)
            else:
                turn_fn(t)
            send({"type": "turn.done", "turn": t.id})
        except Cancelled:
            send({"type": "turn.done", "turn": t.id})
        except SystemExit:
            raise
        except Exception as e:  # noqa
            log("turn failed:", repr(e))
            send({"type": "error", "turn": t.id, "message": str(e)})
