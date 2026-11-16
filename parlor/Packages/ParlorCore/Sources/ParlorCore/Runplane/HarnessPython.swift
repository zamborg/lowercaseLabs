import Foundation

/// The Python that ships with parlor: the harness helper and the two harnesses built on it. Written to
/// `~/parlor/harnesses/` with the managed marker, so an edited copy is never overwritten.
enum HarnessPython {
    static let marker = "parlor-harnesses-v2"

    static let header = "# \(marker): managed by parlor. Copy this file to make your own; edits here are overwritten.\n"

    // MARK: parlor_harness.py

    static let helper = header + #"""
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
            """A harness setting: the agent's config, else the machine's loop settings, else default."""
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
    """#

    // MARK: chat.py

    static let chat = header + #"""
    """
    chat: the default harness. A helpful assistant with tools, memory and a rolling summary.

    Per turn: compact if the prompt is near the context limit, then stream a reply, run any tool calls the model makes
    (parlor runs them and enforces the agent's tool list and tier), feed the results back, and repeat until the model
    answers in plain text. Edits to this file are overwritten when parlor updates; copy the folder to change it.
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import json
    import parlor_harness as ph


    def request_messages(t, summary):
        a = t.agent
        system = (a.get("persona") or "").strip() or "You are %s." % a["name"]
        # Date only: a clock in the system prompt would change every minute and defeat the prompt cache.
        system += "\n\nToday is %s." % ph.today()
        others = t.others
        if others:
            system += ("\nYou are %s, in a conversation with the user and %s. Reply only as %s: never write other participants' "
                       "lines and never start your reply with your own name.") % (a["name"], ", ".join(n for _, n in others), a["name"])
        if t.tools:
            system += ("\nYou have tools. When a request needs information or an action from a tool, call the tool immediately in this reply. "
                       "Never say you will look something up or do something without actually calling the tool. "
                       "Never ask permission to use a tool for a read-only lookup. After the tool result arrives, answer in plain text.")
        if t.has_tool("parlor__memory.append"):
            system += "\nYou can keep notes across conversations with parlor.memory.append: save lasting facts about the user and your work, not chatter."
        mem = (t.memory or "").strip()
        if mem:
            system += "\n\n## What you remember\n" + mem
        if summary:
            system += "\n\n## Earlier conversation (summary)\n" + summary["text"]
        live = ph.after_summary(t.transcript, summary)
        out = [{"role": "system", "content": system}]

        def assistant(m):
            e = {"role": "assistant", "content": m.get("text") or ""}
            if m.get("toolCalls"):
                e["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}} for c in m["toolCalls"]]
            return e

        def tool(m):
            return {"role": "tool", "tool_call_id": m.get("toolCallID"), "content": m.get("text") or ""}

        if not others:
            for m in live:
                r = m.get("role")
                if r == "user":
                    out.append({"role": "user", "content": m.get("text") or ""})
                elif r == "assistant":
                    out.append(assistant(m))
                elif r == "tool":
                    out.append(tool(m))
            return out
        # Group framing: this agent's own lines stay assistant lines; everyone else's final replies become attributed
        # user lines, and their tool steps are left out. Neighbouring user lines are merged.
        names = dict(others)

        def add_user(text):
            if out[-1]["role"] == "user" and len(out) > 1:
                out[-1]["content"] += "\n" + text
            else:
                out.append({"role": "user", "content": text})

        me = a["id"]
        for m in live:
            who = m.get("speaker") or ("user" if m.get("role") == "user" else me)
            r = m.get("role")
            if r == "user":
                add_user("User: " + (m.get("text") or ""))
            elif r == "assistant":
                if who == me:
                    out.append(assistant(m))
                elif not m.get("toolCalls") and m.get("text"):
                    add_user("%s: %s" % (names.get(who, who), m["text"]))
            elif r == "tool" and who == me:
                out.append(tool(m))
        if out[-1]["role"] == "assistant":
            add_user("(Carry on the conversation.)")
        return out


    def sampling(t):
        return dict(temperature=t.config.get("temperature"), top_p=t.opt("top_p"),
                    repetition_penalty=t.config.get("repetition_penalty"),
                    max_tokens=(t.opt("max_reply_tokens") or None), thinking=bool(t.config.get("thinking")))


    def compact_if_needed(t, summary):
        context = int(t.opt("context", 16384))
        limit = float(t.opt("compact_at", 0.7)) * context
        estimate = ph.estimate_tokens(request_messages(t, summary)[:])
        if max(t.last_prompt_tokens, estimate) <= limit:
            return summary
        t.status("Compacting")
        got = ph.fold(t, int(t.opt("keep_recent", 6)), int(t.opt("summary_tokens", 500)))
        if not got:
            return summary
        text, through, folded = got
        t.summary_event(through, text, folded)
        t.summary = {"through": through, "text": text}
        return t.summary


    def turn(t):
        summary = compact_if_needed(t, t.summary)
        t.status("Thinking")
        messages = request_messages(t, summary)
        max_steps = max(1, int(t.opt("max_steps", 8)))
        for step in range(max_steps):
            t.check()
            offered = [] if step == max_steps - 1 else t.tools   # the last step withholds tools so the model has to answer
            text, calls, _ = t.chat(messages, tools=offered, **sampling(t))
            t.message(text, [{"id": c["id"], "name": c["name"], "arguments": json.loads(c["arguments"] or "{}")} for c in calls])
            entry = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for c in calls]
            messages.append(entry)
            if not calls:
                return
            for c in calls:
                t.check()
                r = t.call_tool(c["name"], json.loads(c["arguments"] or "{}"), c["id"])
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": r.get("text") or ""})


    def compact_now(t):
        t.status("Compacting")
        got = ph.fold(t, int(t.opt("keep_recent", 6)), int(t.opt("summary_tokens", 500)))
        if got:
            text, through, folded = got
            t.summary_event(through, text, folded)


    ph.run(turn, actions={"compact": compact_now})
    """#

    // MARK: character.py

    static let character = header + #"""
    """
    character: plays the agent as a person in a story, not an assistant, and works out when a picture is worth taking.

    Config (all optional): appearance, scenario, user_name, images (auto | ask | off), image_every, image_style,
    image_prompt_style (natural | tags), image_detail (fast | detailed), seed. Pictures need the agent to have an image host
    and the image.generate tool. Memory is part of the character card.

    Actions: "image" draws the current scene, "selfie" draws the character looking at the camera. Neither adds a user line.
    Edits to this file are overwritten when parlor updates; copy the folder to change it.
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import hashlib, json, re
    import parlor_harness as ph

    IMAGE_TOOL = "image__generate"

    STYLE_RULES = """How to play {name}
    - You are {name}. Never say you are an AI, a model or a character, and never mention prompts, instructions or these rules.
    - Talk like a person in the moment, not an assistant: no lists, no headings, no summaries, no "As {name}", no offering help unless it is natural for {name}.
    - Speech is plain text. Put small physical beats (a look, a gesture, the room) in *italics*, one or two at most per reply.
    - Match the length of what you were sent. A casual line gets one to three sentences. Never write more than about 120 words.
    - Keep the conversation open: react to what they said, then add one new thing (a detail, a feeling, a small move, one question) so they have something to answer.
    - Never speak, act or decide for the person you are talking to.
    - Keep continuity: names, places, promises and what has happened stay true. Do not repeat phrases or openings from your earlier replies. Mood shifts gradually.
    - Stay in {name}'s voice: the vocabulary, rhythm and habits from the card.
    - If they step out of the story with (OOC: ...), answer briefly out of character, then go back in.
    - Pictures are handled for you. Never write image links or prompts, and never say you cannot send pictures. If they ask for a photo, answer as {name} would while sending one."""

    ART_DIRECTOR = """You are the art director for a character chat. Decide whether the latest moment deserves a picture and, if it does, write the prompt for a local text-to-image model (Z-Image Turbo).

    Reply with JSON only: {"image": true or false, "prompt": "...", "shape": "portrait" or "square" or "landscape"}

    image is true when the user asked to see something or someone (a photo, a selfie, "show me", "what do you look like"), or when the scene clearly changed place, outfit, mood or action in a way worth seeing. Small talk is false.

    When true, the prompt is one paragraph of 45 to 90 words in plain descriptive English: no lists, no quotes, no negations (never "no", "without", "not"), no text or logos in the picture. Order it: the subject and exact look, expression and pose, clothing, setting and time of day, light, camera (lens, distance, angle), style.
    Always restate the character's fixed appearance in the same words every time (including gender and age) so the face stays recognisable. Show only what is happening in the story. For a selfie, frame the character holding a phone toward the camera, close and a little off-centre. Use portrait for one person and landscape for places and wide scenes."""

    ART_DIRECTOR_TAGS = """You are the art director for a character chat. Decide whether the latest moment deserves a picture and, if it does, write the prompt for an anime-style Stable Diffusion XL model trained on Danbooru tags (Illustrious).

    Reply with JSON only: {"image": true or false, "prompt": "...", "shape": "portrait" or "square" or "landscape"}

    image is true when the user asked to see something or someone (a photo, a selfie, "show me", "what do you look like"), or when the scene clearly changed place, outfit, mood or action in a way worth seeing. Small talk is false.

    When true, the prompt is a single line of Danbooru-style tags separated by commas, lowercase, no sentences, no negations. Start with the subject count (1girl, 1boy, 2girls), then the character's fixed appearance as tags (hair colour and length and style, eye colour, skin, build, distinguishing marks), then expression, clothing, pose and action, setting, time of day, lighting, and last the framing (portrait, upper body, cowboy shot, from side, close-up, looking at viewer). 25 to 45 tags. Use underscores-free tags with spaces, like "short hair" and "green eyes". Always restate the character's fixed appearance with the same tags every time so the character stays recognisable. Show only what is happening in the story. Use portrait for one person and landscape for places and wide scenes.
    Do not include quality tags (masterpiece, best quality); the app adds them."""

    TAG_MODEL = re.compile(r"illustrious|wai|pony|noob|sdxl|animagine|anime|checkpoint|\.safetensors", re.I)
    ASKED = re.compile(r"\b(selfie|photo|photograph|pic|picture|pics|show me|let me see|see you|draw|what do you look like|your face)\b", re.I)


    def clean(text, name):
        """Visible text: no think blocks, no leading 'Name:' label."""
        text = ph.strip_think(text)
        return re.sub(r"^\s*" + re.escape(name) + r"\s*:\s*", "", text.lstrip())


    def system_prompt(t, photo_now=False):
        a = t.agent
        name = a["name"]
        parts = [(a.get("persona") or "You are %s." % name).strip()]
        if t.config.get("appearance"):
            parts.append("## What %s looks like (never contradict this)\n%s" % (name, t.config["appearance"].strip()))
        who = t.config.get("user_name") or "the person you are talking to"
        parts.append("## Who you are talking to\nThey are %s." % who)
        if t.config.get("scenario"):
            parts.append("## The scene\n" + t.config["scenario"].strip())
        if t.memory:
            parts.append("## What you remember\n" + t.memory.strip())
        others = [n for _, n in t.others]
        if others:
            parts.append("## Others here\n" + ", ".join(others) + " are also in this conversation. Lines from them are labelled with their names.")
        parts.append(STYLE_RULES.format(name=name))
        if photo_now:
            parts.append("Right now: they asked to see you, and a photo of you is being sent with this reply. Say something to go with it "
                         "(tease, pose, comment on the shot if it suits you), but you are sending it. Do not refuse.")
        parts.append("Today is " + ph.today() + ".")
        return "\n\n".join(parts)


    def history(t, budget_chars):
        """The conversation as chat messages: tool traffic dropped, other speakers labelled, oldest trimmed to fit."""
        me = t.agent["id"]
        out = []
        for m in ph.real(t.transcript):
            text = (m.get("text") or "").strip()
            if m["role"] == "user" and text:
                out.append({"role": "user", "content": text})
            elif m["role"] == "assistant" and text:
                who = m.get("speaker") or me
                if who in ("user", me):
                    out.append({"role": "assistant", "content": text})
                else:
                    out.append({"role": "user", "content": "%s: %s" % (t.names.get(who, who), text)})
        kept, used = [], 0
        for m in reversed(out):
            used += len(m["content"]) + 8
            if used > budget_chars and kept:
                break
            kept.append(m)
        return list(reversed(kept))


    def wants_picture(t, user_text):
        mode = (t.config.get("images") or "auto").lower()
        if mode == "off" or not t.has_tool(IMAGE_TOOL) or not t.has_host("image"):
            return False, False
        asked = bool(ASKED.search(user_text))
        if asked or mode == "ask":
            return asked, asked
        since = 0
        for m in reversed(ph.real(t.transcript)):
            if m.get("images"):
                break
            if m["role"] == "assistant" and (m.get("text") or "").strip():
                since += 1
        return since + 1 >= int(t.config.get("image_every") or 4), False


    def tag_mode(t):
        mode = (t.config.get("image_prompt_style") or "").lower()
        if mode in ("tags", "natural"):
            return mode == "tags"
        return bool(TAG_MODEL.search((t.hosts.get("image") or {}).get("model") or ""))


    def art_direct(t, reply, forced, hint=""):
        """One small model call: should there be a picture, and what is it? Returns (prompt, shape) or None."""
        a = t.agent
        tags = tag_mode(t)
        recent = history(t, 2500)[-6:]
        convo = "\n".join(("%s: %s" % ("User" if m["role"] == "user" else a["name"], m["content"])) for m in recent)
        appearance = t.config.get("appearance") or ""
        style = t.config.get("image_style") or ("" if tags else "natural photograph, soft cinematic light, shallow depth of field, 85mm lens")
        user = ("Character: %s\nFixed appearance: %s\nVisual style: %s\n\nRecent conversation:\n%s\n%s: %s\n\n%s"
                % (a["name"], appearance or "(not described; invent one detail-rich look and keep it)", style or "(the model's own)", convo,
                   a["name"], reply, ("The user asked to see something, so image must be true. " + hint).strip() if forced else "Decide."))
        raw = t.complete([{"role": "system", "content": ART_DIRECTOR_TAGS if tags else ART_DIRECTOR}, {"role": "user", "content": user}],
                         temperature=0.3, max_tokens=360)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            data = {}
        prompt = (data.get("prompt") or "").strip()
        if not (data.get("image") is True or forced) or (not prompt and not forced):
            return None
        if not prompt:  # forced but the model gave nothing usable: build one from what we know
            prompt = ("1girl, solo, %s, looking at viewer" % (appearance or a["name"])) if tags else \
                "%s. %s. %s." % (appearance or a["name"], reply.replace("*", "")[:200], style)
        shape = data.get("shape") if data.get("shape") in ("portrait", "square", "landscape") else "portrait"
        return prompt, shape


    def take_picture(t, reply, forced, hint=""):
        try:
            t.status("Framing a picture")
            shot = art_direct(t, reply, forced, hint)
        except ph.Cancelled:
            raise
        except Exception as e:  # a failed art director never breaks the chat
            ph.log("art director failed:", e)
            return
        if not shot:
            return
        prompt, shape = shot
        seed_cfg = str(t.config.get("seed", ""))
        seed = int(seed_cfg) if seed_cfg.isdigit() else int(hashlib.sha1(t.agent["id"].encode()).hexdigest()[:8], 16) % 100000
        call = {"id": "pic-" + t.id, "name": IMAGE_TOOL,
                "arguments": {"prompt": prompt, "shape": shape, "detail": t.config.get("image_detail") or ("detailed" if tag_mode(t) else "fast"), "seed": seed}}
        ph.log("picture:", prompt)
        t.status("Taking a picture")
        t.message("", [call])
        r = t.call_tool(call["name"], call["arguments"], call["id"])
        if r.get("isError"):
            ph.log("image tool error:", r.get("text"))


    def turn(t):
        a = t.agent
        name = a["name"]
        context = int(t.opt("context", 8192))
        budget = context * 3 - 3500
        user_text = next((m["text"] for m in reversed(ph.real(t.transcript)) if m["role"] == "user"), "")
        consider, forced = wants_picture(t, user_text)
        sysmsg = system_prompt(t, photo_now=forced)
        messages = [{"role": "system", "content": sysmsg}] + history(t, max(2000, budget - len(sysmsg)))
        t.status("%s is typing" % name)
        reply, _, _ = t.chat(messages, visible=lambda raw: clean(raw, name), temperature=t.config.get("temperature"),
                             top_p=t.config.get("top_p"), repetition_penalty=t.config.get("repetition_penalty"),
                             max_tokens=t.config.get("max_reply_tokens") or 500)
        t.message(reply)
        if consider and reply:
            take_picture(t, reply, forced)


    def last_scene(t):
        me = t.agent["id"]
        for m in reversed(ph.real(t.transcript)):
            if m["role"] == "assistant" and (m.get("text") or "").strip() and (m.get("speaker") or me) == me:
                return m["text"].strip()
        return ""


    def act_image(t):
        take_picture(t, last_scene(t), True, "Picture the current scene.")


    def act_selfie(t):
        take_picture(t, last_scene(t), True, "Make it a selfie: the character holding a phone toward the camera.")


    ph.run(turn, actions={"image": act_image, "selfie": act_selfie})
    """#
}
