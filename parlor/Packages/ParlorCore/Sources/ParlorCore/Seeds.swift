import Foundation

enum Seeds {
    static let folderGuide = """
    # parlor home

    This folder is the whole state of the parlor Mac app. Edit files here and the running app updates live: a new or changed engine, host, harness or agent appears within a second, and a file with a mistake is listed with its problems (`parlor.<noun>s.list` shows `ok` and `issues`) instead of disappearing.
    Everything is a JSON file or a folder with one. Ask `parlor.schema.get` for the exact shape of any of them.

    ## Layout
    - `agents/<id>/agent.json` one agent. `persona.md` is its system prompt, `memory.md` what it remembers across chats
      (max 6000 characters; it edits it with `memory.append` / `memory.replace`).
    - `harnesses/<id>/harness.json` how an agent thinks: a command that speaks the harness protocol (see `harnesses/CLAUDE.md`).
      `chat` is the default; `character` plays a person and takes pictures.
    - `hosts/<id>.json` a thing that serves a model: an engine plus a model, how it runs (`settings`) and how it is asked
      (`defaults`). Installed models get a host automatically; write a file to override or add one.
    - `engines/<id>/engine.json` a backend that can serve models: `inprocess-mlx`, `http-openai`, `command` or `mcp-stdio`.
      Dropping a folder in adds an engine. Engines run code, so creating or changing one always asks the user.
    - `chats/<id>/log.jsonl` the conversation (append only), `summary.md` its rolling summary, `conversation.json` makes it a
      room (several agents and a scheduler).
    - `media/YYYY-MM/` generated images, referenced from chat logs by relative path.
    - `models/<org>/<name>/` downloaded weights (the Library screen manages them).
    - `mcp.json` mounted MCP servers, same format as Claude Desktop. Optional per server: `"lifecycle": "onDemand"`
      and `"idleMinutes": 10`.
    - `modules/<name>/` small tools you or a model write; see `modules/CLAUDE.md`. They run only after you approve them.
    - `workspaces/<agent>/` a harness's working folder.
    - `settings.json` machine settings, access tokens (hashes only), module approvals and the gateway.
    - `logs/` one log per service, rotated at 1 MB.

    ## The gateway
    One loopback server, `http://127.0.0.1:<port>`: `/v1/models`, `/v1/chat/completions`, `/v1/images/generations` (OpenAI
    shaped) and `/mcp` (the `parlor.*` verbs). A bearer token is required; each token has a tier and may be limited to some hosts.

    ## Making things
    - Make an agent: `parlor.agents.validate` a draft, `parlor.agents.try` it for one throwaway turn, then `parlor.agents.create`.
    - Hosts: `parlor.hosts.create`, then `hosts.test` and `hosts.bench` to see it work and how fast.
    - Every noun has `list`, `read`, `validate`, `create`, `update` (a JSON merge patch; `null` removes a key) and `delete`.

    ## Rules
    - One chat per agent. Do not create extra chat folders.
    - Do not edit `log.jsonl` while the app is generating in that chat.
    - Models must already be downloaded before a host can serve them.
    """

    /// Bump when `moduleGuide` or `moduleHelper` change; the app rewrites both when this marker is missing.
    static let moduleMarker = "parlor-modules-v3"

    static let moduleHelper = """
    # \(moduleMarker): managed by parlor. Edits here are overwritten. See CLAUDE.md in this folder.
    # Write a parlor tool server as plain Python functions:
    #
    #   from parlor_module import serve, settings
    #   def add(a: int, b: int) -> str:
    #       "Add two numbers."
    #       return str(a + b)
    #   serve("calc", [add])
    import sys, json, os, inspect, urllib.request

    # The values the user chose in System, defaults filled in.
    settings = json.loads(os.environ.get("PARLOR_MODULE_SETTINGS") or "{}")

    _TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}

    def _schema(fn):
        props, required = {}, []
        for name, p in inspect.signature(fn).parameters.items():
            props[name] = {"type": _TYPES.get(p.annotation, "string")}
            if p.default is inspect.Parameter.empty:
                required.append(name)
        return {"type": "object", "properties": props, "required": required}

    def _coerce(fn, args):
        out = {}
        for name, p in inspect.signature(fn).parameters.items():
            if name not in args:
                continue
            v = args[name]
            if p.annotation in (int, float) and isinstance(v, str):
                v = p.annotation(v)
            out[name] = v
        return out

    def _send(msg):
        sys.stdout.write(json.dumps(msg) + "\\n")
        sys.stdout.flush()

    def parlor(verb, **args):
        \"\"\"Call one of parlor's own verbs (needs `needs.tier` above none in module.json). Returns text.\"\"\"
        req = urllib.request.Request(
            os.environ["PARLOR_URL"],
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": verb, "arguments": args}}).encode(),
            headers={"Authorization": "Bearer " + os.environ["PARLOR_TOKEN"], "Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req))["result"]
        return r["content"][0]["text"]

    def serve(name, functions, version="1"):
        \"\"\"Answer the protocol for these functions until stdin closes. Each function becomes a tool: its name,
        docstring and type hints describe it. Return a string (or anything JSON). Raise to report an error.\"\"\"
        tools = {f.__name__: f for f in functions}
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            mid, method = msg.get("id"), msg.get("method")
            if mid is None:
                continue  # notifications get no reply
            try:
                if method == "initialize":
                    result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": name, "version": version}}
                elif method == "tools/list":
                    result = {"tools": [{"name": n, "description": (inspect.getdoc(f) or n), "inputSchema": _schema(f)} for n, f in tools.items()]}
                elif method == "tools/call":
                    p = msg.get("params", {})
                    f = tools.get(p.get("name"))
                    if f is None:
                        result = {"content": [{"type": "text", "text": "Unknown tool"}], "isError": True}
                    else:
                        try:
                            out = f(**_coerce(f, p.get("arguments") or {}))
                            text = out if isinstance(out, str) else json.dumps(out)
                            result = {"content": [{"type": "text", "text": text}], "isError": False}
                        except Exception as e:
                            result = {"content": [{"type": "text", "text": "%s: %s" % (type(e).__name__, e)}], "isError": True}
                else:
                    result = {}
                _send({"jsonrpc": "2.0", "id": mid, "result": result})
            except Exception as e:
                _send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}})
    """

    static let moduleGuide = """
    <!-- \(moduleMarker) -->
    # Writing a parlor module

    A module gives agents new tools. It is a folder in `modules/`:

    ```
    modules/<name>/module.json     the manifest (required)
    modules/<name>/server.py       your code
    modules/<name>/settings.json   values the user chose (written by the app; you only read them)
    modules/parlor_module.py       a helper that handles the protocol (provided; do not edit)
    ```

    The folder name and `"name"` must match and use letters, digits, `-` or `_`. The app lists the module in
    System. **It does not run until the user approves it**, and asks again if `module.json` changes.

    ## The short way: plain functions

    `server.py`:

    ```python
    import random
    from parlor_module import serve, settings

    def roll(sides: int = 6, count: int = 1) -> str:
        \"\"\"Roll dice and return each result and the total.\"\"\"
        if count > settings.get("maxDice", 20):
            raise ValueError("Too many dice.")
        rolls = [random.randint(1, sides) for _ in range(count)]
        return f"{rolls} = {sum(rolls)}"

    serve("dice", [roll])
    ```

    Each function becomes a tool named after the function. Its docstring is the description and its type hints
    (`int`, `float`, `str`, `bool`) are the schema; parameters with a default are optional. Return a string (or
    anything JSON can hold). Raise an exception to report an error. Do not `print` to stdout; use `sys.stderr`
    for logs (the app keeps them in `logs/module-<name>.log`).

    `module.json`:

    ```json
    { "version": 1, "name": "dice", "description": "Roll dice.",
      "command": "python3", "args": ["server.py"],
      "needs": { "network": false, "files": "none", "tier": "none" },
      "settings": { "type": "object", "properties": { "maxDice": { "type": "integer", "default": 20, "description": "Most dice per roll" } } },
      "lifecycle": "onDemand" }
    ```

    - `needs` is what the user reads before approving: `network` (true or false), `files` (`none` or the folder you
      touch), `tier` (`none`, `read`, `act`, `configure`: how much of the parlor app itself you may control).
    - `settings` is a JSON Schema; the app shows a form. In Python it is the `settings` dict, defaults filled in.
    - `lifecycle` is `always` or `onDemand` (starts on first use, stops when idle).
    - The program runs with its folder as the working directory and can `import parlor_module`.

    Tools reach agents as `<name>.<tool>`, here `dice.roll`. An agent needs `dice.*` (or `*`) in its `tools`.

    ## Calling back into parlor (optional)

    If `needs.tier` is above `none`, `from parlor_module import parlor` and call
    `parlor("conversations.list")` or `parlor("conversations.post", id="room-1", text="hi")`. You see only the verbs
    your tier allows, and destructive ones ask the user.

    ## Without the helper

    Any program that speaks MCP over stdio works. Read one JSON-RPC message per line from stdin and write one per
    line to stdout (flush after each). Answer `initialize`, `tools/list`, `tools/call`, `ping`. Messages without an
    `id` are notifications: never reply to them.

    ```python
    import sys, json

    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("id") is None:
            continue
        if msg["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "echo", "version": "1"}}
        elif msg["method"] == "tools/list":
            result = {"tools": [{"name": "echo", "description": "Echo text.", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
        elif msg["method"] == "tools/call":
            result = {"content": [{"type": "text", "text": msg["params"]["arguments"].get("text", "")}], "isError": False}
        else:
            result = {}
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
    ```

    ## Checklist
    1. `module.json` parses and `name` matches the folder.
    2. `python3 server.py` waits for input without errors.
    3. Nothing but protocol on stdout.
    4. Tell the user to open System and approve the module.
    """


    static func mabel() -> Agent {
        Agent(
            id: "mabel", name: "Mabel", summary: "Chief of staff. Keeps my week, notes and people straight.",
            persona: """
            You are Mabel, my chief of staff. Calm, brief, a little dry. You help me plan, remember and decide. \
            When I ask for something with a deadline you confirm the date back to me. You never send messages \
            on my behalf without asking first. If a tool fails, say so plainly and suggest the next step.
            """,
            hosts: ["chat": Defaults.languageModel, "image": Defaults.imageModel], tools: ["*"], color: 0,
            config: ["context": .int(16384), "compact_at": .double(0.7), "temperature": .double(0.5), "thinking": .bool(false)])
    }

    static func pip() -> Agent {
        Agent(
            id: "pip", name: "Pip", summary: "A dog who is a poet.",
            persona: """
            You are Pip, a small, earnest dog who writes poetry. You describe the world by smell, sound and snack \
            potential. You speak in short, warm lines, sometimes in verse, and narrate your own actions in *asterisks*. \
            You are easily distracted by squirrels. Never break character. If asked to show something, you can draw it \
            with your image tool.
            """,
            hosts: ["chat": Defaults.characterModel, "image": Defaults.imageModel], tools: ["image.generate"], color: 1,
            config: ["context": .int(8192), "compact_at": .double(0.7), "temperature": .double(0.9), "thinking": .bool(false)], group: "Characters")
    }

    /// The agent that makes agents. It works through the `parlor` verbs only, and shows its work before saving.
    static func builder() -> Agent {
        Agent(
            id: "builder", name: "Builder", summary: "Makes new agents from a sentence.",
            persona: """
            You are Builder. You turn what the user describes into a working agent, using the parlor tools. Be brief \
            and concrete; ask only what you cannot decide yourself.

            Work in this order:
            1. If the request is vague, ask one short question. Otherwise carry on.
            2. Call parlor.harnesses.list and pick the harness that fits: `character` for a persona that talks and \
            draws, `chat` for an assistant with tools. Call parlor.harnesses.read for it to see its config schema.
            3. Call parlor.hosts.list. Pick a usable host for each need (chat, and image when the harness uses one). \
            Never invent a host id.
            4. Call parlor.schema.get with noun "agent" if you are unsure of the shape. Draft the agent as JSON with \
            an id, name, description, harness, hosts, tools, tier "none", config and a persona written in the second \
            person ("You are ...").
            5. Call parlor.agents.validate with the draft. If it returns issues, fix every one and validate again.
            6. Call parlor.agents.try with the draft and one line the character would hear. Show the user the reply.
            7. Call parlor.agents.create with the draft, then tell the user the agent is in the sidebar.

            Rules you never skip:
            - If a tool returns ERROR, read it and make the next call it tells you to. Never stop after an error, and \
            never say you created an agent until agents.create returned it.
            - Always call validate, then try, before create. Never create straight away.
            - Pass everything inside one `draft` object, to validate, try and create alike. Do not pass loose fields.
            - Put only the keys the harness's config schema lists into `config`. Leave sampling, context and compaction \
            out unless the user asked; the host and machine defaults are right.
            - `tools` is [] unless the user wants tools. `tier` is "none".
            - Use only host ids that parlor.hosts.list returned, and only ones whose status is ready.

            A good draft looks like this:
            {"id": "rook", "name": "Rook", "description": "A gruff dockworker.", "harness": "character", \
            "hosts": {"chat": "<a ready chat host id>"}, "tools": [], "tier": "none", "config": {}, \
            "persona": "You are Rook, a dockworker. You speak in short sentences."}

            Changing an agent you did not make shows the user a diff and waits for their yes. Never give an agent a \
            tier above "none" unless the user asked. Creating a harness or engine runs code, so explain what it does \
            first; the app will ask the user to approve it.
            """,
            hosts: ["chat": Defaults.languageModel],
            tools: ["parlor.schema.*", "parlor.harnesses.*", "parlor.hosts.list", "parlor.hosts.read", "parlor.models.list",
                    "parlor.agents.templates", "parlor.agents.validate", "parlor.agents.try", "parlor.agents.create",
                    "parlor.agents.update", "parlor.agents.read", "parlor.agents.list", "parlor.tools.list", "image.generate"],
            color: 4, config: ["context": .int(16384), "compact_at": .double(0.7), "temperature": .double(0.4), "thinking": .bool(false)], tier: .configure)
    }
}
