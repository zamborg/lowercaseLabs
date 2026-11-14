import Foundation

enum Seeds {
    static let folderGuide = """
    # parlor home

    This folder is the whole state of the parlor Mac app. Edit files here and the running app updates live.

    ## Layout
    - `agents/<id>.md` one agent per file. Frontmatter + persona body.
    - `chats/<id>/log.jsonl` the single chat for that agent (append only; one JSON message per line).
    - `chats/<id>/summary.md` rolling summary written by compaction.
    - `media/YYYY-MM/` generated images, referenced from chat logs by relative path.
    - `models/<org>/<name>/` downloaded MLX weights (managed by the Models screen).
    - `chats/<id>/conversation.json` optional; makes the folder a room (several agents, a scheduler).
    - `mcp.json` mounted MCP servers, same format as Claude Desktop. Optional per server: `"lifecycle": "onDemand"`
      and `"idleMinutes": 10` start it on first use and stop it when idle.
    - `modules/<name>/` small tools you or a model write; see `modules/CLAUDE.md`. They run only after you approve them in System.
    - `settings.json` machine settings, access tokens (hashes only) and module approvals.
    - `logs/` one log per service (`mcp-<name>.log`, `image.log`, `module-<name>.log`), rotated at 1 MB.

    ## Agent file
    ```
    ---
    name: Mabel
    description: Chief of staff.
    kind: assistant            # assistant | character
    model: mlx-community/Qwen3-4B-4bit
    image_model: mlx-community/Z-Image-Turbo-bf16   # optional
    tools:                     # server.tool, server.*, or *
      - zass.*
      - image.generate
    context: 16384             # token budget
    compact_at: 0.7            # compact when prompt passes this share of context
    temperature: 0.6
    thinking: false
    color: 0                   # avatar palette 0-5
    ---
    The persona goes here, as the system prompt.
    ```
    The file name (without `.md`) is the agent id. To add an agent, write a new file. To remove one, delete it.

    Optional agent key `tier: none|read|act|configure` lets the agent use the app's own `parlor.*` tools
    (list or change agents, post in rooms, change settings). It defaults to `none`. Add `parlor.*` to `tools` too.

    ## Rules
    - One chat per agent. Do not create extra chat folders.
    - Do not edit `log.jsonl` while the app is generating in that chat.
    - Models must already be downloaded (Models screen) before an agent can use them.
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
            kind: .assistant, model: Defaults.languageModel, imageModel: Defaults.imageModel,
            tools: ["*"], contextTokens: 16384, compactAt: 0.7, temperature: 0.5, thinking: false, color: 0)
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
            kind: .character, model: Defaults.characterModel, imageModel: Defaults.imageModel,
            tools: ["image.generate"], contextTokens: 8192, compactAt: 0.7, temperature: 0.9, thinking: false, color: 1)
    }
}
