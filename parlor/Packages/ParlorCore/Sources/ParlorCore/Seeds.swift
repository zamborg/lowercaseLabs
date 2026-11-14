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

    static let moduleGuide = """
    # Writing a parlor module

    A module is a small program that gives agents new tools. It is an MCP server over stdio, in a folder here:

    ```
    modules/<name>/module.json     the manifest (required)
    modules/<name>/server.py       your code
    modules/<name>/settings.json   values the user chose (written by the app; you only read them)
    ```

    The folder name and `"name"` must match and use letters, digits, `-` or `_`. The app lists the module in
    System. **It does not run until the user approves it**, and asks again if `module.json` changes.

    ## module.json

    ```json
    { "version": 1, "name": "dice", "description": "Roll dice with notation like 2d6+1.",
      "command": "python3", "args": ["server.py"],
      "needs": { "network": false, "files": "none", "tier": "none" },
      "settings": { "type": "object", "properties": { "maxDice": { "type": "integer", "default": 20, "description": "Most dice per roll" } } },
      "lifecycle": "onDemand" }
    ```

    - `needs` is a declaration the user reads before approving: `network` (true/false), `files` (`none` or a folder
      you will touch), `tier` (`none`, `read`, `act`, `configure`: what you may do to the parlor app itself).
    - `settings` is a JSON Schema; the app shows a form for it. Read the values from the environment variable
      `PARLOR_MODULE_SETTINGS` (a JSON string). Defaults are already filled in.
    - `lifecycle` is `always` or `onDemand` (start on first use, stop when idle).
    - The program runs with its folder as the working directory.

    ## The protocol, with no libraries

    Read one JSON-RPC message per line on stdin, write one per line on stdout. Print nothing else on stdout; use
    stderr for logs (the app keeps them in `logs/module-<name>.log`). Answer `initialize`, `tools/list`, `tools/call`
    and `ping`. Do not answer notifications (messages without an `id`).

    ```python
    import sys, json, os, random, re

    settings = json.loads(os.environ.get("PARLOR_MODULE_SETTINGS", "{}"))

    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\\n")
        sys.stdout.flush()

    TOOLS = [{
        "name": "roll",
        "description": "Roll dice, e.g. 2d6+1.",
        "inputSchema": {"type": "object", "properties": {"notation": {"type": "string"}}, "required": ["notation"]},
    }]

    def roll(args):
        m = re.fullmatch(r"(\\d+)d(\\d+)([+-]\\d+)?", args.get("notation", "").replace(" ", ""))
        if not m:
            return "Use notation like 2d6+1.", True
        n, sides, bonus = int(m[1]), int(m[2]), int(m[3] or 0)
        if n > settings.get("maxDice", 20):
            return "Too many dice.", True
        rolls = [random.randint(1, sides) for _ in range(n)]
        return f"{rolls} {bonus:+d} = {sum(rolls) + bonus}", False

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            continue
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2025-03-26",
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "dice", "version": "1"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            text, is_error = roll(msg["params"].get("arguments", {}))
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": is_error}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
    ```

    Tools appear to agents as `<name>.<tool>` (here `dice.roll`). An agent needs `dice.*` (or `*`) in its `tools`.

    ## Calling back into parlor (optional)

    If `needs.tier` is above `none`, the app sets `PARLOR_URL` and `PARLOR_TOKEN`. POST JSON-RPC (`tools/list`,
    `tools/call`) to `PARLOR_URL` with the header `Authorization: Bearer $PARLOR_TOKEN`. You see only the verbs your
    tier allows, and destructive ones ask the user.

    ## Checklist
    1. `module.json` parses, `name` matches the folder.
    2. `python3 server.py` answers a piped `tools/list` line.
    3. Nothing but protocol lines on stdout.
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
