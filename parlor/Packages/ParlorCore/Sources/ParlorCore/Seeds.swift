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
    - `mcp.json` mounted MCP servers, same format as Claude Desktop.

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

    ## Rules
    - One chat per agent. Do not create extra chat folders.
    - Do not edit `log.jsonl` while the app is generating in that chat.
    - Models must already be downloaded (Models screen) before an agent can use them.
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
