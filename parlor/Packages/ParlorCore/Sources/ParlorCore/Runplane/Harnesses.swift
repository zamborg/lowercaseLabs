import Foundation

/// A program that runs an agent's turns (`harnesses/<id>/harness.json` beside its script). The manifest declares the config
/// schema (rendered as the editor form), starter templates and the actions (buttons) the UI shows.
public struct HarnessManifest: Noun, Equatable {
    public static let kind = "harness"

    public struct Action: Codable, Equatable, Sendable, Identifiable {
        public var id: String
        public var title: String
        public var icon: String?
        /// `composer` (by the message box), `message` (a reply's menu) or `settings` (the agent editor).
        public var placement: String
        /// A host need (`image`) the agent must have, or the button is hidden.
        public var needs: String?
        public var hint: String?
        public var args: JSON?
        public init(id: String, title: String, icon: String? = nil, placement: String = "composer", needs: String? = nil, hint: String? = nil, args: JSON? = nil) {
            self.id = id; self.title = title; self.icon = icon; self.placement = placement; self.needs = needs; self.hint = hint; self.args = args
        }
        public init(from d: Decoder) throws {
            let c = try d.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            title = try c.decodeIfPresent(String.self, forKey: .title) ?? id
            icon = try c.decodeIfPresent(String.self, forKey: .icon)
            placement = try c.decodeIfPresent(String.self, forKey: .placement) ?? "composer"
            needs = try c.decodeIfPresent(String.self, forKey: .needs)
            hint = try c.decodeIfPresent(String.self, forKey: .hint)
            args = try c.decodeIfPresent(JSON.self, forKey: .args)
        }
    }

    public struct Template: Codable, Equatable, Sendable, Identifiable {
        public var id: String
        public var title: String
        public var description: String
        /// `{name}` is replaced by the agent's name.
        public var persona: String
        public var tools: [String]
        public var config: [String: JSON]
        public var group: String?
        /// Gets the default image host when one exists.
        public var image: Bool
        public init(from d: Decoder) throws {
            let c = try d.container(keyedBy: CodingKeys.self)
            id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
            title = try c.decodeIfPresent(String.self, forKey: .title) ?? id
            description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
            persona = try c.decodeIfPresent(String.self, forKey: .persona) ?? "You are {name}."
            tools = try c.decodeIfPresent([String].self, forKey: .tools) ?? []
            config = try c.decodeIfPresent([String: JSON].self, forKey: .config) ?? [:]
            group = try c.decodeIfPresent(String.self, forKey: .group)
            image = try c.decodeIfPresent(Bool.self, forKey: .image) ?? false
        }
    }

    public var version: Int = 1
    public var id: String
    public var title: String
    public var description: String = ""
    public var command: String
    public var args: [String] = []
    /// What the agent must have: `{"chat": "required", "image": "optional"}`.
    public var needs: [String: String] = ["chat": "required"]
    public var config: JSON?
    public var templates: [Template] = []
    public var actions: [Action] = []
    /// The manifest's folder, once loaded.
    public var directory: URL?

    enum CodingKeys: String, CodingKey { case version, id, title, description, command, args, needs, config, templates, actions }

    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
        id = try c.decode(String.self, forKey: .id)
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? id
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        command = try c.decode(String.self, forKey: .command)
        args = try c.decodeIfPresent([String].self, forKey: .args) ?? []
        needs = try c.decodeIfPresent([String: String].self, forKey: .needs) ?? ["chat": "required"]
        config = try c.decodeIfPresent(JSON.self, forKey: .config)
        templates = try c.decodeIfPresent([Template].self, forKey: .templates) ?? []
        actions = try c.decodeIfPresent([Action].self, forKey: .actions) ?? []
    }

    public static func check(_ d: [String: JSON]) -> [Issue] {
        var out: [Issue] = []
        if case .array(let acts)? = d["actions"] {
            var seen = Set<String>()
            for (i, a) in acts.enumerated() {
                guard let id = a.objectValue?["id"]?.stringValue else { continue }
                if !seen.insert(id).inserted { out.append(Issue(path: "actions[\(i)].id", message: "duplicate action id \"\(id)\"")) }
                if let p = a.objectValue?["placement"]?.stringValue, !["composer", "message", "settings"].contains(p) {
                    out.append(Issue(path: "actions[\(i)].placement", message: "unknown placement \"\(p)\"", hint: "composer, message or settings"))
                }
            }
        }
        return out
    }

    /// The shell line that starts this harness. The script is looked up in the manifest's folder.
    public var launchCommand: String {
        func q(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: #"'\''"#) + "'" }
        let dir = directory
        let parts = [command] + args.map { a -> String in
            if let dir, FileManager.default.fileExists(atPath: dir.appendingPathComponent(a).path) { return q(dir.appendingPathComponent(a).path) }
            return q(a)
        }
        return parts.enumerated().map { $0.offset == 0 ? $0.element : $0.element }.joined(separator: " ")
    }

    /// Config defaults from the schema.
    public var configDefaults: [String: JSON] { SchemaCheck.defaults(config) }

    /// Whether an action should be offered to an agent that has these host needs.
    public func available(_ action: Action, hosts: [String: String]) -> Bool {
        guard let need = action.needs else { return true }
        return !(hosts[need] ?? "").isEmpty
    }
}

// MARK: - the shipped harnesses

public enum HarnessSeeds {
    public static var marker: String { HarnessPython.marker }

    static let chatManifest = """
    { "managed": "\(HarnessPython.marker)", "version": 1, "id": "chat", "title": "Chat",
      "description": "An assistant with tools, memory and a rolling summary of long chats.",
      "command": "python3", "args": ["chat.py"],
      "needs": { "chat": "required" },
      "config": { "type": "object", "properties": {
          "temperature": { "type": "number", "minimum": 0, "maximum": 2, "description": "blank uses the host's default" },
          "top_p": { "type": "number", "minimum": 0, "maximum": 1, "description": "blank uses the host's default" },
          "repetition_penalty": { "type": "number", "minimum": 0.5, "maximum": 2, "description": "blank uses the host's default" },
          "max_reply_tokens": { "type": "integer", "minimum": 0, "description": "0 means no cap; blank uses the host's default" },
          "thinking": { "type": "boolean", "default": false, "description": "let the model think before answering" },
          "context": { "type": "integer", "default": 16384, "minimum": 512, "maximum": 262144, "description": "tokens of conversation kept in the prompt" },
          "compact_at": { "type": "number", "default": 0.7, "minimum": 0.1, "maximum": 1, "description": "fold older messages into a summary at this fraction of the context" },
          "max_steps": { "type": "integer", "minimum": 1, "maximum": 64, "description": "tool steps per turn; blank uses the machine default" },
          "keep_recent": { "type": "integer", "minimum": 0, "maximum": 200, "description": "messages kept verbatim when compacting" },
          "summary_tokens": { "type": "integer", "minimum": 64, "maximum": 8192, "description": "length of the rolling summary" },
          "tool_result_chars": { "type": "integer", "minimum": 100, "maximum": 200000, "description": "longest tool result kept whole" },
          "tool_timeout": { "type": "integer", "minimum": 1, "maximum": 86400, "description": "seconds a tool may run" } } },
      "actions": [
        { "id": "compact", "title": "Compact now", "icon": "arrow.down.right.and.arrow.up.left", "placement": "settings",
          "hint": "Fold older messages into the summary now" } ],
      "templates": [
        { "id": "assistant", "title": "Assistant", "description": "A helpful personal assistant with every tool.",
          "persona": "You are {name}, a helpful personal assistant. Be concise and direct.", "tools": ["*"], "config": { "temperature": 0.5 }, "image": true },
        { "id": "researcher", "title": "Researcher", "description": "Looks things up with its tools before answering, and says how sure it is.",
          "persona": "You are {name}, a careful researcher. Use your tools to gather facts before you answer. Say what you found, which tool it came from and how sure you are. If you could not find something, say so rather than guess.",
          "tools": ["*"], "config": { "temperature": 0.3, "context": 32768, "thinking": true } },
        { "id": "blank", "title": "Blank", "description": "Just a name and a persona. No tools.", "persona": "You are {name}.", "tools": [] } ] }
    """

    static let characterManifest = """
    { "managed": "\(HarnessPython.marker)", "version": 1, "id": "character", "title": "Character",
      "description": "Plays a person in a story, keeps their look consistent and takes pictures when the moment calls for it.",
      "command": "python3", "args": ["character.py"],
      "needs": { "chat": "required", "image": "optional" },
      "config": { "type": "object", "properties": {
          "temperature": { "type": "number", "minimum": 0, "maximum": 2, "description": "blank uses the host's default" },
          "top_p": { "type": "number", "minimum": 0, "maximum": 1, "description": "blank uses the host's default" },
          "repetition_penalty": { "type": "number", "minimum": 0.5, "maximum": 2, "description": "blank uses the host's default" },
          "max_reply_tokens": { "type": "integer", "minimum": 16, "description": "blank uses 500" },
          "context": { "type": "integer", "default": 8192, "minimum": 512, "maximum": 262144, "description": "tokens of conversation kept in the prompt" },
          "appearance": { "type": "string", "format": "multiline", "description": "how the character looks; repeated in every picture so the face stays recognisable" },
          "scenario": { "type": "string", "format": "multiline", "description": "where and when the story is, and what is going on" },
          "user_name": { "type": "string", "description": "what the character calls the person talking to them" },
          "images": { "enum": ["auto", "ask", "off"], "default": "auto", "description": "auto: when the moment calls for it; ask: only when asked; off" },
          "image_every": { "type": "integer", "default": 4, "minimum": 1, "description": "fewest replies between unprompted pictures" },
          "image_style": { "type": "string", "description": "the look of the pictures (blank: a natural photograph)" },
          "image_prompt_style": { "enum": ["natural", "tags"], "description": "tags for Illustrious/Pony/SDXL checkpoints; blank picks from the image model" },
          "image_detail": { "enum": ["fast", "detailed"], "description": "blank: detailed for tag models, fast otherwise" },
          "seed": { "type": "integer", "minimum": 0, "description": "the same seed keeps the character's pictures more alike; blank derives one from the id" } } },
      "actions": [
        { "id": "image", "title": "Gen image", "icon": "photo", "placement": "composer", "needs": "image", "hint": "Picture the current scene" },
        { "id": "selfie", "title": "Selfie", "icon": "camera", "placement": "composer", "needs": "image", "hint": "The character takes a selfie" } ],
      "templates": [
        { "id": "character", "title": "Character", "description": "A person to talk with, with pictures.",
          "persona": "You are {name}. Describe who this character is and how they speak.", "tools": ["image.generate"], "image": true,
          "group": "Characters", "config": { "temperature": 0.9, "context": 8192, "images": "auto" } },
        { "id": "scene", "title": "Scene", "description": "A character in a scenario, keeping their look consistent.",
          "persona": "You are {name}. Describe who this character is: their history, voice, habits and what they want.", "tools": ["image.generate"], "image": true,
          "group": "Characters",
          "config": { "temperature": 0.9, "context": 8192, "images": "auto",
                      "appearance": "Describe how they look: age, hair, eyes, build, usual clothes, one distinctive detail.",
                      "scenario": "Where and when this is, and what is going on." } } ] }
    """

    static let guide = "<!-- \(HarnessPython.marker) -->\n" + #"""
    # Harnesses

    A harness is a program that runs an agent's turns. Every agent has one (`chat` unless it says otherwise). The app owns the
    log, the tools and the tiers; the harness owns what is said and when to call tools or take a picture.

    Drop a folder here and it appears in the app within a second, no restart. If it does not appear, or shows a problem,
    `parlor.harnesses.list` lists it with `ok: false` and the exact issues. Fixing the file clears them.

    A harness is a folder:

        harnesses/<id>/harness.json     the manifest
        harnesses/<id>/<script>         the program (any language that can read and write JSON lines)

    `parlor_harness.py` (one level up, shared) is the Python helper: the protocol loop, the gateway client (streaming, cancel,
    priorities), tool calls, history and compaction helpers. `chat/` and `character/` are small programs on top of it; copy a
    folder to make your own. Folders you write are never overwritten; the shipped ones carry a `managed` marker and are
    refreshed when the app updates.

    ## harness.json

        { "version": 1, "id": "dice", "title": "Dice narrator",
          "command": "python3", "args": ["dice.py"],
          "needs": { "chat": "required", "image": "optional" },
          "config": { "type": "object", "properties": {
              "sides": { "type": "integer", "default": 6, "minimum": 2, "description": "faces on the die" } } },
          "templates": [ { "id": "dice", "title": "Dice narrator", "persona": "You are {name}.", "tools": [], "config": {} } ],
          "actions": [ { "id": "roll", "title": "Roll", "icon": "dice", "placement": "composer" } ] }

    - `needs` says which hosts the agent must have (`chat`, `image`). `required` is checked when an agent is saved.
    - `config` is a JSON-Schema subset: `type` (string, number, integer, boolean, array of string), `enum`, `default`,
      `minimum`/`maximum`, `format: "multiline"`, `description`. The agent editor renders it as a form. Unknown keys in an
      agent's config are kept and passed through, with a warning.
    - Sampling settings (`temperature`, `top_p`, `repetition_penalty`, `max_reply_tokens`) get no schema default, so a blank
      value lets the host's own defaults apply.
    - `templates` appear under New agent. `{name}` in a persona is replaced by the agent's name.
    - `actions` become buttons: `composer` (by the message box), `message` (a reply's menu) or `settings` (the agent editor).
      `needs: "image"` hides the button when the agent has no image host. Clicking one sends a normal `turn.start` with
      `action: {id, args, messageID}` and no user line.

    ## Protocol v2 (JSON lines, one object per line)

    The program runs with its working directory set to `workspaces/<agent>/` and these variables: `PARLOR_GATEWAY_URL` (the
    OpenAI-compatible base, `.../v1`), `PARLOR_TOKEN` (scoped to the agent's hosts), `PARLOR_AGENT`, `PARLOR_WORKSPACE`,
    `PARLOR_HOME`, and `PARLOR_MCP_URL` for the control server.

    app to harness, on stdin:

        {"type":"turn.start","version":2,"turn":"<id>","conversation":"<id>",
         "agent":{"id","name","description","persona","tools","tier"},
         "hosts":{"chat":{"model":"stheno","baseURL":"http://127.0.0.1:11501/v1"},"image":{...}},
         "config":{...},"settings":{...machine defaults...},"participants":[...],"transcript":[...log lines...],
         "summary":{"through":"<message id>","text":"..."} or null,"lastPromptTokens":0,
         "tools":[{"name","description","schema"}],"memory":"","action":null or {"id","args","messageID"}}
        {"type":"tool.result","turn":"<id>","callID":"...","text":"...","isError":false,"images":[]}
        {"type":"turn.cancel","turn":"<id>"}

    harness to app, on stdout:

        {"type":"text.delta","turn":"<id>","text":"the whole visible text so far"}
        {"type":"status","turn":"<id>","text":"Searching"}
        {"type":"message","turn":"<id>","role":"assistant","text":"...","toolCalls":[{"id","name","arguments":{}}]}
        {"type":"tool.call","turn":"<id>","callID":"...","name":"server__tool","arguments":{}}
        {"type":"summary","turn":"<id>","through":"<message id>","text":"...","folded":12}
        {"type":"stats","turn":"<id>","promptTokens":0,"completionTokens":0,"tokensPerSecond":0}
        {"type":"turn.done","turn":"<id>"}
        {"type":"error","turn":"<id>","message":"..."}

    Send a `message` for every model step (with `toolCalls` if it called tools), then a `tool.call` per call and wait for its
    `tool.result`. The app checks the agent's tool list and tier, runs the tool and records the result itself; do not send
    tool messages. A `summary` makes the app write `summary.md` and note the fold in the chat. End every turn with `turn.done`
    (or `error`). Print diagnostics to stderr: they land in `logs/harness-<agent>.log`.

    On `turn.cancel` stop at once and close any open gateway connection (the gateway stops generating when the client goes).
    """#

    /// Write the shipped harnesses. A file is rewritten only when it carries the marker (or is missing) and differs.
    public static func seed(home: ParlorHome) {
        let fm = FileManager.default
        let dir = home.harnessesDir
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        func put(_ url: URL, _ text: String, marked: Bool = true) {
            try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let current = try? String(contentsOf: url, encoding: .utf8)
            if current == nil || (current!.contains("parlor-harnesses-") && current != text) || (marked == false && current != text && current!.contains("parlor-harnesses-v1")) {
                try? text.write(to: url, atomically: true, encoding: .utf8)
            }
        }
        put(dir.appendingPathComponent("CLAUDE.md"), guide)
        put(dir.appendingPathComponent("parlor_harness.py"), HarnessPython.helper)
        put(dir.appendingPathComponent("chat/harness.json"), chatManifest)
        put(dir.appendingPathComponent("chat/chat.py"), HarnessPython.chat)
        put(dir.appendingPathComponent("character/harness.json"), characterManifest)
        put(dir.appendingPathComponent("character/character.py"), HarnessPython.character)
    }
}
