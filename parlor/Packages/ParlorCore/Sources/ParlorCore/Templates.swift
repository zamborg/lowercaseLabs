import Foundation

/// A starting point for a new agent. There is one kind of agent; a template only fills in the fields.
public struct AgentTemplate: Identifiable, Sendable {
    public var id: String
    public var title: String
    public var blurb: String
    public var kind: Agent.Kind
    public var group: String?
    public var persona: @Sendable (String) -> String
    public var languageModel: String
    public var tools: [String]
    public var contextTokens: Int
    public var temperature: Double
    public var thinking: Bool
    public var withImage: Bool
    /// Runs the agent's turns in the sample external harness.
    public var scripted: Bool

    public init(id: String, title: String, blurb: String, kind: Agent.Kind = .assistant, group: String? = nil,
                persona: @escaping @Sendable (String) -> String, languageModel: String = Defaults.languageModel,
                tools: [String] = ["*"], contextTokens: Int = 16384, temperature: Double = 0.5,
                thinking: Bool = false, withImage: Bool = true, scripted: Bool = false) {
        self.id = id; self.title = title; self.blurb = blurb; self.kind = kind; self.group = group; self.persona = persona
        self.languageModel = languageModel; self.tools = tools; self.contextTokens = contextTokens
        self.temperature = temperature; self.thinking = thinking; self.withImage = withImage; self.scripted = scripted
    }
}

public enum AgentTemplates {
    public static let all: [AgentTemplate] = [
        AgentTemplate(id: "assistant", title: "Assistant", blurb: "A helpful personal assistant with every tool.",
                      persona: { "You are \($0), a helpful personal assistant. Be concise and direct." }),
        AgentTemplate(id: "character", title: "Character", blurb: "A persona to talk with, with the image tool.", kind: .character,
                      persona: { "You are \($0). Describe who this character is and how they speak." },
                      languageModel: Defaults.characterModel, tools: ["image.generate"], contextTokens: 8192, temperature: 0.9),
        AgentTemplate(id: "researcher", title: "Researcher", blurb: "Looks things up with its tools before answering, and says how sure it is.",
                      persona: { "You are \($0), a careful researcher. Use your tools to gather facts before you answer. Say what you found, which tool it came from and how sure you are. If you could not find something, say so rather than guess." },
                      contextTokens: 32768, temperature: 0.3, thinking: true, withImage: false),
        AgentTemplate(id: "scripted", title: "Scripted agent", blurb: "Runs its turns in your own program (the sample harness) instead of the built-in loop.",
                      group: "Scripted", persona: { "You are \($0), a helpful assistant. Use your tools when they apply. Be brief." },
                      withImage: false, scripted: true),
        AgentTemplate(id: "blank", title: "Blank", blurb: "Just a name and a persona. No tools.",
                      persona: { "You are \($0)." }, tools: [], withImage: false),
    ]

    public static func template(_ id: String) -> AgentTemplate? { all.first { $0.id == id } }
}
