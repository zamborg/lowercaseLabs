import ControllrCore
import XCTest
@testable import Controllr

final class ProviderSettingsTests: XCTestCase {
    func testSystemInstructionsPersistAndRestoreDefault() throws {
        let suite = "controllr-tests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        XCTAssertNil(Settings.systemInstructions(defaults: defaults))
        let instructions = "Keep answers concise.\nAsk only essential follow-ups."
        Settings.setSystemInstructions(instructions, defaults: defaults)
        XCTAssertEqual(Settings.systemInstructions(defaults: defaults), instructions)

        Settings.setSystemInstructions(Prompt.defaultInstructions, defaults: defaults)
        XCTAssertNil(Settings.systemInstructions(defaults: defaults))
        Settings.setSystemInstructions(instructions, defaults: defaults)
        Settings.setSystemInstructions("  \n  ", defaults: defaults)
        XCTAssertNil(Settings.systemInstructions(defaults: defaults))
    }

    func testProviderModelsStayIndependentAndKeepExistingSettings() throws {
        let suite = "controllr-tests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        defaults.set("gpt-6-luna", forKey: "openaiModel")
        defaults.set("claude-sonnet", forKey: "anthropicModel")
        XCTAssertEqual(Settings.model(for: .openai, defaults: defaults), "gpt-6-luna")
        XCTAssertEqual(Settings.model(for: .anthropic, defaults: defaults), "claude-sonnet")
        XCTAssertEqual(Settings.model(for: .openrouter, defaults: defaults), "")

        Settings.setModel("anthropic/claude-sonnet", for: .openrouter, defaults: defaults)
        XCTAssertEqual(Settings.model(for: .openrouter, defaults: defaults), "anthropic/claude-sonnet")
        XCTAssertEqual(Settings.model(for: .anthropic, defaults: defaults), "claude-sonnet")
        XCTAssertEqual(Settings.model(for: .openai, defaults: defaults), "gpt-6-luna")
    }
}
