import XCTest
@testable import ParlorCore

final class SceneHarnessTests: XCTestCase {
    @MainActor func testSceneTemplateUsesCharacterHarnessAndSeedsIt() async throws {
        let home = tempHome()
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        let file = home.harnessesDir.appendingPathComponent("character.py")
        let src = try String(contentsOf: file, encoding: .utf8)
        XCTAssertTrue(src.contains("parlor-harnesses-v1") && src.contains("art director"))

        // The seeded script is valid Python.
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); p.arguments = ["-m", "py_compile", file.path]
        p.standardError = FileHandle.nullDevice
        try p.run(); p.waitUntilExit()
        XCTAssertEqual(p.terminationStatus, 0)

        let a = e.create(template: try XCTUnwrap(AgentTemplates.template("scene")), name: "Wren")
        XCTAssertTrue(a.harness?.hasSuffix("character.py'") == true, a.harness ?? "nil")
        XCTAssertEqual(a.kind, .character)
        XCTAssertTrue(a.extra.contains { $0.key == "appearance" })
        XCTAssertTrue(a.extra.contains { $0.key == "images" })
        // Saved to disk, so the front matter round-trips.
        let reread = try String(contentsOf: home.agentsDir.appendingPathComponent("wren.md"), encoding: .utf8)
        XCTAssertTrue(reread.contains("appearance:") && reread.contains("harness:"))
    }
}
