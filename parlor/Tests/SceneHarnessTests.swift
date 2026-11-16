import XCTest
@testable import ParlorCore

final class SceneHarnessTests: XCTestCase {
    @MainActor func testSceneTemplateUsesCharacterHarnessAndSeedsIt() async throws {
        let home = tempHome()
        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        let file = home.harnessesDir.appendingPathComponent("character/character.py")
        let src = try String(contentsOf: file, encoding: .utf8)
        XCTAssertTrue(src.contains(HarnessPython.marker) && src.contains("art director"))

        for f in ["parlor_harness.py", "chat/chat.py", "character/character.py"] {
            let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
            p.arguments = ["-m", "py_compile", home.harnessesDir.appendingPathComponent(f).path]
            p.standardError = FileHandle.nullDevice
            try p.run(); p.waitUntilExit()
            XCTAssertEqual(p.terminationStatus, 0, f)
        }

        let m = try XCTUnwrap(e.harnessList.first { $0.id == "character" })
        let t = try XCTUnwrap(m.templates.first { $0.id == "scene" })
        let a = e.create(harness: "character", template: t, name: "Wren")
        XCTAssertEqual(a.harnessID, "character")
        XCTAssertEqual(a.group, "Characters")
        XCTAssertNotNil(a.config["appearance"]); XCTAssertNotNil(a.config["images"])
        let json = try String(contentsOf: home.agentsDir.appendingPathComponent("wren/agent.json"), encoding: .utf8)
        XCTAssertTrue(json.contains("appearance") && json.contains("\"harness\""))
    }
}
