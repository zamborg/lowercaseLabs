import XCTest

final class ParlorUITests: XCTestCase {
    var app: XCUIApplication!
    var home: URL!

    override func setUpWithError() throws {
        continueAfterFailure = false
        home = FileManager.default.temporaryDirectory.appendingPathComponent("parlor-ui-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        app = XCUIApplication()
        app.launchEnvironment["PARLOR_HOME"] = home.path
        app.launchEnvironment["PARLOR_DEMO"] = "1"
    }

    func testCreateChatPruneRelaunch() throws {
        app.launch()
        // Seeded agents appear in the sidebar.
        XCTAssertTrue(app.staticTexts["Mabel"].firstMatch.waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Pip"].firstMatch.exists)

        // Chat with Mabel.
        app.staticTexts["Mabel"].firstMatch.click()
        let composer = app.textFields["composer"]
        XCTAssertTrue(composer.waitForExistence(timeout: 5))
        composer.click()
        composer.typeText("hello there\n")
        XCTAssertTrue(app.staticTexts["(demo) You said: hello there"].waitForExistence(timeout: 10), "reply streams in")

        // Edit the agent: rename Pip → the file on disk follows.
        app.staticTexts["Pip"].firstMatch.click()
        app.radioButtons["Agent"].firstMatch.click()
        let name = app.textFields["agent-name"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        name.click()
        name.typeKey("a", modifierFlags: .command)
        name.typeText("Pippa")
        let file = home.appendingPathComponent("agents/pip.md")
        let deadline = Date().addingTimeInterval(5)
        var onDisk = ""
        while Date() < deadline {
            onDisk = (try? String(contentsOf: file, encoding: .utf8)) ?? ""
            if onDisk.contains("name: Pippa") { break }
            Thread.sleep(forTimeInterval: 0.2)
        }
        XCTAssertTrue(onDisk.contains("name: Pippa"), "editor saves to agents/pip.md, got: \(onDisk.prefix(80))")

        // Prune: back to Mabel, select the user message, delete it.
        app.staticTexts["Mabel"].firstMatch.click()
        app.radioButtons["Chat"].firstMatch.click()
        app.menuButtons["prune-menu"].firstMatch.click()
        app.menuItems["Select Messages…"].click()
        app.staticTexts["hello there"].firstMatch.click()
        app.buttons["delete-selected"].click()
        XCTAssertFalse(app.staticTexts["hello there"].exists, "message deleted")

        // Relaunch: chat and agent edits survive.
        app.terminate()
        app.launch()
        XCTAssertTrue(app.staticTexts["Pippa"].firstMatch.waitForExistence(timeout: 15))
    }

    func testRoomRoundRobin() throws {
        let dir = home.appendingPathComponent("chats/duo-ab12")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let json = #"{"version":1,"title":"Duo","participants":[{"agent":"mabel"},{"agent":"pip"},{"user":true}],"scheduler":{"kind":"roundRobin","maxTurns":2}}"#
        try json.write(to: dir.appendingPathComponent("conversation.json"), atomically: true, encoding: .utf8)
        app.launchEnvironment["PARLOR_ROUTE"] = "room:duo-ab12"
        app.launch()
        let composer = app.textFields["room-composer"]
        XCTAssertTrue(composer.waitForExistence(timeout: 15))
        composer.click()
        composer.typeText("hi both\n")
        let reply = app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH '(demo)'")).firstMatch
        if !reply.waitForExistence(timeout: 15) { print(app.debugDescription) }
        XCTAssertTrue(reply.exists)
        // Two agents answered: both names appear as speaker labels.
        let deadline = Date().addingTimeInterval(15)
        while Date() < deadline, app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH '(demo)'")).count < 2 { Thread.sleep(forTimeInterval: 0.3) }
        XCTAssertGreaterThanOrEqual(app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH '(demo)'")).count, 2)
        let shot = XCUIScreen.main.screenshot()
        let att = XCTAttachment(screenshot: shot); att.lifetime = .keepAlways; add(att)
    }
}
