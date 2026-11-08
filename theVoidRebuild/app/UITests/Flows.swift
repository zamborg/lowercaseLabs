import XCTest

/// Simulator flows. Real UI and real SwiftData storage; enrichment is seeded because MLX and SpeechAnalyzer
/// can't run in the simulator. Capture/recording flows live in the Mac harness and on the phone.
final class Flows: XCTestCase {
    override func setUp() { continueAfterFailure = false }

    private func launch(_ args: [String]) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = args
        app.launch()
        return app
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        if let dir = ProcessInfo.processInfo.environment["VOID_SHOTS"] {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? png.write(to: URL(fileURLWithPath: "\(dir)/\(name).png"))
        }
        let a = XCTAttachment(data: png, uniformTypeIdentifier: "public.png"); a.name = name; a.lifetime = .keepAlways
        add(a)
    }

    // F4 edit, F8 delete (and the privacy toggle on the card), through the real Void tab.
    func testVoidListEditShareDelete() {
        let app = launch(["-seed", "-tab", "1"])
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 10))
        XCTAssertEqual(app.cells.count, 4)
        shot("void-list")

        app.cells.firstMatch.tap()                                          // newest: "Big launch day", shared
        let title = app.textViews["titleField"].exists ? app.textViews["titleField"] : app.textFields["titleField"]
        XCTAssertTrue(title.waitForExistence(timeout: 5))
        XCTAssertEqual(title.value as? String, "Big launch day")
        title.tap()
        title.press(forDuration: 1.0)
        app.menuItems["Select All"].tap()
        title.typeText("Renamed by me\n")
        shot("card")

        app.buttons["tag_low_unpleasant"].tap()                             // add a second tag
        let toggle = app.switches.firstMatch
        XCTAssertEqual(toggle.value as? String, "1", "seeded entry is shared")
        toggle.tap()
        XCTAssertEqual(toggle.value as? String, "0", "flipped to private")

        app.navigationBars.buttons.firstMatch.tap()                         // back to the list
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'Renamed by me'")).firstMatch.waitForExistence(timeout: 5),
                      "title edit did not persist to the list")

        app.cells.firstMatch.tap()
        XCTAssertEqual(app.switches.firstMatch.value as? String, "0", "visibility did not persist")
        app.buttons["Delete message"].tap()
        app.sheets.buttons["Delete"].firstMatch.tap()
        XCTAssertTrue(app.cells.firstMatch.waitForExistence(timeout: 5))
        XCTAssertEqual(app.cells.count, 3, "delete did not remove the entry")
        shot("void-after-delete")
    }

    // D5: private on first use, then remembers the last choice across launches.
    func testVisibilityDefaultsPrivateThenRemembers() {
        var app = launch([])
        let toggle = app.buttons["visibilityToggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 10))
        XCTAssertEqual(toggle.label, "private", "first use must be private")
        toggle.tap()
        XCTAssertEqual(app.buttons["visibilityToggle"].label, "shared with friends")
        shot("capture-shared")
        app.terminate()
        app = launch([])
        XCTAssertEqual(app.buttons["visibilityToggle"].label, "shared with friends", "last choice not remembered")
        app.buttons["visibilityToggle"].tap()   // leave the simulator in the default state
        XCTAssertEqual(app.buttons["visibilityToggle"].label, "private")
    }

    // F7 (display side): friends' dots appear with tags and time only, never content.
    func testSocialShowsDotsNotContent() {
        let app = launch(["-seed", "-tab", "2"])
        XCTAssertTrue(app.otherElements["friendRow"].firstMatch.waitForExistence(timeout: 10)
                      || app.staticTexts["Maya"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Maya"].exists && app.staticTexts["Dev"].exists)
        shot("social")
        XCTAssertFalse(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'shipped it'")).firstMatch.exists,
                       "transcript text must never appear in Social")
    }
}
